import os
import time
from tqdm import tqdm

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, SequentialSampler
from torch_geometric.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, lr_scheduler
from PrefFEND import PrefFEND
from evaluation import evaluate
from config import parser
from DatasetLoader import GraphDataset
from utils import pytorch_cos_sim


if __name__ == "__main__":
    # Parse command line arguments
    args = parser.parse_args()

    # If debug mode is enabled, set save path and epochs for debugging
    if args.debug:
        args.save = './ckpts/debug'
        args.epochs = 2

    # If preference map is not used, set weights for different losses
    if not args.use_preference_map:
        args.weight_of_normal_loss = 1.0
        args.weight_of_preference_loss = .0
        args.weight_of_reversed_loss = .0

    # Remove existing save directory if it exists, then create a new one
    if os.path.exists(args.save):
        os.system('rm -r {}'.format(args.save))
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    # Print experimental dataset details and start time
    print('\n{} Experimental Dataset: {} {}\n'.format(
        '=' * 20, args.dataset, '=' * 20))
    print('Start time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    print('save path: ', args.save)

    # Set device to GPU if available, otherwise use CPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load and print the model
    print('-----------------------------------------\nLoading model...\n')
    start = time.time()
    model = PrefFEND(args)
    print(model)
    print('\nLoading model time: {:.2f}s\n-----------------------------------------\n'.format(
        time.time() - start))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = AdamW(filter(lambda p: p.requires_grad,
                             model.parameters()), lr=args.lr)

    # If using mixed precision, initialize GradScaler
    if args.fp16:
        scaler = GradScaler()
    if torch.cuda.is_available():
        model = model.cuda()

    # Resume from a checkpoint if provided
    if args.resume != '':
        resume_dict = torch.load(args.resume)
        model.load_state_dict(resume_dict['state_dict'])
        optimizer.load_state_dict(resume_dict['optimizer'])
        args.start_epoch = resume_dict['epoch'] + 1

    # Load datasets for training, validation, and testing
    print('-----------------------------------------\nLoading data...\n')
    start = time.time()
    train_dataset = GraphDataset(args, 'train')
    val_dataset = GraphDataset(args, 'val')
    test_dataset = GraphDataset(args, 'test')

    # Create data loaders for datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=(torch.cuda.is_available()),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(torch.cuda.is_available()),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(torch.cuda.is_available()),
        drop_last=False,
    )

    print('\nLoading data time: {:.2f}s\n-----------------------------------------\n'.format(
        time.time() - start))

    last_epoch = args.start_epoch if args.start_epoch != 0 else -1

    # Save arguments and model structure to a file
    with open(os.path.join(args.save, 'args.txt'), 'w') as f:
        print('\n---------------------------------------------------\n')
        print('[Arguments] \n')
        for arg in vars(args):
            v = getattr(args, arg)
            s = '{}\t{}'.format(arg, v)
            f.write('{}\n'.format(s))
            print(s)
        print('\n---------------------------------------------------\n')
        f.write('\n{}\n'.format(model))

    # Training
    print('Start training...')

    # Save only the best result on validation dataset
    best_val_result = 0
    best_val_epoch = -1

    start = time.time()
    args.global_step = 0
    for epoch in range(args.start_epoch, args.epochs):
        args.current_epoch = epoch
        print('\n------------------------------------------------\n')
        print('Start Training Epoch', epoch, ':', time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        model.train()

        # Initialize training loss variables
        train_loss = 0.
        train_normal_loss = 0.
        train_reversed_loss = 0.
        train_preference_loss = 0.

        lr = optimizer.param_groups[0]['lr']
        # Training loop for each batch
        for step, (idxs, graphs_entity, graphs_pattern, graphs_others, nums_nodes, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                PrefFEND_out = model(idxs, train_loader.dataset, graphs_entity,
                                     graphs_pattern, graphs_others, nums_nodes)
                mlp_out, mlp_reversed_out, map_entity, map_pattern = PrefFEND_out[:4]

                # Calculate normal loss
                normal_loss = criterion(mlp_out, labels.long().to(args.device))
                reversed_loss = torch.tensor([0.], device=args.device)
                preference_loss = torch.tensor([0.], device=args.device)

                CEloss = normal_loss.item()

                # Calculate reversed loss if applicable
                if mlp_reversed_out is not None:
                    reversed_loss = criterion(
                        mlp_reversed_out, abs(labels - 1).long().to(args.device))

                # Calculate preference loss if applicable
                if map_entity is not None:
                    sim_matrix = pytorch_cos_sim(map_entity, map_pattern)
                    sim_matrix = sim_matrix.masked_fill(
                        torch.isnan(sim_matrix), 0.)
                    preference_loss = torch.mean(torch.diag(sim_matrix))

                # Handle event loss for pattern-based model
                if args.pattern_based_model == 'EANN_Text':
                    event_labels = [
                        train_loader.dataset.event_labels[idx.item()] for idx in idxs]
                    event_labels = torch.tensor(
                        event_labels, dtype=torch.long, device=args.device)

                    event_out, event_reversed_out = PrefFEND_out[-2:]

                    normal_event_loss = criterion(event_out, event_labels)
                    reversed_event_loss = torch.tensor(
                        [0.], device=args.device)
                    if event_reversed_out is not None:
                        reversed_event_loss = criterion(
                            event_reversed_out, event_labels)

                    normal_loss = normal_loss + args.eann_weight_of_event_loss * normal_event_loss
                    reversed_loss = reversed_loss + args.eann_weight_of_event_loss * reversed_event_loss

                # Apply weights to losses
                normal_loss *= args.weight_of_normal_loss
                reversed_loss *= args.weight_of_reversed_loss
                preference_loss *= args.weight_of_preference_loss
                loss = normal_loss + reversed_loss + preference_loss

                # Check for NaN values in losses
                if torch.any(torch.isnan(normal_loss)) or torch.any(torch.isnan(reversed_loss)) or torch.any(torch.isnan(preference_loss)):
                    print('mlp_out: ', mlp_out)
                    print('mlp_reversed_out:, ', mlp_reversed_out)
                    print('map_entity: ', map_entity)
                    print('map_pattern: ', map_pattern)
                    print('normal_loss = {:.4f}, reversed_loss = {:.4f}, preference_loss = {:.4f}\n'.format(
                        normal_loss.item(), reversed_loss.item(), preference_loss.item()))
                    exit()

                # Print loss details every 10 steps
                print_step = 10
                if step % print_step == 0:
                    print('\n\nEpoch: {}, Step: {}, CEloss = {:.4f}'.format(
                        epoch, step, CEloss))
                    print('normal_loss = {:.4f}, reversed_loss = {:.4f}, preference_loss = {:.4f}\n'.format(
                        normal_loss.item(), reversed_loss.item(), preference_loss.item()))

                    if args.pattern_based_model == 'EANN_Text':
                        print('event_loss = {:.4f}, reversed_event_loss = {:.4f}'.format(
                            normal_event_loss.item(), reversed_event_loss.item()))

                    print('mlp_out: ', mlp_out)
                    if map_entity is not None:
                        print('map_entity[0][:20] = ', map_entity[0][:20])
                        print('map_pattern[0][:20] = ', map_pattern[0][:20])
                    print()

            # Backward pass and optimization step
            if args.fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Update training loss values
            train_loss += loss.item()
            train_normal_loss += normal_loss.item()
            train_reversed_loss += reversed_loss.item()
            train_preference_loss += preference_loss.item()

            args.global_step += 1

        # Calculate average training losses
        train_losses = [train_loss, train_normal_loss,
                        train_reversed_loss, train_preference_loss]
        train_losses = [l/len(train_loader) for l in train_losses]
        train_loss, train_normal_loss, train_reversed_loss, train_preference_loss = train_losses

        # Evaluate model on validation and test datasets
        val_losses, val_result = evaluate(
            args, val_loader, model, criterion, 'val')
        test_losses, test_result = evaluate(
            args, test_loader, model, criterion, 'test')

        val_loss, val_normal_loss, val_reversed_loss, val_preference_loss = val_losses
        test_loss, test_normal_loss, test_reversed_loss, test_preference_loss = test_losses

        # Logging
        print('='*10, 'Epoch: {}/{}'.format(epoch, args.epochs),
              'lr: {}'.format(lr), '='*10)
        print('\n[Loss]\nTrain: {:.6f}\tVal: {:.6f}\tTest: {:.6f}'.format(
            train_loss, val_loss, test_loss))
        print('[Macro F1]\nVal: {:.6f}\tTest: {:.6f}\n'.format(
            val_result, test_result))
        print('*'*8, 'Normal Loss\tReversed Loss\tPreference Loss', '*'*8)
        print('\n[Train]\t{:.5f}\t{:.5f}\t{:.5f}'.format(
            train_normal_loss, train_reversed_loss, train_preference_loss))
        print('[Val]\t{:.5f}\t{:.5f}\t{:.5f}'.format(
            val_normal_loss, val_reversed_loss, val_preference_loss))
        print('[Test]\t{:.5f}\t{:.5f}\t{:.5f}\n'.format(
            test_normal_loss, test_reversed_loss, test_preference_loss))

        # Save model if validation result is the best so far
        if val_result >= best_val_result:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
                os.path.join(args.save, '{}.pt'.format(epoch))
            )

            # Remove the previous best model checkpoint
            if best_val_epoch != -1:
                os.system('rm {}'.format(os.path.join(
                    args.save, '{}.pt'.format(best_val_epoch))))

            best_val_result = val_result
            best_val_epoch = epoch

    # Print total training time and end time
    print('Training Time: {:.2f}s'.format(time.time() - start))
    print('End time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
