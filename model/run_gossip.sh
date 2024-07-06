# 我租的机器只有一个卡，所以统一设置 CUDA_VISIBLE_DEVICES=0
# ========================= Pattern-based =========================

# BiLSTM
CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'gossip' \
--use_preference_map False --use_pattern_based_model True --use_fact_based_model False \
--pattern_based_model 'BiLSTM' \
--lr 1e-4 --batch_size 4 --epochs 20 \
--save 'ckpts/BiLSTM'

# BiLSTM w/ Pref-FENDs
CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'gossip' \
--use_preference_map True --use_pattern_based_model True --use_fact_based_model False \
--pattern_based_model 'BiLSTM' \
--lr 1e-4 --batch_size 4 --epochs 20 \
--save 'ckpts/BiLSTM_with_Pref-FENDs'

# **************************************** #

# EANN_Text
CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'gossip' \
--use_preference_map False --use_pattern_based_model True --use_fact_based_model False \
--pattern_based_model 'EANN_Text' \
--lr 5e-5 --batch_size 4 --epochs 20 \
--save 'ckpts/EANN_Text'

# EANN_Text w/ Pref-FENDs
CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'gossip' \
--use_preference_map True --use_pattern_based_model True --use_fact_based_model False \
--pattern_based_model 'EANN_Text' \
--lr 5e-5 --batch_size 4 --epochs 20 \
--save 'ckpts/EANN_Text_with_Pref-FENDs'


# ========================= Fact-based =========================

# DeClarE
CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'gossip' \
--use_preference_map False --use_pattern_based_model False --use_fact_based_model True \
--fact_based_model 'DeClarE' \
--lr 1e-3 --batch_size 4 --epochs 20 \
--save 'ckpts/DeClarE'

# DeClarE w/ Pref-FENDs
CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'gossip' \
--use_preference_map True --use_pattern_based_model False --use_fact_based_model True \
--fact_based_model 'DeClarE' \
--lr 1e-3 --batch_size 4 --epochs 20 \
--save 'ckpts/DeClarE_with_Pref-FENDs'

# **************************************** #

# MAC
CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'gossip' \
--use_preference_map False --use_pattern_based_model False --use_fact_based_model True \
--fact_based_model 'MAC' \
--lr 5e-5 --batch_size 4 --epochs 20 \
--save 'ckpts/MAC'

# MAC w/ Pref-FENDs
CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'gossip' \
--use_preference_map True --use_pattern_based_model False --use_fact_based_model True \
--fact_based_model 'MAC' \
--lr 5e-5 --batch_size 4 --epochs 20 \
--save 'ckpts/MAC_with_Pref-FENDs'


# ========================= Integrated Models =========================

# BiLSTM + DeClarE (last-layer concat)
CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'gossip' \
--use_preference_map False --use_pattern_based_model True --use_fact_based_model True \
--pattern_based_model 'BiLSTM' --fact_based_model 'DeClarE' \
--lr 1e-4 --batch_size 4 --epochs 20 \
--save 'ckpts/BiLSTM+DeClarE'

# BiLSTM + DeClarE (Pref-FNED)
CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'gossip' \
--use_preference_map True --use_pattern_based_model True --use_fact_based_model True \
--pattern_based_model 'BiLSTM' --fact_based_model 'DeClarE' \
--lr 1e-4 --batch_size 4 --epochs 20 \
--save 'ckpts/BiLSTM+DeClarE_with_Pref-FEND'