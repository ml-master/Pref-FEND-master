import json
import os
import spacy
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 为NER加载spaCy模型
nlp = spacy.load("en_core_web_sm")

# 定义输入和输出文件地址
# input_files = ['./GossipCop/gossipcop_v3-2_content_based_fake.json', './GossipCop/gossipcop_v3-4_story_based_fake.json']
input_files = ['./GossipCop/gossipcop_v3-4_story_based_fake.json']
output_dir = './gossip/raw'

# 创建必需的文件夹（当它们不存在时），文件夹名称与原始数据集一样
os.makedirs(os.path.join(output_dir, 'post'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'articles'), exist_ok=True)

total_data = {}
# 加载输入数据
for input_file in input_files:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        total_data.update(data)

# 做了一下采样，10条取一条。可以改成2条取一条，不过跑的时间要长一点
sampled_data = {k: v for i, (k, v) in enumerate(total_data.items()) if i % 1 == 0}

# 函数：进一步处理文本，将文本中的每个单词分类成“OTHERS”、“ENTITY”或“PATTERN”
def process_text(text):
    doc = nlp(text)
    words = []
    for token in doc:
        token_type = "OTHERS"
        if token.ent_type_:
            token_type = "ENTITY"
        elif token.is_alpha:
            token_type = "PATTERN"
        words.append((token.text, token_type))
    return words

# 为输出初始化数据结构
posts = []
articles = []

# 处理输入数据
print("Processing posts and articles...")
for key, value in tqdm(sampled_data.items(), desc="Processing"):
    post = {
        "content": value["origin_text"],
        "label": value["origin_label"],
        "words": process_text(value["origin_text"])
    }
    posts.append(post)

    article = {
        "text": value["origin_text"],
        # "title": value["origin_text"].split(',')[0],
        "title": value["origin_title"],
        "url": None
    }
    articles.append(article)

# 将数据分到训练集、验证集和测试集中
train_posts, temp_posts = train_test_split(posts, test_size=0.2, random_state=42)
val_posts, test_posts = train_test_split(temp_posts, test_size=0.5, random_state=42)

# 保存处理过的posts数据
print("Saving posts data...")
with open(os.path.join(output_dir, 'post', 'train.json'), 'w', encoding='utf-8') as f:
    json.dump(train_posts, f, ensure_ascii=False, indent=4)

with open(os.path.join(output_dir, 'post', 'val.json'), 'w', encoding='utf-8') as f:
    json.dump(val_posts, f, ensure_ascii=False, indent=4)

with open(os.path.join(output_dir, 'post', 'test.json'), 'w', encoding='utf-8') as f:
    json.dump(test_posts, f, ensure_ascii=False, indent=4)

# 保存处理过的articles数据
print("Saving articles data...")
with open(os.path.join(output_dir, 'articles', 'articles.json'), 'w', encoding='utf-8') as f:
    json.dump(articles, f, ensure_ascii=False, indent=4)

# 打印统计的数据长度
print(f"Total posts: {len(posts)}")
print(f"Training posts: {len(train_posts)}")
print(f"Validation posts: {len(val_posts)}")
print(f"Testing posts: {len(test_posts)}")

print("Processing complete!")
