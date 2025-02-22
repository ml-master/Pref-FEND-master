# ================== Weibo ==================
# Post
python get_post_tokens.py --dataset 'Weibo' --pretrained_model 'bert-base-chinese'

# Articles
python get_articles_tokens.py --dataset 'Weibo' --pretrained_model 'bert-base-chinese'

# ================== Twitter ==================
# Post
python get_post_tokens.py --dataset 'Twitter' --pretrained_model 'bert-base-uncased'

# Articles
python get_articles_tokens.py --dataset 'Twitter' --pretrained_model 'bert-base-uncased'

# ================== GossipCop ==================
# Post
python get_post_tokens.py --dataset 'gossip' --pretrained_model 'bert-base-cased'

# Articles
python get_articles_tokens.py --dataset 'gossip' --pretrained_model 'bert-base-cased'
