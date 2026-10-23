from ir_measures import R, nDCG

from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='amazon_gts',
    sequence_length=50,
    embedding_dim=128,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.3,
    negs_per_pos=128,
    gbce_t=0.5,  # 0.0 for standard SASRec
    reuse_item_embeddings=False,
    recommendation_limit=100,    # 50 for HitRatio@50 (equals to Recall@50)
    val_metric=R@50,
    metrics=[nDCG@100, nDCG@50, nDCG@20, nDCG@10, R@50, R@20, R@10, R@1]
)
