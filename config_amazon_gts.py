from ir_measures import R, nDCG

from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='amazon_gts',
    sequence_length=50,
    embedding_dim=256,
    num_heads=2,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.2,
    negs_per_pos=128,
    gbce_t=0.75,  # 0.0 for standard SASRec
    reuse_item_embeddings=False,
    train_batch_size=512,
    recommendation_limit=100,    # 50 for HitRatio@50 (equals to Recall@50)
    val_metric=nDCG@10,
    metrics=[nDCG@100, nDCG@50, nDCG@20, nDCG@10, R@50, R@20, R@10, R@1]
)
