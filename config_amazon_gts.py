from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='amazon_gts',
    sequence_length=50,
    embedding_dim=128,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.5,
    negs_per_pos=256,
    gbce_t=0.0,  # 0.0 for standard SASRec
    reuse_item_embeddings=False,
)
