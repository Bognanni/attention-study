from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='video_games',
    sequence_length=200,
    embedding_dim=128,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.5,
    negs_per_pos=1,
    gbce_t=0.0,  # 0.0 for standard SASRec
    reuse_item_embeddings=False,
)
