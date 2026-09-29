from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='video_games',
    sequence_length=100,  # 100 is a good standard for video games
    embedding_dim=64,
    num_heads=1,
    max_batches_per_epoch=200,
    num_blocks=2,
    dropout_rate=0.5,
    negs_per_pos=256,
    gbce_t=0.0,  # 0.0 for standard SASRec
)
