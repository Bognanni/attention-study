from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='beauty',
    sequence_length=50,  # Beauty dataset is sparser, shorter sequences are better
    embedding_dim=128,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.5,
    negs_per_pos=256,
    gbce_t=0.0,
)
