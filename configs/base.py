import ml_collections

from constants import TRAIN_DATA_PATH, ModelType


def get_config():
    config = ml_collections.ConfigDict()
    config.n_classes = 3

    # random seeds
    config.seed = 42

    # optimizer
    config.learning_rate = 0.001
    config.lr_warmup_steps = 3_000
    config.lr_cosine_decay = False
    config.lr_exp_decay = True
    config.lr_decay_begin_epochs = 6
    config.lr_decay_rate = 0.5

    config.beta1 = 0.9
    config.beta2 = 0.95
    config.weight_decay = 0.1
    config.grad_norm_clip = 1.0
    config.batch_size = 12 * 8
    config.epochs = 10
    config.train_steps = 250_000

    # model
    config.context_length = 2000
    config.model_type = ModelType.DILATED_CONV
    config.deterministic = True

    # transformer size
    config.emb_dim = 64
    config.n_blocks = 1
    config.n_heads = 2

    config.emb_dropout_prob = 0.1
    config.attn_dropout_prob = 0.1
    config.block_dropout_prob = 0.1

    # dataset
    config.data_file = TRAIN_DATA_PATH
    config.shuffle = True

    # dataloader
    config.num_workers = 48

    # logging
    config.wandb = True
    config.logging_interval = 50
    config.eval_interval = -1  # no sampled eval during training (still does full val each epoch)
    config.ckpt_interval_epochs = 2

    return config
