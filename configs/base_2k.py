import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

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
    # config.emb_dim = 192
    # config.n_blocks = 6
    # config.n_heads = 6
    # config.block_size = 128

    # config.emb_dropout_prob = 0.1
    # config.attn_dropout_prob = 0.1
    # config.block_dropout_prob = 0.1

    # dataset
    config.data_file = '/home/akashc/splice/splice_2019/dataset_train_all.h5'

    # dataloader
    config.num_workers = 48

    # logging
    config.wandb = True
    config.logging_interval = 50
    config.eval_interval = 500
    config.ckpt_interval = 1000

    return config
