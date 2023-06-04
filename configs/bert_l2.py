from configs import bert as base_bert_config


def get_config():
    config = base_bert_config.get_config()
    config.n_blocks = 2

    return config
