from configs import bert as base_config


def get_config():
    config = base_config.get_config()
    config.perceiver = True

    return config
