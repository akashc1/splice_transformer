from enum import Enum, auto

# Length of sequence / context
CONTEXT_LENGTHS = {80, 400, 2000, 10000}
MAX_CONTEXT_LENGTH = 10_000
SEQUENCE_LENGTH = 5000

# Paths to training/test datasets
TRAIN_DATA_PATH = '/home/akashc/splice/splice_2019/dataset_train_all.h5'
TEST_DATA_PATH = '/home/akashc/splice/splice_2019/dataset_test_0.h5'


class ModelType(Enum):
    DILATED_CONV = auto()
    BERT = auto()
    ENFORMER = auto()
