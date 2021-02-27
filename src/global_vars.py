import torch


SEED = 11
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 2
LR = 1e-3
BATCH_SIZE = 128
THRESHOLD = 0.65
PARAMS_DIR = '../params'
