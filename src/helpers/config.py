DATA_PATH = "/kaggle/input/predicting-stock-trends-rise-or-fall/train.csv"
# DATA_PATH = "data/raw/train.csv"

MODEL_PATH = "bst_model.pt"

WINDOW_SIZE = 30
HORIZON = 30

BATCH_SIZE = 128
LEARNING_RATE = 3e-4
EPOCHS = 20

USE_CUDA = True