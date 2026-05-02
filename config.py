import torch

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_DIR = 'data'
SUPERVISED_MODEL_WEIGHTS = 'supervised_model_weights.pth'
