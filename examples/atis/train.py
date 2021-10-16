import random
import numpy as np
import torch

from ludwig.api import LudwigModel
from ludwig.backend import LOCAL

# Set random seeds for reproducibility.
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

dataset = "/home/tanl/data/atis_intents_train.csv"

config = {
    'input_features': [{'name': 'message', 'type': 'text', 'encoder': 'bert'}],
    'output_features': [{'name': 'intent', 'type': 'category'}],
    'training': {'batch_size': 4}
}

model = LudwigModel(config=config, backend=LOCAL)

train_stats, _, _ = model.train(dataset)