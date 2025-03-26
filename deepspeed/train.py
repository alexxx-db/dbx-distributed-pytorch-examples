import argparse
import os
import deepspeed
from deepspeed import get_accelerator
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.utils import logger as ds_logger

import time
import datasets
from datasets import Dataset, load_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models, ResNet18_Weights
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


import mlflow
from peft import get_peft_model

datasets.utils.logging.disable_progress_bar()

class ResNet18(nn.Module):
    def __init__(self, num_classes=200):  # num_classes for cifar is 10 
        super(ResNet18, self).__init__()
        
        # Load the pre-trained ResNet-18 model from torchvision
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)  # ResNet-18 architecture

        # Freeze the layers except the final fully connected layer
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Modify the final fully connected layer for the desired number of classes
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),  # Add dropout to reduce overfitting
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)
      
def download_data(dataset_path: str = 'zh-plus/tiny-imagenet', cache_dir: str = None, trust_remote: bool = False):
  if cache_dir:
    loaded_dataset = load_dataset(dataset_path, cache_dir=cache_dir, trust_remote_code=trust_remote)
  else: 
    loaded_dataset = load_dataset(dataset_path, trust_remote_code=trust_remote)

  if dataset_path == 'zh-plus/tiny-imagenet':
    def transforms(examples):
      examples["image"] = [image.convert("RGB").resize((64,64)) for image in examples["image"]]
      return examples
    
    loaded_dataset = loaded_dataset.map(transforms, batched=True)

  return loaded_dataset 

def train_loop():

   # We need different optimizer depending on whether it is using offload or not
    if offload_device == 'cpu':
       AdamOptimizer = DeepSpeedCPUAdam 
    else:
       AdamOptimizer = FusedAdam
    
    optimizer = AdamOptimizer(model.parameters(),
                              lr=training_arguments.learning_rate,
                              betas=(0.9, 0.95))
    
    # model, optimizer
    initialised_var  = deepspeed.initialize(
       model = model,
       optimizer = optimizer,
       dist_init_required=False,
       config = training_arguments.deepspeed
    )
    













