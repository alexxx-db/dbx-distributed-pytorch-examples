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
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


import mlflow
# from peft import get_peft_model

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

  print(f"Loaded dataset with path: {dataset_path}")

  if cache_dir:
    loaded_dataset = load_dataset(dataset_path, cache_dir=cache_dir, trust_remote_code=trust_remote)
  else: 
    loaded_dataset = load_dataset(dataset_path, trust_remote_code=trust_remote)

  # if dataset_path == 'zh-plus/tiny-imagenet':
  #   def transforms(examples):
  #     examples["image"] = [image.convert("RGB").resize((64,64)) for image in examples["image"]]
  #     return examples
    
  #   loaded_dataset = loaded_dataset.map(transforms, batched=True)

  return loaded_dataset 

def train_loop(dataset_path: str = 'zh-plus/tiny-imagenet', cache_dir: str = None, offload_device: str = "cpu", epochs: int = 5, learning_rate: float = 1e-4, batch_size: int = 64, trust_remote: bool = False, distributor:bool=True, mlflow_parent_run=None, deepspeed_config=None):

  import os
  import mlflow

  import torch
  from torch.utils.data import DataLoader
  
  import deepspeed
  from deepspeed import get_accelerator
  from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

  from deepspeed.utils import logger as ds_logger

  print(f"Starting training loop with arguments: batch_size={batch_size}, dataset_path={dataset_path}, cache_dir={cache_dir}, trust_remote={trust_remote}, distributor={distributor}, mlflow_parent_run={mlflow_parent_run}")

  local_rank = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(local_rank)
  deepspeed.init_distributed()
  device = torch.device(get_accelerator().device_name())
  global_rank = torch.distributed.get_rank()

  loaded_dataset = download_data(cache_dir='/Volumes/will_smith/datasets/imagenet_tiny', trust_remote=True)

  train_dataset = loaded_dataset.get('train', None)
  test_dataset = loaded_dataset.get('valid', None)

  if train_dataset == None or test_dataset == None: 
    raise Exception("Dataset not found. Please download the dataset and set the cache_dir to the path of the dataset.")

  # Apply transformations directly to the dataset
  train_dataset =  Dataset.from_dict({"image": train_dataset['image'], "label": train_dataset["label"]}).with_format("torch", device="cuda")
  test_dataset =  Dataset.from_dict({"image": test_dataset['image'], "label": test_dataset["label"]}).with_format("torch", device="cuda")

  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
  
  if distributor:
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['NCCL_P2P_DISABLE'] = '1'

  # We need different optimizer depending on whether it is using offload or not
  if offload_device == 'cpu':
      AdamOptimizer = DeepSpeedCPUAdam 
  else:
      AdamOptimizer = FusedAdam

  
  model = ResNet18().to(local_rank)

  optimizer = AdamOptimizer(model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95))
    
  # model, optimizer
  model_engine, optimizer, _, _  = deepspeed.initialize(
      model = model,
      optimizer = optimizer,
      dist_init_required=False,
    #  config = training_arguments.deepspeed
  )

  print("Created model engine and optimizer successfully!")

  # Now we can start the run loop
  for epoch in range(epochs):
    model.train()

    for step, batch in enumerate(train_dataloader):
        batch.to(device)
        outputs = model(**batch, use_cache=False)

        loss = outputs.loss

        model.backward(loss)
        model.step()

        run_dict = {
            'train_loss': loss,
            'step': step
          }

        # we need to make sure step is defined properly
        # We also only log loss on rank 0 node
        # if global_rank == 0:
        #   mlflow.log_metrics(metrics=run_dict, step=step) if global_rank == 0 else None

  return 'done'

# def main():
#   deepspeed_base = {
#     "train_batch_size": "auto",
#     "train_micro_batch_size_per_gpu": shared_parameters["per_device_batch_size"],
#     "gradient_accumulation_steps": shared_parameters['gradient_accumulation_steps'],
#     "gradient_clipping": shared_parameters['gradient_clipping'],
#     "bf16": {
#       "enabled": "true"
#     },
#     "optimizer": {
#           "type": "AdamW",
#           "params": {
#             "lr": shared_parameters["learning_rate"],
#             "betas": [
#               0.9,
#               0.999
#             ],
#             "eps": 1e-08
#           }
#         },
#     "scheduler": {
#       "type": "WarmupLR",
#       "params": {
#         "warmup_min_lr": 0,
#         "warmup_max_lr": shared_parameters["learning_rate"],
#         "warmup_num_steps": shared_parameters["warmup_steps"],
#         "warmup_type": "linear"
#       }
#     },
#     "tensorboard": {
#       "enabled": True,
#       "output_path": '/local_disk0/tensorboard',
#       "job_name": "finetune_llama_2_7b"
#     },
#     "steps_per_print": 10,
#     "wall_clock_breakdown": True,
#     "zero_optimization": {}
#   }

#   train_loop(cache_dir='/Volumes/will_smith/datasets/imagenet_tiny', trust_remote=True, deepspeed_config=deepspeed_base)

# if __name__ == "__main__":
#     main()









