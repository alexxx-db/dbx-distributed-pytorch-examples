# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt  

# COMMAND ----------

# MAGIC %pip install mosaicml-streaming
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../setup/00_setup

# COMMAND ----------

# MAGIC %sh nvidia-smi

# COMMAND ----------

from time import time

def create_log_dir():
  log_dir = os.path.join(PYTORCH_DIR, str(time()))
  os.makedirs(log_dir)
  return log_dir

# COMMAND ----------

# MAGIC %md
# MAGIC ## TorchDistributor 
# MAGIC
# MAGIC - Prepare single node code: Prepare and test the single node code with PyTorch, PyTorch Lightning, or other frameworks that are based on PyTorch/PyTorch Lightning like, the HuggingFace Trainer API.
# MAGIC
# MAGIC - Prepare code for standard distributed training: You need to convert your single process training to distributed training. Have this distributed code all encompassed within one training function that you can use with the TorchDistributor.
# MAGIC
# MAGIC - Move imports within training function: Add the necessary imports, such as import torch, within the training function. Doing so allows you to avoid common pickling errors. Furthermore, the device_id that models and data are be tied to is determined by:
# MAGIC
# MAGIC - Launch distributed training: Instantiate the TorchDistributor with the desired parameters and call .run(*args) to launch training.
# MAGIC
# MAGIC

# COMMAND ----------

import os

os.environ['HF_DATASETS_CACHE'] = tiny_imagenet_cache

# COMMAND ----------

from utils import hf_dataset_utilities as hf_util

tiny_imagenet = hf_util.hfds_download_volume(
  hf_cache = os.environ['HF_DATASETS_CACHE'] ,
  dataset_path= 'zh-plus/tiny-imagenet',
  trust_remote_code = True, 
  disable_progress = False, 
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Splits
# MAGIC
# MAGIC #### Total Rows: 110,000
# MAGIC
# MAGIC
# MAGIC | Split       | # of examples |
# MAGIC |-------------|---------------|
# MAGIC | Train       | 100,000    |
# MAGIC | Validation  | 10,000       |

# COMMAND ----------

tiny_imagenet

# COMMAND ----------

# MAGIC %md
# MAGIC Cannot pickle custom datasets returned from a function so defining it here

# COMMAND ----------

from torch.utils.data import Dataset

class TinyImagenetDataset(Dataset):
  def __init__(self, data, transform=None):
      self.images = data["image"]
      self.labels = data["label"]
      self.transform = transform
      self.num_classes = len(set(data["label"]))

  def __len__(self):
      return len(self.images)

  def __getitem__(self, idx):
      image = self.images[idx]
      label = self.labels[idx]
      
      if self.transform:
          image = self.transform(image)
      
      return image, label

# COMMAND ----------

from torchvision import transforms

def default_image_transforms(
  image_size: int,
  normalize_transform: bool = True,
  convert_rgb: bool = True, 
):
  
  transform_list = [
    transforms.Resize((image_size, image_size)),  # Resize to specified size
    transforms.RandomHorizontalFlip(),            # Random horizontal flip
    transforms.ToTensor()                         # Convert PIL image to Tensor
  ]
  
  if convert_rgb:
    transform_list.insert(0, transforms.Grayscale(num_output_channels=3))  # Convert grayscale to RGB
  
  if normalize_transform:
    transform_list.append(transforms.Normalize(  # Normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],               # RGB mean
        std=[0.229, 0.224, 0.225]                 # RGB std
    ))
  
  default_transforms = transforms.Compose(transform_list)

  return default_transforms

ds_transforms = default_image_transforms(
  image_size = 64, 
  normalize_transform=True, 
  convert_rgb=True
)

# COMMAND ----------

train_dataset = TinyImagenetDataset(tiny_imagenet['train'],transform=ds_transforms)
test_dataset = TinyImagenetDataset(tiny_imagenet['valid'], transform=ds_transforms)

# COMMAND ----------

# MAGIC %md
# MAGIC Save out the transformed DS so that we can use it with MDS 

# COMMAND ----------

import torch

volume_name = "imagenet1k_mds"

mds_volume_path = f'/Volumes/{catalog}/{schema}/{volume_name}'

try:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.imagenet1k_mds")
except Exception as e:
    print(f"Error: Could not create catalog due to {e}")

torch.save(train_dataset, f'{mds_volume_path}/pytorch_datasets/train.pt')
torch.save(test_dataset, f'{mds_volume_path}/pytorch_datasets/test.pt')

# COMMAND ----------

num_classes = train_dataset.num_classes

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create MDS files

# COMMAND ----------

print(type(tiny_imagenet['train'][0]['image']))
print(type(tiny_imagenet['train'][0]['label']))
print("------------------------------------")
print(type(tiny_imagenet['valid'][0]['image']))
print(type(tiny_imagenet['valid'][0]['label']))

# COMMAND ----------

# DBTITLE 1,Write training dataset
import numpy as np
from PIL import Image
from uuid import uuid4
from streaming import MDSWriter

# Local or remote directory path to store the output compressed files.
out_root = f'/Volumes/{catalog}/{schema}/{volume_name}/mds_datasets/train/'

# A dictionary of input fields to an Encoder/Decoder type
columns = {
    'image': 'pil',
    'label': 'int',
}

# Compression algorithm name
compression = 'zstd'

# Use `MDSWriter` to iterate through the input data and write to a collection of `.mds` files.
try:
    with MDSWriter(out=out_root, columns=columns, compression=compression) as out:
        for record in tiny_imagenet['train']:
            sample = {
                "image": record['image'],
                "label": record['label']
            }
            out.write(sample)
except Exception as e:
    print(f"An error occurred: {e}")

# COMMAND ----------

out_root = f'{mds_volume_path}/mds_datasets/test/'

try:
    with MDSWriter(out=out_root, columns=columns, compression=compression) as out:
        for record in tiny_imagenet['valid']:
            sample = {
                "image": record['image'],
                "label": record['label']
            }
            out.write(sample)
            
except Exception as e:
    print(f"An error occurred: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Training Func

# COMMAND ----------

import mlflow

mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC When operating in a standard notebook environment, the Python session is initiated with a login token for MLflow. When running DeepSpeed, however, individual GPUs will each have a separate Python process that does not inherit these credentials. To proceed, we can save these parameters to Python variables using dbutils, then assign them to environment variables within the function that DeepspeedTorchDistributor will distribute.

# COMMAND ----------

from streaming import StreamingDataset
from typing import Callable, Any

class TinyImageNetMDS(StreamingDataset):
    def __init__(self,
                 remote: str,
                 local: str,
                 shuffle: bool,
                 batch_size: int,
                 transforms: Callable
                ) -> None:
        super().__init__(local=local, remote=remote, shuffle=shuffle, batch_size=batch_size)
        self.transforms = transforms

    def __getitem__(self, idx:int) -> Any:
        obj = super().__getitem__(idx)
        image = obj['image']
        label = obj['label']
        return self.transforms(image), label

# COMMAND ----------

from torch.utils.data import DataLoader
from streaming import StreamingDataset

batch_size = 128

# Remote directory where dataset is stored, from above
remote_dir = f'{mds_volume_path}/mds_datasets'
remote_train = remote_dir + "/train/"
remote_test = remote_dir + "/test/"

# Local directory where dataset is cached during training
local_dir = '/local_disk0/mds'

local_train = local_dir + "/train/"
local_test = local_dir + "/test/"

train_dataset = TinyImageNetMDS(remote_train, local_train, True, batch_size=batch_size, transforms=ds_transforms)
test_dataset  = TinyImageNetMDS(remote_test, local_test, False, batch_size=batch_size, transforms=ds_transforms)

# COMMAND ----------

import mlflow

mlflow.autolog(disable=True)

# COMMAND ----------

import streaming

streaming.base.util.clean_stale_shared_memory()

# COMMAND ----------

import torch

PYTORCH_DIR = '/dbfs/ml/pytorch'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 512
num_epochs = 5
momentum = 0.5
log_interval = 100
learning_rate = 1e-3

from streaming import StreamingDataset
from typing import Callable, Any

class TinyImageNetMDS(StreamingDataset):
    def __init__(self,
                 remote: str,
                 local: str,
                 shuffle: bool,
                 batch_size: int,
                 transforms: Callable
                ) -> None:
        super().__init__(local=local, remote=remote, shuffle=shuffle, batch_size=batch_size)
        self.transforms = transforms

    def __getitem__(self, idx:int) -> Any:
        obj = super().__getitem__(idx)
        image = obj['image']
        label = obj['label']
        return self.transforms(image), label

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNet50(nn.Module):
    def __init__(self, num_classes=200):  # num_classes for imagenet 1k is 1000
        super(ResNet50, self).__init__()
        
        # Load the pre-trained ResNet-50 model from torchvision
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)  # ResNet-50 architecture

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
 
 
os.environ['HF_HOME'] = '/local_disk0/hf'
os.environ['TRANSFORMERS_CACHE'] = '/local_disk0/hf'
 
def train_func(
  *,
  # train_dataset,
  # test_dataset,
  batch_size: int = 128, 
  epochs: int = 5,
  mlflow_parent_run = None,
  patience: int = 5
):

  import torch 
  import torch.optim as optim
  from torchvision import datasets, transforms
  from time import time
  import os
  from datasets import Dataset 
  from torch.utils.data import DataLoader
  from PIL import Image
  from streaming import StreamingDataset

  os.environ['MLFLOW_TRACKING_URI'] = 'databricks'
  os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_path
  os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'true'
  
  os.environ['DATABRICKS_HOST'] = db_host
  os.environ['DATABRICKS_TOKEN'] = db_token

  # Remote directory where dataset is stored, from above
  remote_dir = f'{mds_volume_path}/mds_datasets'
  remote_train = remote_dir + "/train/"
  remote_test = remote_dir + "/test/"

  # Local directory where dataset is cached during training
  local_dir = '/local_disk0/mds'
  local_train = local_dir + "/train/"
  local_test = local_dir + "/test/"

  train_dataset = TinyImageNetMDS(remote_train, local_train, True, batch_size=batch_size, transforms=ds_transforms)
  test_dataset  = TinyImageNetMDS(remote_test, local_test, False, batch_size=batch_size, transforms=ds_transforms)

  device = torch.device('cuda')

  train_parameters = {'batch_size': batch_size, 'epochs': num_epochs}
  mlflow.log_params(train_parameters)

  model = ResNet50(num_classes).to(device)

  train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

  learning_rate = 1e-3
  betas = (0.9, 0.999)
  epsilon = 1e-08
  lr_scheduler_warmup_ratio = 0.1

  optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=epsilon, weight_decay=0)

  model.train()
  
  running_loss = 0.0
  correct = 0
  total = 0

  local_rank = int(os.environ["LOCAL_RANK"])
  world_size = int(os.environ["WORLD_SIZE"])


  # Use if distributed, using single node currently
  # global_rank = torch.distributed.get_rank()

  print(f"RANK: {local_rank}")

  # if mlflow_parent_run:
  #       from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID  
  #       run_tags = {MLFLOW_PARENT_RUN_ID: mlflow_parent_run.info.run_id}
  # else:
  #     run_tags = {}

  # We want to log all configs and track loss on our primary node
  # But not on the other mlflow runs that exist just to log system stats
  # if local_rank == 0:
  #     active_run = mlflow.start_run(run_name=f"deepspeed_cifar_{local_rank}",
  #                                     tags={})

  best_val_loss = float('inf')
  patience_counter = 0

  print(f"RANK: {local_rank}")

  for epoch in range(1, epochs + 1):
 
    # Loss function and optimizer
    loss_func = nn.CrossEntropyLoss()

    for batch_idx, (inputs, labels) in enumerate(train_dataloader):

      if ((local_rank == 0) and ((batch_idx) % 10 == 0)):
        print(f"[TRAINING] [RANK {local_rank}] Running training on samples: {batch_idx} of {len(train_dataloader)}")

      inputs, labels = inputs.to(device), labels.to(device)

      # Zero the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(inputs)
      loss = loss_func(outputs, labels)

      # Backward pass and optimization
      loss.backward()
      optimizer.step()

      # Statistics
      running_loss += loss.item()
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    if local_rank == 0:
      print(f"Epoch [{epoch}/{epochs}], Loss: {running_loss:.4f}, Accuracy: {correct / total}")

    # if local_rank == 0:
    #   model.eval()
    #   loss_func = nn.CrossEntropyLoss()

    #   test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #   val_loss = 0.0
    #   val_correct = 0
    #   val_total = 0

    #   for batch_idx, (inputs, labels) in enumerate(test_dataloader):

    #     if ((local_rank == 0) and ((batch_idx) % 10 == 0)):
    #       print(f"[VALIDATING] [RANK {local_rank}] Running validation on samples: {batch_idx} of {len(test_dataloader)}")

    #     device = torch.device('cuda')
    #     inputs, labels = inputs.to(device), labels.to(device)
    #     output = model(inputs)
    #     loss = loss_func(output, labels)

    #     val_loss += loss.item()
    #     _, predicted = torch.max(output, 1)
    #     val_total += labels.size(0)
    #     val_correct += (predicted == labels).sum().item()

    #   val_loss /= len(test_dataloader)
    #   val_acc = val_correct / val_total

    #   print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    #   print("Average test loss: {}".format(val_loss))

    #   mlflow.log_metric('val_loss', val_loss)

    #   if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     patience_counter = 0
    #   else:
    #     patience_counter += 1

    #   if patience_counter >= patience:
    #     print("Early stopping triggered")
    #     break

  if local_rank == 0:
    print(f"Finished training, logging model from RANK {local_rank}")
    mlflow.pytorch.log_model(model, "tiny_imagenet_torch_distributor_resnet_mds")
 
  print("Training finished.")
  return model

# COMMAND ----------

# MAGIC %md ## Distributed Setup
# MAGIC
# MAGIC When you wrap the single-node code in the `train()` function, Databricks recommends you include all the import statements inside the `train()` function to avoid library pickling issues.
# MAGIC
# MAGIC Everything else is what is normally required for getting distributed training to work within PyTorch.
# MAGIC - Calling `dist.init_process_group("nccl")` at the beginning of `train()`
# MAGIC - Calling `dist.destroy_process_group()` at the end of `train()`
# MAGIC - Setting `local_rank = int(os.environ["LOCAL_RANK"])`
# MAGIC - Adding a `DistributedSampler` to the `DataLoader`
# MAGIC - Wrapping the model with a `DDP(model)`
# MAGIC - For more information, view https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html

# COMMAND ----------

single_node_multi_gpu_dir = create_log_dir()
print("Data is located at: ", single_node_multi_gpu_dir)

from pyspark.ml.torch.distributor import TorchDistributor

timer = hf_util.Timer()

num_gpus = torch.cuda.device_count()
num_epochs = 1 

model = TorchDistributor(num_processes=num_gpus, local_mode=True, use_gpu=True).run(train_func, epochs=num_epochs, batch_size = 512)

sn_mgpu_elapsed = timer.stop()
print(f"Elapsed time: {sn_mgpu_elapsed} seconds")
print(f"Elapsed time: {sn_mgpu_elapsed / 60} minutes")

# COMMAND ----------

tiny_imagenet['train'][0]['image']

# COMMAND ----------

import torch
from torchvision import transforms

# Convert the image to a tensor
image_tensor = transforms.ToTensor()(tiny_imagenet['train'][0]['image'])

# Move the tensor to the GPU
image_tensor = image_tensor.to(device)

# COMMAND ----------

# Apply unsqueeze and pass to the model
logits = model(image_tensor.unsqueeze(0))

_, predicted = torch.max(logits, 1)

print(f"Predicted class: {predicted[0]}")
print(f"True class: {tiny_imagenet['train'][0]['label']}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Additional Epochs

# COMMAND ----------

single_node_multi_gpu__extra_epochs_dir = create_log_dir()
print("Data is located at: ", single_node_multi_gpu__extra_epochs_dir)

from pyspark.ml.torch.distributor import TorchDistributor

timer = hf_util.Timer()

num_gpus = torch.cuda.device_count()
num_epochs = 150

trained_model = TorchDistributor(num_processes=num_gpus, local_mode=True, use_gpu=True).run(train_func, epochs=num_epochs, batch_size = 512)

longer_train_elapsed = timer.stop()
print(f"Elapsed time: {longer_train_elapsed:.2f} seconds")
print(f"Elapsed time: {longer_train_elapsed / 60:.2f} minutes")

# COMMAND ----------

print(f"Initial 1 Epoch Elapsed time: {sn_mgpu_elapsed:.2f} seconds")
print(f"Initial 150 Epoch Elapsed time: {longer_train_elapsed:.2f} seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC # Inference

# COMMAND ----------

test_dataset.images[0]

# COMMAND ----------

import torch
from torchvision import transforms

# Convert the image to a tensor
image_tensor = transforms.ToTensor()(test_dataset.images[0])

# Move the tensor to the GPU
image_tensor = image_tensor.to(device)

# COMMAND ----------

# Apply unsqueeze and pass to the model
logits = trained_model(image_tensor.unsqueeze(0))

_, predicted = torch.max(logits, 1)

print(f"Predicted class: {predicted[0]}")
print(f"True class: {test_dataset.labels[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Clear GPU memory

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------


