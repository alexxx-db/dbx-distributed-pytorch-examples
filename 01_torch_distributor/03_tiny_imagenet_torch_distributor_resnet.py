# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt  
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

# DBTITLE 1,Persist dataset to UC Volumes
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

TinyImagenetDataset = hf_util.create_torch_image_dataset(
  image_key="image",
  label_key="label"
)

# COMMAND ----------

ds_transforms = hf_util.default_image_transforms(
  image_size = 64, 
  normalize_transform=True, 
  convert_rgb=True
)

# COMMAND ----------

train_dataset = TinyImagenetDataset(tiny_imagenet['train'],
 transform=ds_transforms)
test_dataset = TinyImagenetDataset(tiny_imagenet['valid'], transform=ds_transforms)

# COMMAND ----------

num_classes = train_dataset.num_classes

# COMMAND ----------

# MAGIC %md
# MAGIC # Training Func

# COMMAND ----------

# MAGIC %md
# MAGIC When operating in a standard notebook environment, the Python session is initiated with a login token for MLflow. When running DeepSpeed, however, individual GPUs will each have a separate Python process that does not inherit these credentials. To proceed, we can save these parameters to Python variables using dbutils, then assign them to environment variables within the function that DeepspeedTorchDistributor will distribute.

# COMMAND ----------

import torch

PYTORCH_DIR = '/dbfs/ml/pytorch'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

momentum = 0.5
log_interval = 100
learning_rate = 1e-4

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNet50(nn.Module):
    def __init__(self, num_classes=num_classes):
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
  train_dataset,
  test_dataset,
  batch_size: int = 32, 
  epochs: int = 5,
  mlflow_run_id=None
):
  import torch 
  import torch.optim as optim
  from torchvision import datasets, transforms
  from time import time
  import os
  from datasets import Dataset 
  from torchvision import transforms
  from torch.utils.data import DataLoader
  from PIL import Image
  import mlflow

  os.environ['MLFLOW_TRACKING_URI'] = 'databricks'
  os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_path
  os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'true'
  
  os.environ['DATABRICKS_HOST'] = db_host
  os.environ['DATABRICKS_TOKEN'] = db_token

  device = torch.device('cuda')

  local_rank = int(os.environ["LOCAL_RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  # Use if distributed, using single node currently
  # global_rank = torch.distributed.get_rank()

  model = ResNet50(test_dataset.num_classes).to(device)

  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  learning_rate = 1e-3
  betas = (0.9, 0.999)
  epsilon = 1e-08
  lr_scheduler_warmup_ratio = 0.1

  train_parameters = {'batch_size': batch_size, 'epochs': epochs, 'learning_rate': learning_rate}

  # Only log from rank 0
  if local_rank == 0 and mlflow_run_id:
    mlflow.log_params(train_parameters)

  optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=epsilon, weight_decay=0)

  model.train()

  print(f"RANK: {local_rank}")
  
  for epoch in range(1, epochs + 1):
    running_loss = 0.0
    correct = 0
    total = 0
 
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

    epoch_loss = running_loss / len(train_dataloader)
    epoch_acc = correct / total
    
    if local_rank == 0:
      print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
      
      # Log metrics for each epoch from rank 0
      if mlflow_run_id:
        mlflow.log_metric('train_loss', epoch_loss, step=epoch)
        mlflow.log_metric('train_accuracy', epoch_acc, step=epoch)

  # Evaluation and model logging from rank 0 only
  if local_rank == 0:
    print(f"Finished training, logging model from RANK {local_rank}")
    
    if mlflow_run_id:
      mlflow.pytorch.log_model(model, "cifar_torch_distributor_resnet")
    
    model.eval()
    loss_func = nn.CrossEntropyLoss()

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    val_loss = 0.0
    val_correct = 0
    val_total = 0
 
    with torch.no_grad():
      for batch_idx, (inputs, labels) in enumerate(test_dataloader):
        if (batch_idx % 10 == 0):
          print(f"[VALIDATING] [RANK {local_rank}] Running validation on samples: {batch_idx} of {len(test_dataloader)}")

        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        loss = loss_func(output, labels)

        val_loss += loss.item()
        _, predicted = torch.max(output, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()

    val_loss /= len(test_dataloader)
    val_acc = val_correct / val_total

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    
    # Log validation metrics from rank 0
    if mlflow_run_id:
      mlflow.log_metric('val_loss', val_loss)
      mlflow.log_metric('val_accuracy', val_acc)
      mlflow.end_run()
 
  print("Training finished.")
  return model

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC TorchDistributor in PySpark handles data distribution for you, including managing the distributed sampling. So, in most cases, you do not need to manually create or use a DistributedSampler when using TorchDistributor. The TorchDistributor API abstracts away the need for manual handling of data partitioning.

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

import mlflow

# Create an MLflow run and get the run_id
with mlflow.start_run() as run:
    run_id = run.info.run_id

# COMMAND ----------

single_node_multi_gpu_dir = create_log_dir()
print("Data is located at: ", single_node_multi_gpu_dir)

from pyspark.ml.torch.distributor import TorchDistributor

timer = hf_util.Timer()

num_gpus = torch.cuda.device_count()
# TODO Update epochs as needed
num_epochs = 1

# TODO Update processes for number of GPUs
distributor = TorchDistributor(
    num_processes=4,
    local_mode=True,
    use_gpu=True
)

model = distributor.run(
    train_func,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    batch_size=256,
    epochs=num_epochs,
    mlflow_run_id=run_id
)

sn_mgpu_elapsed = timer.stop()
print(f"Elapsed time: {sn_mgpu_elapsed} seconds")
print(f"Elapsed time: {sn_mgpu_elapsed / 60} minutes")

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
logits = model(image_tensor.unsqueeze(0))

_, predicted = torch.max(logits, 1)

print(f"Predicted class: {predicted[0]}")
print(f"True class: {test_dataset.labels[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Additional Epochs

# COMMAND ----------

single_node_multi_gpu_longer_dir = create_log_dir()
print("Data is located at: ", single_node_multi_gpu_longer_dir)

timer = hf_util.Timer()

num_gpus = torch.cuda.device_count()
# TODO Update epochs as needed
num_epochs = 100

# TODO Update processes for number of GPUs
distributor = TorchDistributor(
    num_processes=4,
    local_mode=True,
    use_gpu=True
)

longer_model = distributor.run(
    train_func,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    batch_size=256,
    epochs=num_epochs,
    mlflow_run_id=run_id
)

sn_mgpu_longer_elapsed = timer.stop()
print(f"Elapsed time: {sn_mgpu_longer_elapsed} seconds")
print(f"Elapsed time: {sn_mgpu_longer_elapsed / 60} minutes")

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
logits = longer_model(image_tensor.unsqueeze(0))

_, predicted = torch.max(logits, 1)

print(f"Predicted class: {predicted[0]}")
print(f"True class: {test_dataset.labels[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Timing

# COMMAND ----------

print(f"Initial 1 Epoch Elapsed time: {sn_mgpu_elapsed:.2f} seconds")
print(f"Initial 100 Epoch Elapsed time: {sn_mgpu_longer_elapsed:.2f} seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC # Clear GPU memory

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------


