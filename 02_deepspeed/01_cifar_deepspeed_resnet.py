# Databricks notebook source
# MAGIC %pip install -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../setup/00_setup

# COMMAND ----------

import os

os.environ['HF_DATASETS_CACHE'] = cifar_cache

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
# MAGIC
# MAGIC The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Splits
# MAGIC
# MAGIC #### Total Rows: 60000
# MAGIC
# MAGIC
# MAGIC | Split       | # of examples |
# MAGIC |-------------|---------------|
# MAGIC | Train       | 50,000    |
# MAGIC | Validation  | 10,000       |

# COMMAND ----------

from utils import hf_dataset_utilities as hf_util

cifar_dataset = hf_util.hfds_download_volume(
  hf_cache = os.environ['HF_DATASETS_CACHE'],
  dataset_path= 'uoft-cs/cifar10',
  trust_remote_code = True, 
  disable_progress = False, 
)

# COMMAND ----------

CIFARDataset = hf_util.create_torch_image_dataset(
  image_key="img",
  label_key="label"
)

# COMMAND ----------

ds_transforms = hf_util.default_image_transforms(
  image_size = 32, 
  normalize_transform=True, 
  convert_rgb=True
)

# COMMAND ----------

train_dataset = CIFARDataset(cifar_dataset['train'], transform=ds_transforms)
test_dataset = CIFARDataset(cifar_dataset['test'], transform=ds_transforms)

# COMMAND ----------

num_classes = train_dataset.num_classes
num_nodes = 1

# COMMAND ----------

# MAGIC %md
# MAGIC # Training Func

# COMMAND ----------

import torch
 
NUM_WORKERS = int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers", "1"))
 
def get_gpus_per_worker(_):
  import torch
  return torch.cuda.device_count()
 
NUM_GPUS_PER_WORKER = sc.parallelize(range(4), 4).map(get_gpus_per_worker).collect()[0]

# COMMAND ----------

from pyspark.ml.deepspeed.deepspeed_distributor import DeepspeedTorchDistributor
 
dist = DeepspeedTorchDistributor(
  numGpus=NUM_GPUS_PER_WORKER,
  nnodes=1,
  localMode=True,  # Distribute training across workers.
  # deepspeedConfig=deepspeed_config
  )

# COMMAND ----------

# MAGIC %md
# MAGIC When operating in a standard notebook environment, the Python session is initiated with a login token for MLflow. When running DeepSpeed, however, individual GPUs will each have a separate Python process that does not inherit these credentials. To proceed, we can save these parameters to Python variables using dbutils, then assign them to environment variables within the function that DeepspeedTorchDistributor will distribute.

# COMMAND ----------

# if num_nodes > 1:
#         mlflow.set_experiment(experiment_path)
#         parent_run = mlflow.start_run(
#             run_name='deepspeed_distributor_w_config_low_level')
# else:
#     parent_run = None

# COMMAND ----------

import torch

PYTORCH_DIR = '/dbfs/ml/pytorch'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):  # num_classes for cifar is 10 
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
 
 
os.environ['HF_HOME'] = '/local_disk0/hf'
os.environ['TRANSFORMERS_CACHE'] = '/local_disk0/hf'
 
def train_func(
  *,
  train_dataset,
  test_dataset,
  batch_size: int = 256, 
  num_epochs: int = 5,
  mlflow_parent_run = None
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

  os.environ['MLFLOW_TRACKING_URI'] = 'databricks'
  os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_path
  os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'true'
  #os.environ['HF_MLFLOW_LOG_ARTIFACTS'] = 'True'
  
  os.environ['DATABRICKS_HOST'] = db_host
  os.environ['DATABRICKS_TOKEN'] = db_token

  device = torch.device('cuda')

  train_parameters = {'batch_size': batch_size, 'epochs': num_epochs}
  mlflow.log_params(train_parameters)

  model = ResNet18().to(device)

  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  momentum = 0.5
  log_interval = 100
  learning_rate = 1e-5
  betas = (0.9, 0.999)
  epsilon = 1e-08
  lr_scheduler_warmup_ratio = 0.1

  optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=epsilon, weight_decay=0)

  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  local_rank = int(os.environ["LOCAL_RANK"])
  # Use if distributed, using single node currently
  # global_rank = torch.distributed.get_rank()
  world_size = int(os.environ["WORLD_SIZE"])

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
  
  for epoch in range(1, num_epochs + 1):
 
    # Loss function and optimizer
    loss_func = nn.CrossEntropyLoss()

    for batch_idx, (inputs, labels) in enumerate(train_dataloader):

      if ((local_rank == 0) and ((batch_idx) % 10 == 0)):
        print(f"[TRAINING] [RANK {local_rank}] Running training on samples: {batch_idx - 1} of {len(train_dataloader)}")

      # Extract the input (image) and label tensors
      # inputs, labels = batch['image'].float().to(device), batch['label'].long().to(device)

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
      print(f"Epoch [{epoch}/{num_epochs}], Loss: {running_loss:.4f}, Accuracy: {correct / total}")

  if local_rank == 0:
    print(f"Finished training, logging model from RANK {local_rank}")
    mlflow.pytorch.log_model(model, "deepspeed_cifar_model")
    
    model.eval()
    loss_func = nn.CrossEntropyLoss()

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    for batch_idx, (inputs, labels) in enumerate(test_dataloader):

      if ((local_rank == 0) and ((batch_idx) % 10 == 0)):
        print(f"[VALIDATING] [RANK {local_rank}] Running validation on samples: {batch_idx - 1} of {len(test_dataloader)}")

      device = torch.device('cuda')
      inputs, labels = inputs.to(device), labels.to(device)
      output = model(inputs)
      loss = loss_func(output, labels)

      val_loss += loss.item()
      _, predicted = torch.max(output, 1)
      val_total += labels.size(0)
      val_correct += (predicted == labels).sum().item()

      val_loss /= len(test_dataloader)
      val_acc = val_correct / val_total

    val_loss /= len(test_dataloader.dataset)
    
    if local_rank == 0:

      print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
      print("Average test loss: {}".format(val_loss))

      mlflow.log_metric('val_loss', val_loss)
 
  print("Training finished.")
  return model 

# COMMAND ----------

# DBTITLE 1,Single epoch for testing
trained_model = dist.run(train_func, num_epochs=1, batch_size = 128, train_dataset = train_dataset, test_dataset = test_dataset, mlflow_parent_run=None)

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

# DBTITLE 1,Clear GPU Memory
# MAGIC %restart_python
