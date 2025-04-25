# Databricks notebook source
# MAGIC %pip install deepspee
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../../setup/00_setup

# COMMAND ----------

# MAGIC %sh nvidia-smi

# COMMAND ----------

import mlflow
import os

username = spark.sql("SELECT current_user()").first()['current_user()']
username

experiment_path = f'/Users/{username}/deepspeed-distributor'

browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()

db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

os.environ['DATABRICKS_HOST'] = db_host
os.environ['DATABRICKS_TOKEN'] = db_token

# Manually create the experiment so that you know the ID and can send that to the worker nodes when you are ready to scale
experiment = mlflow.set_experiment(experiment_path)

num_nodes = 1 

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

import os

HF_DATASETS_CACHE = "/Volumes/will_smith/datasets/imagenet_1k"
os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

# COMMAND ----------

# MAGIC %sh 
# MAGIC
# MAGIC export TORCH_DISTRIBUTED_DEBUG="DETAIL"
# MAGIC export TORCH_CPP_LOG_LEVEL="INFO"
# MAGIC export TORCH_SHOW_CPP_STACKTRACES="1"
# MAGIC export NCCL_DEBUG="INFO"
# MAGIC export NCCL_DEBUG_SUBSYS="1"

# COMMAND ----------

from datasets import load_dataset
import datasets

# datasets.utils.logging.disable_progress_bar()
imagenet_1k = load_dataset('ILSVRC/imagenet-1k', cache_dir=HF_DATASETS_CACHE, trust_remote_code=True)

# COMMAND ----------

imagenet_1k

# COMMAND ----------

labels = set(imagenet_1k["train"]["label"])
num_class = len(labels)

# COMMAND ----------

import torch
from torch.utils.data import Dataset

class Imagenet1kDataset(Dataset):
    def __init__(self, data, transform=None):
        self.images = data['image']
        self.labels = data['label']
        self.transform = transform

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

transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),            # Random crop to 224x224
    transforms.RandomHorizontalFlip(),            # Random horizontal flip
    transforms.ToTensor(),                        # Convert PIL image to Tensor
    transforms.Normalize(                         # Normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],                # RGB mean
        std=[0.229, 0.224, 0.225]                  # RGB std
    ),
])

# COMMAND ----------

train_dataset = Imagenet1kDataset(imagenet_1k['train'], transform=transforms)
test_dataset = Imagenet1kDataset(imagenet_1k['test'], transform=transforms)

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

# if num_nodes > 1:
#         mlflow.set_experiment(experiment_path)
#         parent_run = mlflow.start_run(
#             run_name='deepspeed_distributor_w_config_low_level')
# else:
#     parent_run = None

# COMMAND ----------

# mlflow.end_run()

# COMMAND ----------

import torch

PYTORCH_DIR = '/dbfs/ml/pytorch'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
num_epochs = 5
momentum = 0.5
log_interval = 100
learning_rate = 1e-4

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):  # num_classes for imagenet 1k is 1000
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

  model = ResNet50().to(device)

  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
  
  for epoch in range(1, epochs + 1):
 
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
      print(f"Epoch [{epoch}/{epochs}], Loss: {running_loss:.4f}, Accuracy: {correct / total}")

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

# COMMAND ----------

dist.run(train_func, epochs=1, batch_size = 32, train_dataset = train_dataset, test_dataset = test_dataset)


# COMMAND ----------


