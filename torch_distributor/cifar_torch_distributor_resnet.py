# Databricks notebook source
# MAGIC %run ../setup/00_setup

# COMMAND ----------

import mlflow

username = spark.sql("SELECT current_user()").first()['current_user()']
username

experiment_path = f'/Users/{username}/pytorch-distributor'

# Retrieve workspace URL and API token using dbutils notebook commands
db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Manually create the experiment so that you know the ID and can send that to the worker nodes when you are ready to scale
experiment = mlflow.set_experiment(experiment_path)

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

# MAGIC %md
# MAGIC # Resnet definition 

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocessing

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

# COMMAND ----------

import os

HF_DATASETS_CACHE = "/Volumes/will_smith/datasets/cifar"
os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE

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

# MAGIC %md
# MAGIC
# MAGIC ### Load dataset does not segment files to download based on split 
# MAGIC https://discuss.huggingface.co/t/how-can-i-download-a-specific-split-of-a-dataset/79027
# MAGIC

# COMMAND ----------

from datasets import load_dataset
import datasets

datasets.utils.logging.disable_progress_bar()
cifar_dataset = load_dataset('uoft-cs/cifar10')

# COMMAND ----------

cifar_dataset

# COMMAND ----------

labels = set(cifar_dataset["train"]["label"])
num_class = len(labels)

# COMMAND ----------

# MAGIC %md
# MAGIC ### TODO Use Pytorch transform instead of huggingface 

# COMMAND ----------

# MAGIC %md
# MAGIC # Training

# COMMAND ----------

import torch

PYTORCH_DIR = '/dbfs/ml/pytorch'

NUM_WORKERS = 2
NUM_GPUS_PER_NODE = torch.cuda.device_count()
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
      
def save_checkpoint(log_dir, model, optimizer, epoch):
  filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  state = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
  }
  torch.save(state, filepath)
  
def load_checkpoint(log_dir, epoch=num_epochs):
  filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  return torch.load(filepath)

def create_log_dir():
  log_dir = os.path.join(PYTORCH_DIR, str(time()))
  os.makedirs(log_dir)
  return log_dir

import torch.optim as optim
from torchvision import datasets, transforms
from time import time
import os

base_log_dir = create_log_dir()
print("Log directory:", base_log_dir)

def train_one_epoch(model, device, data_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0


    # Loss function and optimizer
    loss_func = nn.CrossEntropyLoss()

    for batch in data_loader:
      # Extract the input (image) and label tensors
      inputs, labels = batch['image'].float().to(device), batch['label'].long().to(device)
    
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

    epoch_loss = running_loss / len(data_loader)
    epoch_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

def train(log_dir):
  from datasets import Dataset 
  from torchvision import transforms
  from torch.utils.data import DataLoader
  from PIL import Image
  
  device = torch.device('cuda')

  train_parameters = {'batch_size': batch_size, 'epochs': num_epochs}
  mlflow.log_params(train_parameters)
  
  # Apply transformations directly to the dataset
  train_dataset =  Dataset.from_dict({"image": cifar_dataset['train']['img'], "label": cifar_dataset["train"]["label"]}).with_format("torch", device=device)

  # Create DataLoaders with batching and shuffling
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  model = ResNet18().to(device)

  # Hyperparameters
  learning_rate = 1e-5
  betas = (0.9, 0.999)
  epsilon = 1e-08
  lr_scheduler_warmup_ratio = 0.1

  optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=epsilon, weight_decay=0)

  for epoch in range(1, num_epochs + 1):
    train_one_epoch(model, device, train_dataloader, optimizer, epoch)
    save_checkpoint(log_dir, model, optimizer, epoch)

def test(log_dir):
  from datasets import Dataset 
  from torchvision import transforms
  from torch.utils.data import DataLoader
  from PIL import Image
  import torch.nn as nn

  loss_func = nn.CrossEntropyLoss()

  device = torch.device('cuda')
  loaded_model = ResNet18().to(device)  

  checkpoint = load_checkpoint(log_dir)
  loaded_model.load_state_dict(checkpoint['model'])
  loaded_model.eval()

  test_dataset =  Dataset.from_dict({"image": cifar_dataset['test']['img'], "label": cifar_dataset["test"]["label"]}).with_format("torch", device=device)

  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  val_loss = 0.0
  val_correct = 0
  val_total = 0
  for batch in test_dataloader:
    inputs, labels = batch['image'].float().to(device), batch['label'].long().to(device)
    output = loaded_model(inputs)
    loss = loss_func(output, labels)
    val_loss += loss.item()
    _, predicted = torch.max(output, 1)
    val_total += labels.size(0)
    val_correct += (predicted == labels).sum().item()

    val_loss /= len(test_dataloader)
    val_acc = val_correct / val_total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

  val_loss /= len(test_dataloader.dataset)
  print("Average test loss: {}".format(val_loss))
  
  mlflow.log_metric('val_loss', val_loss)
  
  mlflow.pytorch.log_model(loaded_model, "model")

# COMMAND ----------

# import torch.optim as optim
# import torch.nn as nn

# # Compute learning rate based on batch size
# lr = 5e-5
# # lr = base_lr * batch_size

# # Loss function and optimizer
# loss_func = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
# # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
# # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

# COMMAND ----------

with mlflow.start_run():
  mlflow.log_param('run_type', 'local')
  train(base_log_dir)
  test(base_log_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC # Torch Distributor training
# MAGIC

# COMMAND ----------

single_node_single_gpu_dir = create_log_dir()
print("Data is located at: ", single_node_single_gpu_dir)

def train_one_epoch(model, device, data_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0


    # Loss function and optimizer
    loss_func = nn.CrossEntropyLoss()

    for batch in data_loader:
      # Extract the input (image) and label tensors
      inputs, labels = batch['image'].float().to(device), batch['label'].long().to(device)
    
      # Zero the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      output = model(inputs)
      loss = loss_func(output, labels)

      # Backward pass and optimization
      loss.backward()
      optimizer.step()

      # Statistics
      running_loss += loss.item()
      _, predicted = torch.max(output, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(data_loader)
    epoch_acc = correct / total

    if int(os.environ["RANK"]) == 0:
        mlflow.log_metric('train_loss', loss.item())
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

def save_checkpoint(log_dir, model, optimizer, epoch):
  filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  state = {
    'model': model.module.state_dict(),
    'optimizer': optimizer.state_dict(),
  }
  torch.save(state, filepath)

# For distributed training we will merge the train and test steps into 1 main function
def main_fn(directory):
  
  #### Added imports here ####
  import mlflow
  import torch.distributed as dist
  from torch.nn.parallel import DistributedDataParallel as DDP
  from torch.utils.data.distributed import DistributedSampler
  from datasets import Dataset 
  from torchvision import transforms
  from torch.utils.data import DataLoader
  from PIL import Image
  import torch.nn as nn
  
  ############################

  ##### Setting up MLflow ####
  # We need to do this so that different processes that will be able to find mlflow
  os.environ['DATABRICKS_HOST'] = db_host
  os.environ['DATABRICKS_TOKEN'] = db_token

  # We set the experiment details here
  experiment = mlflow.set_experiment(experiment_path)
  ############################
  
  print("Running distributed training")
  if not dist.is_initialized():
      dist.init_process_group("nccl")
  
  local_rank = int(os.environ["LOCAL_RANK"])
  global_rank = int(os.environ["RANK"])
  
  if global_rank == 0:
    train_parameters = {'batch_size': batch_size, 'epochs': num_epochs, 'trainer': 'TorchDistributor'}
    mlflow.log_params(train_parameters)
  
  # Apply transformations directly to the dataset
  train_dataset =  Dataset.from_dict({"image": cifar_dataset['train']['img'], "label": cifar_dataset["train"]["label"]}).with_format("torch", device="cuda")

  # Create DataLoaders with batching and shuffling
  # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  
  #### Added Distributed Dataloader ####
  train_sampler = DistributedSampler(dataset=train_dataset)
  data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
  ######################################
  
  model = ResNet18().to(local_rank)
  #### Added Distributed Model ####
  ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
  #################################

  # Hyperparameters
  learning_rate = 1e-5
  betas = (0.9, 0.999)
  epsilon = 1e-08
  lr_scheduler_warmup_ratio = 0.1

  optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=epsilon, weight_decay=0)

  for epoch in range(1, num_epochs + 1):
    train_one_epoch(ddp_model, local_rank, data_loader, optimizer, epoch)
    
    if global_rank == 0: 
      print(f"Saving model after epoch: {epoch}")
      save_checkpoint(directory, ddp_model, optimizer, epoch)
  
  # save out the model for test
  if global_rank == 0:
    mlflow.pytorch.log_model(ddp_model, "model")
    
    ddp_model.eval()
    loss_func = nn.CrossEntropyLoss()

    test_dataset =  Dataset.from_dict({"image": cifar_dataset['test']['img'], "label": cifar_dataset["test"]["label"]}).with_format("torch", device="cuda")
    data_loader = torch.utils.data.DataLoader(test_dataset)    

    val_loss = 0.0
    val_correct = 0
    val_total = 0
    for batch in data_loader:
      device = torch.device('cuda')
      inputs, labels = batch['image'].float().to(device), batch['label'].long().to(device)
      output = ddp_model(inputs)
      loss = loss_func(output, labels)
      val_loss += loss.item()
      _, predicted = torch.max(output, 1)
      val_total += labels.size(0)
      val_correct += (predicted == labels).sum().item()

      val_loss /= len(data_loader)
      val_acc = val_correct / val_total

    val_loss /= len(data_loader.dataset)
    
    if int(os.environ["RANK"]) == 0:
      print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
      print("Average test loss: {}".format(val_loss))
      mlflow.log_metric('val_loss', val_loss)
    
    
  dist.destroy_process_group()
  
  return "finished" # can return any picklable object

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test without the distributor (single node single gpu)

# COMMAND ----------

# single node distributed run to quickly test that the whole process is working
with mlflow.start_run():
  mlflow.log_param('run_type', 'test_dist_code')
  main_fn(single_node_single_gpu_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Single node multi-GPU training distributor
# MAGIC - set num_processes to number of GPUs and local_mode = true

# COMMAND ----------

single_node_multi_gpu_dir = create_log_dir()
print("Data is located at: ", single_node_multi_gpu_dir)

# COMMAND ----------

# MAGIC %sh nvidia-smi

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

output = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True).run(main_fn, single_node_multi_gpu_dir)

# COMMAND ----------

test(single_node_multi_gpu_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multi-node multi-gpu 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ```
# MAGIC multi_node_dir = create_log_dir()
# MAGIC print("Data is located at: ", multi_node_dir)
# MAGIC
# MAGIC output_dist = TorchDistributor(num_processes=2, local_mode=False, use_gpu=True).run(main_fn, multi_node_dir)
# MAGIC test(multi_node_dir)
# MAGIC ```
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Inference
# MAGIC

# COMMAND ----------

import torch
import torchvision.transforms as transforms

def predict(input, model, device, transform=None):
    """
    Predict the class label for a single image from a dictionary.
    
    Args:
    - input (dict): A dictionary containing 'image' and 'label'.
    - model (torch.nn.Module): The trained ResNet model.
    - device (torch.device): The device (CPU or GPU) where the model is loaded.
    - transform (torchvision.transforms.Compose, optional): Optional transform to be applied to the image.
    
    Returns:
    - predicted_class (int): The predicted class label.
    """
    
    # Extract the image from the dictionary
    image = input['img']

    
    # Apply transformations (resize, crop, normalize)
    # if transform:
    #     image = image.convert('RGB')
    #     image = transform(image)
    # else:
    #     # Default transformation for ResNet (resize, crop, normalize)
    #     image = image.convert('RGB')
    #     default_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #     image = default_transform(image)

    # Add a batch dimension (as model expects a batch of images)
    # image = image.unsqueeze(0).to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    # Return the predicted class index
    return predicted.item()

# COMMAND ----------

import torchvision.transforms as transforms

test_index = 2


# Convert the PIL Image to a PyTorch tensor and move it to the device
transform = transforms.ToTensor()
image = transform(cifar_dataset['test']['img'][test_index]).float().to(device)

model.eval()

# Perform inference
with torch.no_grad():
    outputs = model(image.unsqueeze(0))  # Add batch dimension
    prediction = torch.max(outputs, 1)

# COMMAND ----------

cifar_dataset['test']['label'][test_index]

# COMMAND ----------

prediction

# COMMAND ----------


