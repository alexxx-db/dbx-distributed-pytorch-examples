# Databricks notebook source
# MAGIC %md
# MAGIC # End to end distributed training on a Databricks Notebook
# MAGIC
# MAGIC Distributed training on PyTorch is often done by creating a file (`train.py`) and using the `torchrun` CLI to run distributed training using that file. Databricks offers a method of doing distributed training directly on a Databricks notebook. You can define the `train()` function within a notebook and use the `TorchDistributor` API to train the model across the workers.
# MAGIC
# MAGIC This notebook illustrates how to develop interactively within a notebook. Particularly with larger deep learning projects, Databricks recommends leveraging the `%run` command in order to split up your code into manageable chunks.
# MAGIC
# MAGIC In this notebook, you: 
# MAGIC - Train a simple single GPU model on the classic MNIST dataset 
# MAGIC - Adapt that code for distributed training 
# MAGIC - Learn how the TorchDistributor can be leveraged to help you scale up the model training across multiple GPUs or multiple nodes. 
# MAGIC
# MAGIC ## Requirements
# MAGIC - Databricks Runtime ML 13.0 and above
# MAGIC - This notebook should be run on a cluster with Single User access mode. If the cluster should be shared with other team members, contact your Databricks account team for solutions.
# MAGIC - (Recommended) GPU instances [AWS](https://docs.databricks.com/clusters/gpu.html) | [Azure](https://learn.microsoft.com/en-gb/azure/databricks/clusters/gpu) | [GCP](https://docs.gcp.databricks.com/clusters/gpu.html)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### MLflow setup
# MAGIC
# MAGIC MLflow is a tool to support the tracking of machine learning experiments and logging of models. The `db_host` variable controls the MLflow tracking server and needs to be set to the URL of the workspace.
# MAGIC
# MAGIC ***NOTE*** The MLflow PyTorch Autologging APIs are designed for PyTorch Lightning and won't work with Native PyTorch

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

# MAGIC %md ## Define train and test functions
# MAGIC
# MAGIC The following cell contains code that describes the model, the train function, and the testing function; all of which are designed to run locally. Next, the code introduces the changes needed to move training from the local setting to a distributed setting.
# MAGIC
# MAGIC All the torch code leverages standard PyTorch APIs, there are no custom libraries required or alterations in the way the code is written. This notebook focuses on how to scale your training with `TorchDistributor` and does not go through the model code. 

# COMMAND ----------

import torch
NUM_WORKERS = 2
NUM_GPUS_PER_NODE = torch.cuda.device_count()

# COMMAND ----------

NUM_GPUS_PER_NODE

# COMMAND ----------

PYTORCH_DIR = '/dbfs/ml/pytorch'

batch_size = 100
num_epochs = 50
momentum = 0.5
log_interval = 100
learning_rate = 0.001

import torch
import torch.nn as nn
import torch.nn.functional as F

# Our Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def train_one_epoch(model, device, data_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader) * len(data),
                100. * batch_idx / len(data_loader), loss.item()))
            
            mlflow.log_metric('train_loss', loss.item())

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

def train(log_dir):
  device = torch.device('cuda')

  train_parameters = {'batch_size': batch_size, 'epochs': num_epochs}
  mlflow.log_params(train_parameters)
  
  train_dataset = datasets.MNIST(
    'data', 
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
  data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  model = Net().to(device)

  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  for epoch in range(1, num_epochs + 1):
    train_one_epoch(model, device, data_loader, optimizer, epoch)
    save_checkpoint(log_dir, model, optimizer, epoch)
    
def test(log_dir):
  device = torch.device('cuda')
  loaded_model = Net().to(device)  

  checkpoint = load_checkpoint(log_dir)
  loaded_model.load_state_dict(checkpoint['model'])
  loaded_model.eval()

  test_dataset = datasets.MNIST(
    'data', 
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
  data_loader = torch.utils.data.DataLoader(test_dataset)

  test_loss = 0
  for data, target in data_loader:
      data, target = data.to(device), target.to(device)
      output = loaded_model(data)
      test_loss += F.nll_loss(output, target)
        
  test_loss /= len(data_loader.dataset)
  print("Average test loss: {}".format(test_loss.item()))
  
  mlflow.log_metric('test_loss', test_loss.item())
  
  mlflow.pytorch.log_model(loaded_model, "model")

# COMMAND ----------

# MAGIC %md ### Train the model locally
# MAGIC
# MAGIC To test that this runs correctly, you can trigger a train and test iteration using the functions defined above.

# COMMAND ----------

import timeit

start_time = timeit.default_timer()

with mlflow.start_run():
  mlflow.log_param('run_type', 'local')
  train(base_log_dir)
  test(base_log_dir)
  
elapsed = timeit.default_timer() - start_time
print(f"Elapsed time: {elapsed} seconds")

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

single_node_single_gpu_dir = create_log_dir()
print("Data is located at: ", single_node_single_gpu_dir)

def train_one_epoch(model, device, data_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(data_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(data_loader) * len(data),
          100. * batch_idx / len(data_loader), loss.item()))
      
      if int(os.environ["RANK"]) == 0:
        mlflow.log_metric('train_loss', loss.item())

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
  
  train_dataset = datasets.MNIST(
    'data',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
  
  #### Added Distributed Dataloader ####
  train_sampler = DistributedSampler(dataset=train_dataset)
  data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
  ######################################
  
  model = Net().to(local_rank)
  #### Added Distributed Model ####
  ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
  #################################

  optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=momentum)
  for epoch in range(1, num_epochs + 1):
    train_one_epoch(ddp_model, local_rank, data_loader, optimizer, epoch)
    
    if global_rank == 0: 
      save_checkpoint(directory, ddp_model, optimizer, epoch)
  
  # save out the model for test
  if global_rank == 0:
    mlflow.pytorch.log_model(ddp_model, "model")
    
    ddp_model.eval()
    test_dataset = datasets.MNIST(
      'data', 
      train=False,
      download=True,
      transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    data_loader = torch.utils.data.DataLoader(test_dataset)    

    test_loss = 0
    for data, target in data_loader:
      device = torch.device('cuda')
      data, target = data.to(device), target.to(device)
      output = ddp_model(data)
      test_loss += F.nll_loss(output, target)
          
    test_loss /= len(data_loader.dataset)
    print("Average test loss: {}".format(test_loss.item()))
    
    mlflow.log_metric('test_loss', test_loss.item())

    
  dist.destroy_process_group()
  
  return "finished" # can return any picklable object

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Test without TorchDistributor
# MAGIC
# MAGIC The following validates our training loop by running training on a single GPU.

# COMMAND ----------

import timeit

start_time = timeit.default_timer()

main_fn(single_node_single_gpu_dir)

elapsed = timeit.default_timer() - start_time
print(f"Elapsed time: {elapsed} seconds")

# COMMAND ----------

# # single node distributed run to quickly test that the whole process is working
# with mlflow.start_run():
#   mlflow.log_param('run_type', 'test_dist_code')
#   main_fn(single_node_single_gpu_dir)

# COMMAND ----------

# MAGIC %md ### Single node multi-GPU training
# MAGIC
# MAGIC PyTorch provides a [roundabout way](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html) for doing single node multi-GPU training. Databricks provides a more streamlined solution that allows you to move from single node multi-GPU to multi node training seamlessly. To do single node multi-GPU training on Databricks, you need to invoke the `TorchDistributor` API and set `num_processes` equal to the number of available GPUs on the driver node that you want to use and set `local_mode=True`.

# COMMAND ----------

single_node_multi_gpu_dir = create_log_dir()
print("Data is located at: ", single_node_multi_gpu_dir)

from pyspark.ml.torch.distributor import TorchDistributor
import timeit

start_time = timeit.default_timer()

output = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True).run(main_fn, single_node_multi_gpu_dir)

elapsed = timeit.default_timer() - start_time
print(f"Elapsed time: {elapsed} seconds")

test(single_node_multi_gpu_dir)

# COMMAND ----------

# MAGIC %md ### Multi-node training
# MAGIC
# MAGIC To move from single node multi-GPU training to multi-node training, you just change `num_processes` to the number of GPUs that you want to use across all worker nodes. This example uses all available GPUs (`NUM_GPUS_PER_NODE * NUM_WORKERS`). You also change `local_mode` to `False`. Additionally, to configure how many GPUs to use for each Spark task that runs the train function, `set spark.task.resource.gpu.amount <num_gpus_per_task>` in the Spark Config cell on the cluster page before creating the cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC : org.apache.spark.SparkException: Job aborted due to stage failure: Could not recover from a failed barrier ResultStage. Most recent failure reason: Stage failed because barrier task ResultTask(5, 0) finished unsuccessfully.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC [W socket.cpp:432] [c10d] While waitForInput, poolFD failed with (errno: 0 - Success).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC E0319 23:40:35.118955 139795948843008 torch/distributed/elastic/agent/server/api.py:951] Error waiting on exit barrier. Elapsed: 300.10335206985474 seconds
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC [W socket.cpp:432] [c10d] While waitForInput, poolFD failed with (errno: 0 - Success).
# MAGIC E0319 23:40:35.118955 139795948843008 torch/distributed/elastic/agent/server/api.py:951] Error waiting on exit barrier. Elapsed: 300.10335206985474 seconds
# MAGIC E0319 23:40:35.118955 139795948843008 torch/distributed/elastic/agent/server/api.py:951] Traceback (most recent call last):
# MAGIC E0319 23:40:35.118955 139795948843008 torch/distributed/elastic/agent/server/api.py:951]   File "/databricks/python/lib/python3.11/site-packages/torch/distributed/elastic/agent/server/api.py", line 937, in _exit_barrier
# MAGIC E0319 23:40:35.118955 139795948843008 torch/distributed/elastic/agent/server/api.py:951]     store_util.barrier(
# MAGIC E0319 23:40:35.118955 139795948843008 torch/distributed/elastic/agent/server/api.py:951]   File "/databricks/python/lib/python3.11/site-packages/torch/distributed/elastic/utils/store.py", line 78, in barrier
# MAGIC E0319 23:40:35.118955 139795948843008 torch/distributed/elastic/agent/server/api.py:951]     synchronize(store, data, rank, world_size, key_prefix, barrier_timeout)
# MAGIC E0319 23:40:35.118955 139795948843008 torch/distributed/elastic/agent/server/api.py:951]   File "/databricks/python/lib/python3.11/site-packages/torch/distributed/elastic/utils/store.py", line 64, in synchronize
# MAGIC E0319 23:40:35.118955 139795948843008 torch/distributed/elastic/agent/server/api.py:951]     agent_data = get_all(store, rank, key_prefix, world_size)
# MAGIC E0319 23:40:35.118955 139795948843008 torch/distributed/elastic/agent/server/api.py:951]                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# MAGIC E0319 23:40:35.118955 139795948843008 torch/distributed/elastic/agent/server/api.py:951]   File "/databricks/python/lib/python3.11/site-packages/torch/distributed/elastic/utils/store.py", line 34, in get_all
# MAGIC E0319 23:40:35.118955 139795948843008 torch/distributed/elastic/agent/server/api.py:951]     data = store.get(f"{prefix}{idx}")
# MAGIC E0319 23:40:35.118955 139795948843008 torch/distributed/elastic/agent/server/api.py:951]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# MAGIC E0319 23:40:35.118955 139795948843008 torch/distributed/elastic/agent/server/api.py:951] torch.distributed.DistStoreError: Socket Timeout

# COMMAND ----------

# MAGIC %md
# MAGIC  An error occurred while calling o653.collectToPython.
# MAGIC : org.apache.spark.SparkException: Job aborted due to stage failure: Could not recover from a failed barrier ResultStage. Most recent failure reason: Stage failed because barrier task ResultTask(6, 0) finished unsuccessfully.

# COMMAND ----------

multi_node_dir = create_log_dir()
print("Data is located at: ", multi_node_dir)

from pyspark.ml.torch.distributor import TorchDistributor
import timeit

start_time = timeit.default_timer()

output_dist = TorchDistributor(num_processes=(NUM_GPUS_PER_NODE * NUM_WORKERS), local_mode=False, use_gpu=True).run(main_fn, multi_node_dir)

elapsed = timeit.default_timer() - start_time
print(f"Elapsed time: {elapsed} seconds")

test(multi_node_dir)

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------


