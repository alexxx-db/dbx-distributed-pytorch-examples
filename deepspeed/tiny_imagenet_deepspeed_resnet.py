# Databricks notebook source
# MAGIC %run ../setup/00_setup

# COMMAND ----------

# MAGIC %pip install deepspeed 
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %sh nvidia-smi

# COMMAND ----------

import mlflow

username = spark.sql("SELECT current_user()").first()['current_user()']
username

experiment_path = f'/Users/{username}/deepspeed-distributor'

# Retrieve workspace URL and API token using dbutils notebook commands
os.environ['DATABRICKS_HOST'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

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

HF_DATASETS_CACHE = "/Volumes/will_smith/datasets/imagenet_tiny"
os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Splits
# MAGIC
# MAGIC #### Total Rows: 110,000 (0.35 gb)
# MAGIC
# MAGIC
# MAGIC | Split       | # of examples |
# MAGIC |-------------|---------------|
# MAGIC | Train       | 100,000 |
# MAGIC | Validation  | 10,000      |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Load dataset does not segment files to download based on split 
# MAGIC https://discuss.huggingface.co/t/how-can-i-download-a-specific-split-of-a-dataset/79027
# MAGIC

# COMMAND ----------

# DBTITLE 1,Load the dataset from cache to avoid redownloading
from datasets import load_dataset
import datasets

datasets.utils.logging.disable_progress_bar()
tiny_imagenet = load_dataset('zh-plus/tiny-imagenet', cache_dir=HF_DATASETS_CACHE, trust_remote_code=True)

# COMMAND ----------

tiny_imagenet

# COMMAND ----------

tiny_imagenet_train = tiny_imagenet['train']
tiny_imagenet_test = tiny_imagenet['valid']

# COMMAND ----------

def transforms(examples):
    examples["image"] = [image.convert("RGB").resize((64,64)) for image in examples["image"]]
    return examples

tiny_imagenet_train = tiny_imagenet_train.map(transforms, batched=True)
tiny_imagenet_test = tiny_imagenet_test.map(transforms, batched=True)

# COMMAND ----------

labels = set(tiny_imagenet["train"]["label"])
num_class = len(labels)
num_class

# COMMAND ----------

print(f"Dataset size: {tiny_imagenet['train'].info.size_in_bytes / (1024 ** 3):.2f} gb")

# COMMAND ----------

import os 

os.environ['MLFLOW_TRACKING_URI']
# os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_path
# os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING']

# COMMAND ----------

# MAGIC %md
# MAGIC # Train with DeepSpeed without Distributor

# COMMAND ----------

import argparse
import os
import deepspeed
import torch
import torchvision
import torchvision.transforms as transforms
import time
from datasets import Dataset 

# COMMAND ----------

from torchvision.models import ResNet18_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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

# COMMAND ----------

# DBTITLE 1,Hyperparameters
batch_size = 64
epochs = 5

# COMMAND ----------

# Apply transformations directly to the dataset
train_dataset =  Dataset.from_dict({"image": tiny_imagenet_train['image'], "label": tiny_imagenet_train["label"]}).with_format("torch", device="cuda")
test_dataset =  Dataset.from_dict({"image": tiny_imagenet_test['image'], "label": tiny_imagenet_test["label"]}).with_format("torch", device="cuda")

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# COMMAND ----------


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# COMMAND ----------

def full_train_loop(peft_config, training_arguments, dataset, 
                    distributor:bool=True, mlflow_parent_run=None):

    """
    Deepspeed isn't handling train_batch here if it is string:
    https://github.com/microsoft/DeepSpeed/blob/f57fc4c95a6a5194757b57704f60f009dde25680/deepspeed/runtime/config.py#L903
    guessing we need to specify a batch size which means we need to calculate it all first
    """

    import os
    import mlflow

    import torch
    from torch.utils.data import DataLoader
    
    import deepspeed
    from deepspeed import get_accelerator
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

    from transformers import (
       AutoModelForCausalLM, AutoTokenizer,
       DataCollatorForLanguageModeling
    )
    from peft import get_peft_model
    from deepspeed.utils import logger as ds_logger

    os.environ['MLFLOW_TRACKING_URI'] = 'databricks'
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_path
    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'true'
    #os.environ['HF_MLFLOW_LOG_ARTIFACTS'] = 'True'
    
    os.environ['DATABRICKS_HOST'] = db_host
    os.environ['DATABRICKS_TOKEN'] = db_token

    if distributor:
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['NCCL_P2P_DISABLE'] = '1'

    mlflow.set_registry_uri('databricks')

    model_path = f'{model_cache_root}/llama_3_1_8b/'

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    device = torch.device(get_accelerator().device_name())
    global_rank = torch.distributed.get_rank()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        #device_map = {"":int(local_rank)},
        torch_dtype=torch.bfloat16,
        cache_dir=model_path,
        local_files_only=True,
        low_cpu_mem_usage=False
    )

    model = get_peft_model(model, peft_config)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=model_path)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = setup_data(tokenizer, dataset)

    # removve string columns
    train_dataset = train_dataset.remove_columns(['text', 'category', 'instruction', 
                                                 'context', 'response'])

    # setup trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    train_dataloader = DataLoader(
       train_dataset,
       collate_fn = data_collator,
       batch_size = training_arguments.per_device_train_batch_size
    )

    deepspeed_arg = training_arguments.deepspeed
    ds_logger.info(f'deepspeed argument is of type: {type(deepspeed_arg)}')

    # Deepspeed Args can be a dict or a string
    ## When it is a string we need to load the file first into a dict
    if type(deepspeed_arg) == str:
        import json
        with open(training_arguments.deepspeed, 'r') as file:
            deepspeed_config_load = json.load(file)

    elif type(deepspeed_arg) == dict:
        deepspeed_config_load = deepspeed_arg

    try: 
        offload_device = deepspeed_config_load['zero_optimization']['offload_optimizer']['device']
        ds_logger.info(f'DeepSpeed Offload: {offload_device}')
    except (TypeError, KeyError) as e:
        ds_logger.info(f'Offload detection error: {e}')
        offload_device = None

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

    model = initialised_var[0]
    optimizer = initialised_var[1]

    # with manual loop we will have to add manual mlflow
    # variables:
    # training_arguments, model, train_dataloader
    # device 

    ## setup distributed mlflow system metric logging
    ## We want to log system usage on all nodes so we need to make sure that they are all
    ## nested back with the correct parent

    if mlflow_parent_run:
        from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID  
        run_tags = {MLFLOW_PARENT_RUN_ID: mlflow_parent_run.info.run_id}
    else:
        run_tags = {}

    ds_logger.info(f'run_tags are: {run_tags}')

    # We want to log all configs and track loss on our primary node
    # But not on the other mlflow runs that exist just to log system stats
    if global_rank == 0:
        active_run = mlflow.start_run(run_name=training_arguments.run_name,
                                      tags=run_tags)

        # Manually log the training_arguments
        mlflow.log_params(training_arguments.to_dict())

        ## Deepspeed config needs to be unpacked separately
        ## some DS variables overlap with HF ones
        mod_ds_args = {"ds_" + key: value for key, value in deepspeed_config_load.items()}
        mlflow.log_params(mod_ds_args)
    
    else:
        active_run = mlflow.start_run(run_name=f"{training_arguments.run_name}_rank_{global_rank}",
                                      tags=run_tags)


    # Now we can start the run loop
    for epoch in range(training_arguments.num_train_epochs):
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
          if global_rank == 0:
            mlflow.log_metrics(metrics=run_dict, step=step) if global_rank == 0 else None

    return 'done'
