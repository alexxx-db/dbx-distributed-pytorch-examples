# Databricks notebook source
# %pip install -r requirements.txt
# %restart_python

# COMMAND ----------

catalog = "will_smith"
schema = "datasets"
num_nodes = 1

# COMMAND ----------

try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
except Exception as e:
    print(f"Error: Could not create catalog due to {e}")

try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
except Exception as e:
    print(f"Error: Could not create schema due to {e}")

# COMMAND ----------

import os
TARGET_HUGGINGFACE_HUB_CACHE = f"/local_disk0/hf_cache"
LOCAL_HUGGINGFACE_HUB_CACHE = "/root/.cache/huggingface/hub"

# os.environ["TRANSFORMERS_CACHE"] = TARGET_HUGGINGFACE_HUB_CACHE
os.environ["HF_HOME"] = TARGET_HUGGINGFACE_HUB_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = TARGET_HUGGINGFACE_HUB_CACHE
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "True"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

# COMMAND ----------

from huggingface_hub import login

try:
  login(token=dbutils.secrets.get('william_smith_secrets', 'HF_KEY'))
  print("Successfully logged in to huggingface!")
except Exception as e:
  print(f"Error: Could not log into huggingface due to {e}")

# COMMAND ----------

import os

username = spark.sql("SELECT current_user()").first()['current_user()']
username

browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()

db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

os.environ['DATABRICKS_HOST'] = db_host
os.environ['DATABRICKS_TOKEN'] = db_token

# COMMAND ----------

import yaml

config_path = "../setup/local_config.yaml"
with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

# COMMAND ----------

import mlflow

username = spark.sql("SELECT current_user()").first()['current_user()']

# Retrieve workspace URL and API token using dbutils notebook commands
db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

import mlflow

notebook_name = dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().get().split('/')[-1]

experiment_path = f'/Users/{username}/experiments/{notebook_name}'
experiment = mlflow.set_experiment(experiment_path)

# COMMAND ----------

import torch
 
NUM_WORKERS = int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers", "1"))
 
def get_gpus_per_worker(_):
  import torch
  return torch.cuda.device_count()
 
NUM_GPUS_PER_WORKER = sc.parallelize(range(4), 4).map(get_gpus_per_worker).collect()[0]

# COMMAND ----------

# MAGIC %sh 
# MAGIC
# MAGIC export TORCH_DISTRIBUTED_DEBUG="DETAIL"
# MAGIC export TORCH_CPP_LOG_LEVEL="INFO"
# MAGIC export TORCH_SHOW_CPP_STACKTRACES="1"
# MAGIC export NCCL_DEBUG="INFO"
# MAGIC export NCCL_DEBUG_SUBSYS="1"
