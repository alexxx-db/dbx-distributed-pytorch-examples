# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os
TARGET_HUGGINGFACE_HUB_CACHE = f"/local_disk0/hf_cache"
LOCAL_HUGGINGFACE_HUB_CACHE = "/root/.cache/huggingface/hub"

os.environ["TRANSFORMERS_CACHE"] = TARGET_HUGGINGFACE_HUB_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = TARGET_HUGGINGFACE_HUB_CACHE
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "True"

# COMMAND ----------

from huggingface_hub import login

try:
  login(token=dbutils.secrets.get('william_smith_secrets', 'HF_KEY'))
  print("Successfully logged in to huggingface!")
except Exception as e:
  print(f"Error: Could not log into huggingface due to {e}")
