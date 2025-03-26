# Configs from https://github.com/Data-drone/Deepspeed_distributor/blob/main/configs/deepspeed_configs.py

from copy import deepcopy

shared_parameters = {
   "gradient_accumulation_steps": 1,
   "gradient_clipping": 0.3,
   "per_device_batch_size": 4,
   "learning_rate": 2e-4,
   "warmup_steps": 100
}

# Base config for deepspeed
deepspeed_base = {
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": shared_parameters["per_device_batch_size"],
	"gradient_accumulation_steps": shared_parameters['gradient_accumulation_steps'],
  "gradient_clipping": shared_parameters['gradient_clipping'],
	"bf16": {
		"enabled": "true"
	},
  "optimizer": {
        "type": "AdamW",
        "params": {
          "lr": shared_parameters["learning_rate"],
          "betas": [
            0.9,
            0.999
          ],
          "eps": 1e-08
        }
      },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": shared_parameters["learning_rate"],
      "warmup_num_steps": shared_parameters["warmup_steps"],
      "warmup_type": "linear"
    }
  },
  "tensorboard": {
    "enabled": True,
    "output_path": '/local_disk0/tensorboard',
    "job_name": "finetune_llama_2_7b"
  },
  "steps_per_print": 10,
  "wall_clock_breakdown": True,
  "zero_optimization": {}
}

# ZeRo 1
deepspeed_zero_1 = deepcopy(deepspeed_base)
deepspeed_zero_1['zero_optimization'] = {
        "stage": 1,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "allgather_partitions": True,
        "allgather_bucket_size": 500000000,
        "reduce_scatter": True,
        "reduce_bucket_size": 500000000,
        "cpu_offload": False
      }

# ZeRo 2
deepspeed_zero_2 = deepcopy(deepspeed_base)
deepspeed_zero_2['zero_optimization'] = {
        "stage": 2,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto"
    }

# ZeRo 3
deepspeed_zero_3 = deepcopy(deepspeed_base)
deepspeed_zero_3['zero_optimization'] = {
        "stage": 3,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e7,
        "stage3_max_reuse_distance": 1e7,
        "stage3_gather_16bit_weights_on_model_save": True
    }

# ZeRo 3 Offload
deepspeed_zero_3_offload = deepcopy(deepspeed_base)
deepspeed_zero_3_offload['zero_optimization'] = {
        "stage": 3,
        "offload_optimizer": {
            "device": 'cpu'
        },
        "offload_param": {
            "device": 'cpu'
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e7,
        "stage3_max_reuse_distance": 1e7,
        "stage3_gather_16bit_weights_on_model_save": True
    }