# Distributed PyTorch Training on Databricks

This repository contains examples and utilities for performing distributed deep learning training on Databricks using various frameworks with PyTorch. The examples focus on image classification tasks using popular datasets and model architectures.

## Overview

Training deep learning models at scale requires distributed training capabilities. This repository demonstrates how to leverage different distributed training frameworks on Databricks to accelerate model training for computer vision tasks.

## Supported Datasets

- CIFAR-10/100
- TinyImageNet
- ImageNet-1K

## Model Architectures

- ResNet18
- ResNet50

## Frameworks

This repository includes examples for the following distributed training frameworks:

1. **PyTorch Distributor** - Native PyTorch distributed training with Databricks' Torch Distributor
2. **DeepSpeed** - Microsoft's deep learning optimization library
3. **Composer** - MosaicML's training library for efficient deep learning
4. **Accelerate** - Hugging Face's library for easy distributed training
5. **Ray** - Distributed computing framework with PyTorch integration

## Repository Structure

```
.
├── 01_torch_distributor/  # Examples using PyTorch's native distributed capabilities with Torch Distributor
├── 02_deepspeed/          # Examples using Microsoft DeepSpeed
├── 03_composer/           # Examples using MosaicML Composer
├── 04_accelerate/         # Examples using Hugging Face Accelerate
├── 05_ray/                # Examples using Ray for distributed training
├── setup/                 # Setup scripts and utilities for Databricks clusters
├── utils/                 # Common utility functions for data loading, metrics, etc.
├── .gitignore             # Git ignore file
├── LICENSE                # License information
└── requirements.txt       # Python package dependencies
```

## Getting Started

### Prerequisites

- Databricks Runtime for Machine Learning (DBR ML) 12.0 or later
- GPU-enabled Databricks cluster

### Installation

1. Clone this repository to your Databricks workspace:

```bash
git clone https://github.com/username/distributed-pytorch-databricks.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your Databricks cluster using the setup scripts provided in the `setup/` directory.

## Usage Examples

Each framework directory contains notebooks and scripts that demonstrate:

- Data loading and preprocessing for the supported datasets
- Model definition and configuration
- Distributed training setup and execution
- Evaluation and metrics tracking
- Integration with Databricks MLflow for experiment tracking

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgements

- PyTorch team
- Microsoft DeepSpeed
- MosaicML Composer
- Hugging Face Accelerate
- Ray Project
- Databricks for their ML runtime and infrastructure
