# Databricks notebook source
# MAGIC %run ../setup/00_setup

# COMMAND ----------

import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
display(device)

# COMMAND ----------

# MAGIC %md
# MAGIC # Resnet definition 

# COMMAND ----------

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):  # num_classes for cifar is 10 
        super(ResNet18, self).__init__()
        
        # Load the pre-trained ResNet-18 model from torchvision
        self.resnet = models.resnet18(pretrained=True)  # ResNet-18 architecture

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

# DBTITLE 1,Modify Datasets for Pytorch
from datasets import Dataset 
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

batch_size = 128

# Apply transformations directly to the dataset
train_dataset =  Dataset.from_dict({"image": cifar_dataset['train']['img'], "label": cifar_dataset["train"]["label"]}).with_format("torch", device=device)

test_dataset =  Dataset.from_dict({"image": cifar_dataset['test']['img'], "label": cifar_dataset["test"]["label"]}).with_format("torch", device=device)

# Create DataLoaders with batching and shuffling
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# COMMAND ----------

# # from datasets import Dataset
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from PIL import Image

# batch_size = 128

# # Define a set of transformations to apply to the images
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),  # Random horizontal flip for data augmentation
#     transforms.RandomRotation(10),      # Random rotation
#     transforms.RandomCrop(32, padding=4),  # Random crop with padding (commonly used in CIFAR)
#     transforms.ToTensor(),              # Convert PIL Image to Tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images (adjust the mean and std values based on your dataset)
# ])

# # Define a custom Dataset class to apply the transformations
# class CifarCustomDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, transform=None):
#         self.dataset = dataset
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         # Fetch image and label
#         image = self.dataset['img'][idx]  # Convert from numpy array to PIL Image
#         label = self.dataset['label'][idx]

#         # Apply transformation if available
#         if self.transform:
#             image = self.transform(image)

#         return {'image': image, 'label': label}

# # Apply transformations to the training and test datasets
# train_dataset = CifarCustomDataset(cifar_dataset['train'], transform=transform)
# test_dataset = CifarCustomDataset(cifar_dataset['test'], transform=transform)

# # Create DataLoaders with batching and shuffling
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# COMMAND ----------

# MAGIC %md
# MAGIC # Training

# COMMAND ----------

if 'model' in locals():
    del model
model = ResNet18(num_classes=num_class).to(device) # For medical image dataset

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

import torch.optim as optim
import torch.nn as nn

# Set seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
learning_rate = 1e-5
betas = (0.9, 0.999)
epsilon = 1e-08
lr_scheduler_warmup_ratio = 0.1

# Loss function and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=epsilon, weight_decay=0)

# Scheduler (linear with warmup)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_scheduler_warmup_ratio if epoch < num_epochs * lr_scheduler_warmup_ratio else 1)

# COMMAND ----------

# Training loop with early stopping
num_epochs = 100
patience = num_epochs / 10 
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Train the model
    for batch in train_dataloader:
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

    epoch_loss = running_loss / len(train_dataloader)
    epoch_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Evaluate on validation data after every epoch
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels = batch['image'].float().to(device), batch['label'].long().to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(test_dataloader)
    val_acc = val_correct / val_total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # Check for overfitting or underfitting
    if epoch_acc > 0.95 and val_acc < 0.7:
        print("Warning: Potential overfitting detected.")
    elif epoch_acc < 0.7 and val_acc < 0.7:
        print("Warning: Potential underfitting detected.")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    # scheduler.step()  


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

# Ensure the model is on the correct device (GPU or CPU)
model = model.to(device)  

for i in range(100):
    predicted_class = predict(cifar_dataset['test'][i], model, device)
    correct_label = cifar_dataset['test'][i]['label']
    print(f"Item {i}: Predicted Class: {predicted_class}, Correct Label: {correct_label}")

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


