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
    def __init__(self, num_classes=1000):  # num_classes for ImageNet is 1000
        super(ResNet18, self).__init__()
        
        # Load the pre-trained ResNet-18 model from torchvision
        self.resnet = models.resnet18(pretrained=True)  # ResNet-18 architecture
        # model.fc = nn.Sequential(
        #     nn.Dropout(0.5),  # Add 50% dropout
        #     nn.Linear(model.fc.in_features, 1000)
        # )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Modify the final fully connected layer for the desired number of classes
        
    def forward(self, x):
        return self.resnet(x)

# COMMAND ----------

model = ResNet18(num_classes=1000).to(device) # For ImageNet Tiny

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC ILSVRC 2012, commonly known as 'ImageNet' is an image dataset organized according to the WordNet hierarchy. Each meaningful concept in WordNet, possibly described by multiple words or word phrases, is called a "synonym set" or "synset". There are more than 100,000 synsets in WordNet, majority of them are nouns (80,000+). ImageNet aims to provide on average 1000 images to illustrate each synset. Images of each concept are quality-controlled and human-annotated.

# COMMAND ----------

import os

HF_DATASETS_CACHE = f"/Volumes/will_smith/datasets/imagenet"
os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE

# COMMAND ----------

from datasets import load_dataset

imagenet_train = load_dataset('ILSVRC/imagenet-1k', split="train", trust_remote_code=True)
imagenet_val = load_dataset('ILSVRC/imagenet-1k', split="validation", trust_remote_code=True)

# COMMAND ----------

imagenet_train

# COMMAND ----------

imagenet_val

# COMMAND ----------

# MAGIC %md
# MAGIC ### TODO Use Pytorch transform instead of huggingface 

# COMMAND ----------

def transforms(examples):
    examples["label"] = [label % 50 + 1 for label in examples["label"]]
    return examples

subset_imagenet_train = imagenet_train.map(transforms, batched=True)
subset_imagenet_val = imagenet_val.map(transforms, batched=True)
# subset_imagenet_test = imagenet_test.map(transforms, batched=True)

# COMMAND ----------

# DBTITLE 1,Modify Datasets for Pytorch
from datasets import Dataset 
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image


# Apply transformations directly to the dataset
train_dataset =  Dataset.from_dict({"image": subset_imagenet_train['image'], "label": subset_imagenet_train['label']}).with_format("torch", device=device)
test_dataset =  Dataset.from_dict({"image": subset_imagenet_val['image'], "label": subset_imagenet_val['label']}).with_format("torch", device=device)

# Create DataLoaders with batching and shuffling
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# COMMAND ----------

# from torchvision import transforms
# from torch.utils.data import DataLoader
# from PIL import Image

# # Define transformations
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),  # Resize image to 64x64 (adjust as needed)
#     transforms.ToTensor(),  # Convert the PIL image to a tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet statistics
# ])

# # Transform function to apply to the images in the dataset
# def preprocess(example):
#     image = example['image'].convert('RGB')
#     label = example['label']
#     image = transform(image)  # Apply the transformations to the image
#     return {'image': image, 'label': label}

# # Apply transformations directly to the dataset
# train_dataset = tiny_imagenet_train.map(preprocess)
# test_dataset = tiny_imagenet_test.map(preprocess)

# # Create DataLoaders with batching and shuffling
# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# COMMAND ----------

import torch.optim as optim
import torch.nn as nn

# Loss function and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# COMMAND ----------

# MAGIC %md
# MAGIC # Training

# COMMAND ----------

# Training loop
num_epochs = 10
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

    scheduler.step()


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Inference
# MAGIC

# COMMAND ----------

tiny_imagenet_test[0]

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
    image = input['image'].convert('RGB')
    
    # Apply transformations (resize, crop, normalize)
    if transform:
        image = transform(image)
    else:
        # Default transformation for ResNet (resize, crop, normalize)
        default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = default_transform(image)
    # Add a batch dimension (as model expects a batch of images)
    image = image.unsqueeze(0).to(device)
    
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
    predicted_class = predict(tiny_imagenet_test[i], model, device)
    correct_label = tiny_imagenet_test[i]['label']
    print(f"Item {i}: Predicted Class: {predicted_class}, Correct Label: {correct_label}")

# COMMAND ----------


