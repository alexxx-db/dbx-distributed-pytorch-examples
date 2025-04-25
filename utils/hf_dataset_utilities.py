from datasets import load_dataset
import datasets
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import timeit

def hfds_download_volume(
  hf_cache: str, 
  dataset_path: str, 
  trust_remote_code: bool = False, 
  disable_progress: bool = False, 
):
  dataset_dict = load_dataset(
    path = dataset_path, 
    cache_dir=hf_cache, trust_remote_code=trust_remote_code)
  
  return dataset_dict

def hf_get_num_classes(
  dataset,
  split_key: str, 
  label_key: str = "label",
  ) -> int:

  labels = set(dataset[split_key][label_key])
  num_class = len(labels)
  return num_class


def create_torch_image_dataset(
  image_key: str, 
  label_key: str,
):

  class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.images = data[image_key]
        self.labels = data[label_key]
        self.transform = transform
        self.num_classes = len(set(data[label_key]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
  
  return CustomDataset


def default_image_transforms(
  image_size: int,
  normalize_transform: bool = True,
  convert_rgb: bool = True, 
):
  
  transform_list = [
    transforms.Resize((image_size, image_size)),  # Resize to specified size
    transforms.RandomHorizontalFlip(),            # Random horizontal flip
    transforms.ToTensor()                         # Convert PIL image to Tensor
  ]
  
  if convert_rgb:
    transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x))  # Convert grayscale to RGB
  
  if normalize_transform:
    transform_list.append(transforms.Normalize(  # Normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],               # RGB mean
        std=[0.229, 0.224, 0.225]                 # RGB std
    ))
  
  default_transforms = transforms.Compose(transform_list)

  return default_transforms

class Timer:
  def __init__(self):
    self.start = timeit.default_timer()

  def stop(self):
    self.end = timeit.default_timer()
    return self.end - self.start

