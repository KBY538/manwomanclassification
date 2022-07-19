import torch
from torchvision import transforms
import os
import PIL.Image as Image


class CustomImageFolder():

  def __init__(self, data_dir, transforms=None):
    self.data_dir = data_dir
    self.transforms = transforms
    self.datapoints = 0
    self.classes = []
    self.class_to_idx = dict()
    self.imgs = []

    for name in os.listdir(data_dir):
      class_index = len(self.classes)
      path = os.path.join(data_dir, name)
      self.datapoints += len(os.listdir(path))
      if os.path.isdir(path):
        self.classes.append(name)
        self.class_to_idx[name] = class_index

        for filename in os.listdir(path):
          self.imgs.append(((os.path.join(path, filename)), class_index))


  def __getitem__(self, index):
    path, target = self.imgs[index]
    img = Image.open(path)

    if self.transforms != None:
      img = self.transforms(img)

    return img, target

  def __len__(self):
    return len(self.imgs)

  def __str__(self):
    return "Dataset ImageFolder\number of datapoints: {}\nRoot location: {}\nStandardTransform\nTransform:{}".format(len(self.imgs), self.data_dir, self.transforms)


class CustomDataset():

    # Data augmentation and normalization for training
    # Just normalization for validation

    def __init__(self, data_dir, crop=224, resize=256, batch_size=4, num_workers=1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transforms = [
                transforms.Resize(resize),
                transforms.RandomResizedCrop(crop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]

        self.data_transforms = {
            'train': transforms.Compose(self.transforms),
            'val': transforms.Compose(self.transforms),
        }

        self.image_datasets = {x: CustomImageFolder(os.path.join(data_dir, x),
                                                  self.data_transforms[x])
                          for x in ['train', 'val']}

        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size,
                                                    shuffle=True, num_workers=num_workers)
                      for x in ['train', 'val']}

        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        self.class_names = self.image_datasets['train'].classes