import os
import shutil
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_resnet18_pretrained(device, checkpoint=None, num_classes=2):
  model_ft = models.resnet18(pretrained=True)
  num_ftrs = model_ft.fc.in_features
  model_ft.fc = nn.Linear(num_ftrs, num_classes)

  if checkpoint != None:
    model_ft.load_state_dict(torch.load(checkpoint))

  model_ft = model_ft.to(device)
  return model_ft


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. '+directory)

def build_biclassify_dataset(data_dir, csv_file, attribute, class_names, n=1000, new_dir_name='dataset', frac=0.2):
  data = pd.read_csv(csv_file)

  data = [data[data[attribute] == -1], data[data[attribute] == 1]]

  data[0] = data[0].sample(n=n)
  data[1] = data[1].sample(n=n)

  if os.path.exists(new_dir_name):
    shutil.rmtree(new_dir_name)

  for name, d in zip(class_names, data):
    train_folder = new_dir_name + '/train/' + name
    val_folder = new_dir_name + '/val/' + name
    create_folder(train_folder)
    create_folder(val_folder)

    test_size = int(n*frac)

    for e, i in enumerate(d.index):
      filename = d["image_id"][i]
      if e <= test_size:
        shutil.copyfile(os.path.join(data_dir, filename), os.path.join(val_folder, filename))
      else:
        shutil.copyfile(os.path.join(data_dir, filename), os.path.join(train_folder, filename))
  
  return new_dir_name

def imshow(inp, title=None, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, dataset, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for inputs, labels in dataset.dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {dataset.class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)