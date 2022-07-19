import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import time
import os

from utils import build_biclassify_dataset, load_resnet18_pretrained, create_folder
from data import CustomDataset

import argparse

BATCH_SIZE = 4
DATA_DIR = './dataset/'
DATA_CSV = 'list_attr_celeba.csv'
DATA_ATTR = 'Male'
CLASSES_NAME = ['female', 'male']
DATA_SIZE = 1000
CROP_SIZE = 224
RESIZE_SIZE = 256
NUM_WORKERS = 4
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CHECKPOINT = None
SAVE_DIR = './checkpoints/'
SAVE_EVERY = 5
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
MOMENTUM = 0.9
STEP_SIZE = 7
GAMMA=0.1
TEST_DATA_RATIO = 0.2

def get_parser():
    parser = argparse.ArgumentParser(description="transfer learing assignment")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--data-csv", type=str, default=DATA_CSV,
                        help="Dataset csv.")
    parser.add_argument("--data-attr", type=str, default=DATA_ATTR,
                        help="Data attribute we focus.")
    parser.add_argument("--classes_name", nargs="+", default=CLASSES_NAME,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--data-size", type=int, default=DATA_SIZE,
                        help="Data size.")
    parser.add_argument("--prepared", action='store_true',
                        help="Dataset directory has proper structure to train.")
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE,
                        help="Crop size.")
    parser.add_argument("--resize-size", type=int, default=RESIZE_SIZE,
                        help="Resize size.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Number of workers.")
    parser.add_argument("--mean", type=list, default=MEAN,
                        help="Mean of channels.")
    parser.add_argument("--std", type=list, default=STD,
                        help="Std of channels.")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT,
                        help="Path to the directory containing the model file to load.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to the directory to save model.")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY,
                        help="Save every n epochs.")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                        help="Number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum.")
    parser.add_argument("--step-size", type=int, default=STEP_SIZE,
                        help="Step size.")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                        help="Gamma.")
    parser.add_argument("--test-data-ratio", type=float, default=TEST_DATA_RATIO,
                        help="Test data split ratio.")
    return parser
    

class Train():

    def __init__(self, model, dataset, device, num_epochs):
        
        self.num_epochs = num_epochs
        self.device = device
        self.dataset = dataset
        self.model = model

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def train_model(self, save_dir, save_every, num_epochs):
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler
        
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataset.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset.dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()
            if epoch%save_every == 0:
                self.save_model(epoch, save_dir)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        self.save_model(epoch, save_dir, "training_complete.pt")
        

    def save_model(self, epoch, save_dir, name=None):

        create_folder(save_dir)
        
        now = time
        if name == None:
            name = now.strftime('t%Y_%m_%d_%Hh_%Mm_%Ss.pt')
        
        name = '{}_'.format(epoch+1) + name
        path = os.path.join(save_dir, name)
        torch.save(self.model.state_dict(), path)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = get_parser()
    args = parser.parse_args()
    
    if args.prepared:
        data_dir = args.data_dir
    else:
        data_dir = build_biclassify_dataset(args.data_dir, args.data_csv, args.data_attr, args.classes_name, args.data_size, frac=args.test_data_ratio)

    face_dataset = CustomDataset(data_dir, crop=args.crop_size, resize=args.resize_size, batch_size=args.batch_size, num_workers=args.num_workers, mean=args.mean, std=args.std)
    model = load_resnet18_pretrained(device, args.checkpoint)
    trainer = Train(model=model, dataset=face_dataset, device=device, num_epochs=args.num_epochs)
    trainer.train_model(args.save_dir, args.save_every, args.num_epochs)

if __name__ == '__main__':
    main()