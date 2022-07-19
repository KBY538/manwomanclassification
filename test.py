import torch
import time

from data import CustomDataset
from utils import build_biclassify_dataset, load_resnet18_pretrained, visualize_model

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
VISUALIZE_IMGS = 6
TEST_DATA_RATIO = 1

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
    parser.add_argument("--visualize-imgs", type=int, default=VISUALIZE_IMGS,
                        help="Number of images to visualize.")
    parser.add_argument("--test-data-ratio", type=float, default=TEST_DATA_RATIO,
                        help="Test data split ratio.")
    return parser
    

class Test():

    def __init__(self, model, dataset, device):
        self.device = device
        self.dataset = dataset
        self.model = model

    def get_model(self):
      return self.get_model

    def test_model(self):
        running_corrects = 0
        
        since = time.time()

        self.model.eval()
        for inputs, labels in self.dataset.dataloaders['val']:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

        final_accr = running_corrects.double() / self.dataset.dataset_sizes['val']

        time_elapsed = time.time() - since
        print(f'Test complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Model Acc: {final_accr:4f}')


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
    tester = Test(model, face_dataset, device)
    tester.test_model()
    visualize_model(model, face_dataset, device, args.visualize_imgs)

if __name__ == '__main__':
    main()