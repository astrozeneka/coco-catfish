import math
import sys
import math
import sys
import time
import argparse

from utils import *
import utils
from torch.utils.data import random_split

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
# For the annotation part (see the commit part)
from engine import *
data_dir = "dataset"
coco_file = "dataset/_annotations.coco.json"

batchSize = {
    'train': 1,
    'valid': 1,
    'test': 1
}

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=str, default=5, help="Epoch number")
parser.add_argument("-r", "--rate", type=float, default=0.003, help="Learning rate")
parser.add_argument("-o", "--outfile", type=str, default="model.pth", help="Out file")
parser.add_argument("-d", "--decay", type=float, default=0.003, help="Weight decay")
args = parser.parse_args()
num_workers_dl = 4
num_epochs = int(args.epochs)
lr = args.rate
momentum = 0.9
weight_decay = args.decay # The weight decay
import torchvision.transforms as transforms
import random

if __name__ == '__main__':
    def get_transform():
        custom_transforms = []
        custom_transforms.append(
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)],
                                   p=0.5))
        custom_transforms.append(
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5))
        custom_transforms.append(torchvision.transforms.ToTensor())
        return torchvision.transforms.Compose(custom_transforms)
    all_dataset = myOwnDataset(root=data_dir, annotation=coco_file, transforms=get_transform())
    datasets = {}
    datasets["train"], datasets["valid"] = random_split(all_dataset, [
        int(0.8 * len(all_dataset)),
        int(0.2 * len(all_dataset)) + 1  # IMPORTANT, for non integer quotient
    ])
    dataloader = {
        phase: torch.utils.data.DataLoader(
            datasets[phase],
            shuffle=True,
            num_workers=num_workers_dl,
            collate_fn=collate_fn,
            batch_size=3
        )
        for phase in ['train', 'valid']
    }
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def get_my_model_instance_segmentation(num_classes):
        # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False) : For the default
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")  # Can also use another model
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        ##model.load_state_dict(torch.load("model-xxx.pth"))  # To resume our progress
        return model

    model = get_my_model_instance_segmentation(2) # SHould add +1 for the backgbround
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    len_dataloader = {
        l: len(dataloader[l]) for l in ['train']
    }
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    for epoch in range(num_epochs):
        res_train = m_train_one_epoch(model, optimizer, dataloader['train'], device, epoch, 10)
        lr_scheduler.step()
        res_eval = m_evaluate(model, dataloader['valid'], device=device)
    outfile = f"models/{args.outfile}"
    torch.save(model.state_dict(), outfile)
    print("Training done")
