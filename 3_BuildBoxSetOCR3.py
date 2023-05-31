import json
import math
import os.path
import random
import sys
import math
import sys
import time
from os.path import basename

import torch.nn.functional as F
import pytesseract
import torch
import torchvision.models.detection.mask_rcnn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import copy
import torch.nn as nn
from torchvision.transforms import transforms

import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from PIL import Image
import torch
import torchvision
import numpy as np
from glob import glob
import cv2
import numpy as np
torch.cuda.empty_cache()
import gc
torch.cuda.memory_summary(device=None, abbreviated=False)
gc.collect()
from scipy import stats


def get_my_model_instance_segmentation(var, num_classes):
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False) : For the default
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")  # Can also use another model
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(f"models/model-{var}.pth")) # Should rename model first
    model.eval()
    ##model.load_state_dict(torch.load("model-xxx.pth"))  # To resume our progress
    return model

CHAR_DETECTION_THRESHOLD = 0.55
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if __name__ == '__main__':
    image_path_list = glob("all-dataset/*.jpg")
    random.shuffle(image_path_list)
    image_path_list = image_path_list[:10]
    image_path_list = image_path_list[:10]

    def get_transform():
        custom_transforms = []
        custom_transforms.append(torchvision.transforms.ToTensor())
        return torchvision.transforms.Compose(custom_transforms)
    transform = get_transform()

    batch_size = 1
    for offset in range(0, len(image_path_list), batch_size):
        print(f"Offset: {offset}")
        # Build the box set for OCR Training
        image_list = [Image.open(p) for p in image_path_list[offset:offset+batch_size]]  # Only for dataset, not for measurement
        image_tensors = [transform(i).to(device) for i in image_list]


        def export_cropped(bbox, name, image, label):
            bbox = bbox.cpu().data.numpy()
            cropped_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            outfile = f"test-crop/{basename(name)}"
            outfile = outfile.replace(".jpg", f"-{label}.jpg")
            cropped_image.save(outfile)

        # Character detection model
        cd_model = get_my_model_instance_segmentation("20-0.0025-0.0025", 2)
        cd_model.to(device)
        detection_result = cd_model(image_tensors)
        for i, (res, path) in enumerate(zip(detection_result, image_path_list[offset:offset+batch_size])):
            image = Image.open(path)
            bboxes = [b for b, s in zip(res['boxes'], res['scores']) if s > CHAR_DETECTION_THRESHOLD]
            bboxes.sort(key=lambda x: float(x[0]))
            for i,bbox in enumerate(bboxes):
                export_cropped(bbox, f"{basename(path).replace('.jpg', '')}-{str(i).zfill(2)}.jpg", image, "label")
        print()