import os
import requests
import torch

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from utils.plotter import ObjectDetectionPlotter

from datasets.sap_scenes import SAPWomenDetectionDataset
from dataloaders.sap_scenes import SAPDetectionDataLoader

from PIL import Image
from torch.utils.data import DataLoader




def main():
    # Create an instance of the OWLv2SceneRecognition class
    owl = OWLv2SceneRecognition()
    train_dataset = SAPWomenDetectionDataset(images_directory="/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/train/images", labels_directory="/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/train/labels", processor=owl.processor)
    val_dataset = SAPWomenDetectionDataset(images_directory="/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/val/images", labels_directory="/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/val/labels", processor=owl.processor)
    
    train_dataloader = SAPDetectionDataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = SAPDetectionDataLoader(val_dataset, batch_size=2)
    batch = next(iter(train_dataloader))
    
    
if __name__ == "__main__":
    main()