import os
import requests
import torch

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transformers import Owlv2Processor
from models.tuned import OWLv2ForSapRecognition 
from utils.plotter import ObjectDetectionPlotter

from datasets.sap_scenes import SAPDetectionDataset
from dataloaders.sap_scenes import SAPDetectionDataLoader

from PIL import Image
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer



def main():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark =  True
    torch.backends.cudnn.enabled =  True
    
    owl = OWLv2ForSapRecognition(lr=1e-4, weight_decay=1e-4)
    
    train_dataset = SAPDetectionDataset(directory="/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/train/", processor=owl.processor)
    val_dataset = SAPDetectionDataset(directory="/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/val/", processor=owl.processor)

    train_dataloader = SAPDetectionDataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = SAPDetectionDataLoader(val_dataset, batch_size=1)

    trainer = Trainer(max_steps=300, gradient_clip_val=0.1, log_every_n_steps=1, precision=16)
    trainer.fit(owl, train_dataloader, val_dataloader)
    
if __name__ == "__main__":
    main()