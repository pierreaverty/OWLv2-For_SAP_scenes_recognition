import os
import requests
import torch

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transformers import Owlv2Processor
from models.owlv2 import OWLv2ForSapRecognition 
from utils.plotter import ObjectDetectionPlotter
from losses.owlv2 import CrossEntropyForOWLv2

from datasets.sap_scenes import SAPDetectionDataset
from dataloaders.sap_scenes import SAPDetectionDataLoader

from PIL import Image
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer



def main():
    owl = OWLv2ForSapRecognition(lr=1, weight_decay=1, loss=CrossEntropyForOWLv2())
    
    train_dataset = SAPDetectionDataset(directory="/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/train/", processor=owl.processor)
    val_dataset = SAPDetectionDataset(directory="/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/val/", processor=owl.processor)

    train_dataloader = SAPDetectionDataLoader(train_dataset, batch_size=5, shuffle=True)
    val_dataloader = SAPDetectionDataLoader(val_dataset, batch_size=5)

    trainer = Trainer(max_steps=2, gradient_clip_val=0.1, log_every_n_steps=1, precision=32, accelerator='gpu')
    trainer.fit(owl, train_dataloader, val_dataloader)

    path = "/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/train/images/woman1_front_1.jpg"
    image = Image.open(path)
    
    text = [["women1_front"]]
    results = owl.predict( image, text)
    plotter = ObjectDetectionPlotter(results, texts=text, image=image)

    plotter.plot_results("output")
    
if __name__ == "__main__":
    main()