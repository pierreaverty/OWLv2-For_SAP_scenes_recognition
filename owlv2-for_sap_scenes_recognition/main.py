import os
import requests
import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from utils.plotter import ObjectDetectionPlotter
from PIL import Image

class OWLv2SceneRecognition:
    def __init__(self):
        self.processor = Owlv2Processor()
        self.model = Owlv2ForObjectDetection()




def main():
    # Create an instance of the OWLv2SceneRecognition class
    owl = OWLv2SceneRecognition()
    
    


if __name__ == "__main__":
    main()