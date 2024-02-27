import os
import requests
import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from utils.plotter import ObjectDetectionPlotter
from datasets.sap_scenes import SAPWomenDetectionDataset
from PIL import Image

class OWLv2SceneRecognition:
    """
    OWLv2SceneRecognition class for scene recognition using OWLv2 model.

    Attributes:
        processor (Owlv2Processor): The processor for text and image inputs.
        model (Owlv2ForObjectDetection): The OWLv2 model for object detection.

    Methods:
        predict(image, texts): Predicts the scene in the given image with the provided texts.

    """

    def __init__(self):
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model =  Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")


    def predict(self, image, texts):
        """
        Predicts the scene in the given image with the provided texts.

        Args:
            image: The input image for scene recognition.
            texts: The input texts for scene recognition.

        Returns:
            results: The predicted scene results in Pascal VOC format.

        """
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

        return results


def main():
    # Create an instance of the OWLv2SceneRecognition class
    owl = OWLv2SceneRecognition()
    train_dataset = SAPWomenDetectionDataset(images_directory="/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/train/images", labels_directory="/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/train/labels", processor=owl.processor)
    val_dataset = SAPWomenDetectionDataset(images_directory="/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/val/images", labels_directory="/home/omilab-gpu/OWLv2-For_SAP_scenes_recognition/data/val/labels", processor=owl.processor)

   

if __name__ == "__main__":
    main()