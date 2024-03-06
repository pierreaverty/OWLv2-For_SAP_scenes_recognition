import os
import datetime
import torchvision

from PIL import Image
from torch.utils.data import Dataset
import json

class SAPDetectionDataset(torchvision.datasets.CocoDetection):
    """
    A custom dataset class for SAP detection.

    Args:
        images_directory (str): The directory path containing the images.
        labels_directory (str): The directory path containing the labels.

    Attributes:
        images_directory (str): The directory path containing the images.
        labels_directory (str): The directory path containing the labels.
        images_files (list): List of image file names in the images directory.
        labels_files (list): List of label file names in the labels directory.
        dataset (list): List of dictionaries representing the dataset.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(int, array): Returns the item at the given index.
        _create_dataset(): Creates the dataset by loading images and labels.
    """

    def __init__(self, directory, processor=None):
        self.directory = directory if directory.endswith("/") else directory + "/"

        
        self.images_directory = directory+"images"
        self.labels_directory = directory+"labels"
        
        self.images_files = os.listdir(self.images_directory)
        self.labels_files = os.listdir(self.labels_directory)
        
        self._create_dataset()
        
        super(SAPDetectionDataset, self).__init__(self.images_directory, self.data_path)
        
        self.processor = processor

    def __getitem__(self, idx):
        """
        Retrieves the image and target label at the given index.

        Args:
            idx (int): The index of the image and target label to retrieve.

        Returns:
            tuple: A tuple containing the pixel values of the image and the target label.
        """
        image, label = super(SAPDetectionDataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        
        target = {
            'image_id': label[0]["image_id"], 
            'annotations': "women1_front",
            'category_id': 0,
            'bbox': label[0]["bbox"],
            'objectness': 1
        }

        encoding = self.processor(images=image, text=target["annotations"], return_tensors="pt")
        
        pixel_values = encoding["pixel_values"].squeeze()
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        
                
        return pixel_values, input_ids, attention_mask, target
    
    def _dataset_init(self):
        """
        Initializes the dataset dictionary with metadata.
        """
        self.dataset = {
            "info": {
                "description": "SAP Detection Dataset",
                "url": "https://github.com/pierreaverty/OWLv2-For_SAP_scenes_recognition/",
                "version": "0.1.0",
                "year": 2024,
                "contributor": "pierreaverty",
                "date_created": datetime.datetime.utcnow().isoformat(' ')
            },
            
            "licenses": {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            },
            "images": [],
            "annotations": [],
            "categories": [{
                "supercategory": "N/A",
                "id": 0,
                "name": "woman1_front"
            }]
        }
        
    def _create_dataset(self):
        """
        Creates the dataset by loading images and labels.
        """
        
        self._dataset_init()
        
        category_id = 0
        for i, image_file in enumerate(self.images_files):
            self._create_coco_object(image_file, i, category_id)
        
        with open(f'{self.directory}label.json', 'w') as f:
            json.dump(self.dataset, f)

        self.dataset = None
        self.data_path = self.directory+"label.json"

    def _create_coco_object(self, image_file, i, category_id):
        """
        Creates a COCO object for a given image file and adds it to the dataset.

        Args:
            image_file (str): The file name of the image.
            i (int): The index of the image.
            category_id (int): The category ID of the image.
        """
        image = Image.open(os.path.join(self.images_directory, image_file))
        label_file = open(os.path.join(self.labels_directory, image_file.split(".")[0] + ".txt"), "r")
        label_content = label_file.read().splitlines()[0].split(" ")
        label = [float(label) for label in label_content]
        
        image_info = {
            "id": i,
            "file_name": os.path.join(self.images_directory, image_file),
            "width": image.width,
            "height": image.height,
            "coco_url": "",
            "flickr_url": "",
            "date_captured": datetime.datetime.utcnow().isoformat(' '),
            "license_id": 1
        }
        annotation = {
            "id": i,
            "image_id": i,
            "category_id": category_id,                   
            "bbox": [label[1] * image.width, label[2] * image.height, label[3] * image.width, label[4] * image.height],
            "area": (label[3] * image.width) * (label[4] * image.height),
            "segmentation": [label[1] * image.width, label[2] * image.height, label[3] * image.width + image.width, label[4] * image.height + image.height],
            "iscrowd": 0
        }
        
        self.dataset["images"].append(image_info)
        self.dataset["annotations"].append(annotation)
