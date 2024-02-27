import os
import datetime

from PIL import Image
from torch.utils.data import Dataset

class SAPDetectionDataset(Dataset):
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

    def __init__(self, images_directory, labels_directory, processor=None):
        self.images_directory = images_directory
        self.labels_directory = labels_directory
        self.images_files = os.listdir(images_directory)
        self.labels_files = os.listdir(labels_directory)
        self.processor = processor
        self._create_dataset()

    def __len__(self):
        """
        Returns the length of the dataset.
        
        Returns:
            int: The length of the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves the image and target label at the given index.

        Args:
            idx (int): The index of the image and target label to retrieve.

        Returns:
            tuple: A tuple containing the pixel values of the image and the target label.
        """
        image, image_id = self.dataset["images"][idx]["image"], self.dataset["images"][idx]["id"]
        target = self.dataset["categories"][ self.dataset["annotations"][idx]["category_id"]]["name"]
            
        target = {
            'image_id': image_id, 
            'annotations': target
        }

        encoding = self.processor(images=image, text=target["annotations"], return_tensors="pt")
        
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]

        return pixel_values, target, attention_mask
    
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
    
    
    def _create_coco_object(self, image_file, i, category_id):
        """
        Creates a COCO object for a given image file and adds it to the dataset.

        Args:
            image_file (str): The file name of the image.
            i (int): The index of the image.
            category_id (int): The category ID of the image.

        Returns:
            None
        """
        image = Image.open(os.path.join(self.images_directory, image_file))
        label_file = open(os.path.join(self.labels_directory, image_file.split(".")[0] + ".txt"), "r")
        label_content = label_file.read().splitlines()[0].split(" ")
        label = [float(label) for label in label_content]
        
        image_info = {
            "id": i,
            "file_name": os.path.join(self.images_directory, image_file),
            "image": image,
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
    
    def get_annotation(self, idx):
        """
        Returns the annotation at the given index.

        Args:
            idx (int): The index of the annotation to retrieve.

        Returns:
            dict: The annotation at the given index.
        """
        return self.dataset["annotations"][idx]
    
    def get_image(self, idx):
        """
        Returns the image at the given index.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            dict: The image at the given index.
        """
        return self.dataset["images"][idx]
    
    def get_category(self, idx):
        """
        Returns the category at the given index.

        Args:
            idx (int): The index of the category to retrieve.

        Returns:
            dict: The category at the given index.
        """
        return self.dataset["categories"][idx]
    
    @property
    def annotations(self):
        """
        Returns the annotations for the dataset.

        Returns:
            list: The annotations for the dataset.
        """
        return self.dataset["annotations"]
    
    @property
    def annotation_ids(self):
        """
        Returns the annotation IDs for the dataset.

        Returns:
            list: The annotation IDs for the dataset.
        """
        return [annotation["id"] for annotation in self.dataset["annotations"]]
    
    @property
    def images(self):
        """
        Returns the images for the dataset.

        Returns:
            list: The images for the dataset.
        """
        return [image["image"] for image in self.dataset["images"]]
    
    @property
    def image_ids(self):    
        """
        Returns the image IDs for the dataset.

        Returns:
            list: The image IDs for the dataset.
        """
        return [image["id"] for image in self.dataset["images"]]
    
    @property
    def categories(self):
        """
        Returns the categories for the dataset.

        Returns:
            list: The categories for the dataset.
        """
        return [cat["name"] for cat in self.dataset["categories"]]
    
    @property
    def category_ids(self):
        """
        Returns the category IDs for the dataset.

        Returns:
            list: The category IDs for the dataset.
        """
        return [cat["id"] for cat in self.dataset["categories"]]

class SAPWomenDetectionDataset(SAPDetectionDataset):
    """
    A custom dataset class for SAP women detection, inheriting from SAPDetectionDataset.

    Args:
        images_directory (str): The directory path containing the images.
        labels_directory (str): The directory path containing the labels.
        processor (optional): The processor to apply to the dataset.

    Methods:
        __init__(images_directory, labels_directory, processor): Initializes the SAPWomenDetectionDataset.
        _create_dataset(): Creates the dataset by loading images and labels, filtering for images with "woman" in the file name.
    """

    def __init__(self, images_directory, labels_directory, processor=None):
        super().__init__(images_directory, labels_directory, processor)

    def _create_dataset(self):
        """
        Creates the dataset by loading images and labels, filtering for images with "woman" in the file name.
        """
        self._dataset_init()
        
        category_id = 0
        for i, image_file in enumerate(self.images_files):
            if "woman" in image_file:
                self._create_coco_object(image_file, i, category_id)
