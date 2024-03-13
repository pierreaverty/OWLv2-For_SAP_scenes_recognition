import config
import os
import datetime
from PIL import Image
from torch.utils.data import Dataset

class SAPDetectionDataset(Dataset):
    """
    A custom dataset class for SAP detection.

    Args:
        directory (str): The directory path containing the images and labels.
        processor (object, optional): The processor object for image and text encoding. Defaults to None.
        target_image (str, optional): The path to the target image for image query. Defaults to None.

    Attributes:
        images_directory (str): The directory path containing the images.
        labels_directory (str): The directory path containing the labels.
        images_files (list): List of image file names in the images directory.
        labels_files (list): List of label file names in the labels directory.
        target_image_path (str): The path to the target image for image query.
        processor (object): The processor object for image and text encoding.
        dataset (dict): Dictionary representing the dataset.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(int): Returns the item at the given index.
        _dataset_init(): Initializes the dataset dictionary with metadata.
        _create_dataset(): Creates the dataset by loading images and labels.
        _create_coco_object(str, int, int): Creates a COCO object for a given image file and adds it to the dataset.
        get_annotation(int): Returns the annotation at the given index.
        get_image(int): Returns the image at the given index.
        get_category(int): Returns the category at the given index.
        annotations: Property that returns the annotations for the dataset.
        annotation_ids: Property that returns the annotation IDs for the dataset.
        images: Property that returns the images for the dataset.
        image_ids: Property that returns the image IDs for the dataset.
        categories: Property that returns the categories for the dataset.
        category_ids: Property that returns the category IDs for the dataset.
    """

    def __init__(self, directory, processor=None, target_image=None):
        # Initialize the dataset directory paths and other attributes
        self.images_directory = directory + "images/"
        self.labels_directory = directory + "labels/"
        self.images_files = os.listdir(self.images_directory)
        self.labels_files = os.listdir(self.labels_directory)
        
        self.target_image_path = target_image
        
        self.processor = processor
        self._create_dataset()

    def __len__(self):
        """
        Returns the length of the dataset.
        
        Returns:
            int: The length of the dataset.
        """
        return len(self.dataset["images"])

    def __getitem__(self, idx):
        """
        Retrieves the image and target label at the given index.

        Args:
            idx (int): The index of the image and target label to retrieve.

        Returns:
            tuple: A tuple containing the pixel values of the image and the target label.
        """
        # Get the image and image ID from the dataset
        image, image_id = self.dataset["images"][idx]["image"], self.dataset["images"][idx]["id"]
        
        # Get the target label and bounding box from the dataset
        target = self.dataset["categories"][ self.dataset["annotations"][idx]["category_id"]]["name"] if not config.IS_IMAGE_QUERY else None
        bbox = self.dataset["annotations"][idx]["bbox"]
        
        # Create the target object
        target = {
            'image_id': image_id, 
            'annotations': target,
            'bbox': bbox[0:4],
            'objectness': 1,
            'category_id': 0,
        }
        
        if not config.IS_IMAGE_QUERY:
            # Encode the image and target label using the processor object
            encoding = self.processor(images=image, text=target["annotations"], return_tensors="pt")
            
            input_ids = encoding["input_ids"][0]
            attention_mask = encoding["attention_mask"][0]
        else:
            # Encode the image and target image for image query using the processor object
            target_image = self.dataset["annotations"][idx]["target_image"]
            
            encoding = self.processor(images=image, query_images=target_image, return_tensors="pt")
            
            target["query_pixel_values"] = encoding['query_pixel_values']
            input_ids = None
            attention_mask = None

        # Get the pixel values of the image
        pixel_values = encoding["pixel_values"].squeeze()

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
                "name": config.TARGET
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
        # Open the image file
        image = Image.open(os.path.join(self.images_directory, image_file))
        
        if config.IS_IMAGE_QUERY:
            # Open the target image file for image query
            target_image = Image.open(self.target_image_path)
            
        # Open the label file
        label_file = open(os.path.join(self.labels_directory, image_file.split(".")[0] + ".txt"), "r")
        label_content = label_file.read().splitlines()[0].split(" ")
        label = [float(label) for label in label_content]
        
        # Create the image info and annotation objects
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
            "bbox": [label[1], label[2], label[3], label[4]],
            "area": (label[3]) * (label[4]),
            "segmentation": [label[1], label[2], label[3] + image.width, label[4] + image.height],
            "iscrowd": 0
        }
        
        if config.IS_IMAGE_QUERY:
            annotation["target_image"] = target_image
        
        # Add the image info and annotation to the dataset
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
