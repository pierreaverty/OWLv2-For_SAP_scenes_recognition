import os
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
        __getitem__(idx): Returns the item at the given index.
        _create_dataset(): Creates the dataset by loading images and labels.

    """

    def __init__(self, images_directory, labels_directory, processor):
        self.images_directory = images_directory
        self.labels_directory = labels_directory
        self.images_files = os.listdir(images_directory)
        self.labels_files = os.listdir(labels_directory)
        self.processor = processor
        self.dataset = self._create_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _create_dataset(self):
        dataset = []
        for i, image_file in enumerate(self.images_files):
            image = Image.open(os.path.join(self.images_directory, image_file))
            label_file = open(os.path.join(self.labels_directory, image_file.split(".")[0] + ".txt"), "r")
            label_content = label_file.read().splitlines()[0].split(" ")
            label = [float(label) for label in label_content]
            dataset.append({
                "image_id": i,
                "image": image,
                "width": image.width,
                "height": image.height,
                "object": {
                    "bbox": [label[1:]],
                    "label": [int(label[0])],
                    "area": [0]
                }
            })
            
        return dataset

# Usage example:
images_directory = "./data/images/"
labels_directory = "./data/labels/"
dataset = SAPDetectionDataset(images_directory, labels_directory)
