import torch

from torch.utils.data import DataLoader

class SAPDetectionDataLoader(DataLoader):
    """
    DataLoader for SAP scene detection dataset.

    Args:
        dataset (Dataset): The dataset to load the data from.
        batch_size (int, optional): The batch size. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=23):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, num_workers=num_workers)
        
    def collate_fn(self, batch) -> dict:
        """
        Collate function for the data loader.

        Args:
            batch (list): A list of tuples containing the batch data.

        Returns:
            dict: A dictionary containing the collated batch data with keys:
                - "pixel_values": The pixel values of the images.
                - "input_ids": The input IDs for the images.
                - "attention_mask": The attention masks for the images.
        """
        
        return {
            "pixel_values": torch.stack([item[0] for item in batch]),
            "input_ids": torch.stack([item[1] for item in batch]),
            "attention_mask": torch.stack([item[2] for item in batch]),
        }
