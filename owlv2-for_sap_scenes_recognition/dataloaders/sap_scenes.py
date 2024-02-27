from torch.utils.data import DataLoader

class SAPDetectionDataLoader(DataLoader):
    """
    DataLoader for SAP scene detection dataset.

    Args:
        dataset (Dataset): The dataset to load the data from.
        batch_size (int, optional): The batch size. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
    """

    def __init__(self, dataset, batch_size=32, shuffle=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
        
    def collate_fn(self, batch):
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
            "pixel_values": batch[0],
            "input_ids": batch[1],
            "attention_mask": batch[2],
        }
