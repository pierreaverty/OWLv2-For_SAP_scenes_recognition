import torch

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
        "target": [item[3] for item in batch],
    }