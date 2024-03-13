import torch
import config 

def collate_fn(batch) -> dict:
    """
    Collate function for the data loader.

    Args:
        batch (list): A list of tuples containing the batch data.

    Returns:
        dict: A dictionary containing the collated batch data with keys:
            - "pixel_values": The pixel values of the images.
            - "input_ids": The input IDs for the images.
            - "attention_mask": The attention masks for the images.
            - "target": The target values.

    Notes:
        - If `config.IS_IMAGE_QUERY` is `True`, the returned dictionary will also contain:
            - "query_pixel_values": The query pixel values.
    """
    
    # Check if it's not an image query
    if not config.IS_IMAGE_QUERY:
        return {
            "pixel_values": torch.stack([item[0] for item in batch]),  # Stack the pixel values of the images
            "input_ids": torch.stack([item[1] for item in batch]),  # Stack the input IDs for the images
            "attention_mask": torch.stack([item[2] for item in batch]),  # Stack the attention masks for the images
            "target": [item[3] for item in batch],  # Get the target values
        }

    # If it's an image query
    return {
            "pixel_values": torch.stack([item[0] for item in batch]),  # Stack the pixel values of the images
            'target': [item[3] for item in batch],  # Get the target values
            "query_pixel_values": torch.stack([item[3]['query_pixel_values'] for item in batch]),  # Stack the query pixel values
        }
