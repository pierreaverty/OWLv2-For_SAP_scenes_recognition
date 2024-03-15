from models import *

import config

import torch
import torch.nn as nn

class OWLv2(nn.Module):
    """
    OWLv2 base model for scene recognition.

    Attributes:
        processor (Owlv2Processor): The processor for text and image inputs.
        model (Owlv2ForObjectDetection): The OWLv2 model for object detection.

    Methods:
        forward(pixel_values, query, attention_mask): Forward pass for the model.
        predict(image, texts): Predicts the scene in the given image with the provided texts.
    """

    def __init__(self):
        super().__init__()
        
        # Initialize the processor and model
        self.processor = Owlv2Processor.from_pretrained(config.MODEL_PATH)
        self.model =  Owlv2ForObjectDetection.from_pretrained(config.MODEL_PATH)

    def forward(self, pixel_values, query, attention_mask) -> dict:
        """
        Forward pass for the model.

        Args:
            pixel_values: The pixel values of the images.
            query: The input IDs for the images.
            attention_mask: The attention masks for the images.

        Returns:
            dict: A dictionary containing the model outputs.
        """
        if not config.IS_IMAGE_QUERYS:
            # If not using image queries, pass the pixel values, input IDs, and attention mask to the model
            return self.model(pixel_values=pixel_values, input_ids=query, attention_mask=attention_mask)
        
        # If using image queries, pass the pixel values and query pixel values to the model
        return self.model.image_guided_detection(pixel_values=pixel_values, query_pixel_values=query)

    def predict(self, image, texts) -> dict:
        """
        Predicts the scene in the given image with the provided texts.

        Args:
            image: The input image for scene recognition.
            texts: The input texts for scene recognition.

        Returns:
            dict: The predicted scene results in Pascal VOC format.
        """
        # Preprocess the inputs using the processor
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        
        # Pass the preprocessed inputs to the model
        outputs = self.model(**inputs)
        
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        
        # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.013)

        return results
    