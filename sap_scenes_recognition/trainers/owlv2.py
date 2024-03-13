import config
from transformers import Trainer
from losses.owlv2 import OWLv2LossFunction


class OWLv2Trainer(Trainer):
    """
    Trainer class for OWLv2 model.

    Inherits from the base Trainer class.

    Args:
        Trainer (class): Base Trainer class.

    Methods:
        compute_loss: Computes the loss for the OWLv2 model.

    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for the OWLv2 model.

        Args:
            model (Model): The OWLv2 model.
            inputs (dict): The input data for the model.
            return_outputs (bool, optional): Whether to return the model outputs along with the loss. Defaults to False.

        Returns:
            Union[Tuple[float, Any], float]: The loss value or a tuple containing the loss value and model outputs.

        """
        labels = inputs["target"]  # Get the target labels
        
        pixel_values = inputs["pixel_values"]  # Get the pixel values
        
        if not config.IS_IMAGE_QUERY:
            input_ids = inputs["input_ids"]  # Get the input IDs
            attention_mask = inputs["attention_mask"]  # Get the attention mask
            
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)  # Forward pass through the model
        else:
            outputs = model.model.image_guided_detection(pixel_values=pixel_values, query_pixel_values=labels[0]['query_pixel_values'])  # Perform image-guided detection
            
        loss = OWLv2LossFunction()(outputs, labels)  # Compute the loss using the OWLv2 loss function

        return (loss, outputs) if return_outputs else loss  # Return the loss value or a tuple containing the loss value and model outputs
