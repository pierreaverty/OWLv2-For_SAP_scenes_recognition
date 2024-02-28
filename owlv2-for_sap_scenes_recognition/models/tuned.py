from models import *

class OWLv2ForSapRecognition(pl.LightningModule):
    """
    OWLv2 model for SAP scenes recognition.

    Args:
        lr (float): Learning rate for the optimizer.
        lr_backbone (float): Learning rate for the backbone.
        weight_decay (float): Weight decay for the optimizer.

    Attributes:
        lr (float): Learning rate for the optimizer.
        lr_backbone (float): Learning rate for the backbone.
        weight_decay (float): Weight decay for the optimizer.
        model (Owlv2ForObjectDetection): OWLv2 model for object detection.
        processor (Owlv2Processor): OWLv2 processor for object detection.
    """

    def __init__(self, lr, weight_decay):
        super().__init__()
        
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        
        self.lr = lr
        self.weight_decay = weight_decay
        
    def forward(self, pixel_values, input_ids, attention_mask) -> dict:
        """
        Forward pass for the model.

        Args:
            pixel_values: The pixel values of the images.
            target: The input IDs for the images.
            attention_mask: The attention masks for the images.

        Returns:
            dict: A dictionary containing the model outputs.
        """
        return self.model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
      
    def common_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Performs a common step in the training/validation loop.

        Args:
            batch: A tuple containing the pixel values, input IDs, and attention mask.
            batch_idx: The index of the current batch.

        Returns:
            The loss value calculated during the common step.
        """
        pixel_values, input_ids, attention_mask = batch["pixel_values"], batch["input_ids"], batch["attention_mask"]
        
        print(pixel_values.shape, input_ids.shape, attention_mask.shape)
        outputs = self.model(pixel_values, input_ids, attention_mask)

        return outputs.loss
            
    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch: The batch data.
            batch_idx: The batch index.

        Returns:
            torch.Tensor: The loss value.
        """
        loss = self.common_step(batch, batch_idx)
            
        self.log("train_loss", loss)
            
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch: The batch data.
            batch_idx: The batch index.

        Returns:
            torch.Tensor: The loss value.
        """
        loss = self.common_step(batch, batch_idx)
            
        self.log("val_loss", loss)
            
        return loss
    
    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            tuple: A tuple containing the optimizer and learning rate scheduler.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    
    