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
        
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble",
                                                             num_labels=1,
                                                             ignore_mismatched_sizes=True,)
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
        
        outputs = self.model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        text_outputs = outputs.text_model_output
        vision_outputs = outputs.vision_model_output
        
        text_embeds = text_outputs[1]
        text_embeds = self.model.owlv2.text_projection(text_embeds)
        image_embeds = vision_outputs[1]
        image_embeds = self.model.owlv2.visual_projection(image_embeds)

        # normalized features
        image_embeds = image_embeds / torch.linalg.norm(image_embeds, ord=2, dim=-1, keepdim=True)
        text_embeds_norm = text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True)

        # cosine similarity as logits and set it on the correct device
        logit_scale = self.model.owlv2.logit_scale.exp().to(image_embeds.device)
        
        logits_per_text = torch.matmul(text_embeds_norm, image_embeds.t()) * logit_scale
        
        return self.owlv2_loss(logits_per_text)
            
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
    
    
    
    # Copied from transformers.models.clip.modeling_clip.contrastive_loss with clip->owlv2
    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


    # Copied from transformers.models.clip.modeling_clip.clip_loss with clip->owlv2
    def owlv2_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0
    