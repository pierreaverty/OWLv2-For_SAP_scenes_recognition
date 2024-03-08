from models import *
        
class OWLv2:
    """
    OWLv2 base mode.

    Attributes:
        processor (Owlv2Processor): The processor for text and image inputs.
        model (Owlv2ForObjectDetection): The OWLv2 model for object detection.

    Methods:
        predict(image, texts): Predicts the scene in the given image with the provided texts.

    """

    def __init__(self):
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model =  Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")


    def predict(self, image, texts) -> dict:
        """
        Predicts the scene in the given image with the provided texts.

        Args:
            image: The input image for scene recognition.
            texts: The input texts for scene recognition.

        Returns:
            results: The predicted scene results in Pascal VOC format.

        """
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

        return results

class OWLv2ForSapRecognition(nn.Module, OWLv2):
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

    def __init__(self):
        super().__init__()
        
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble",
                                                             num_labels=1,
                                                             ignore_mismatched_sizes=True,)
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        
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