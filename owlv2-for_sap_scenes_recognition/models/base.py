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


    def predict(self, image, texts):
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