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

    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        