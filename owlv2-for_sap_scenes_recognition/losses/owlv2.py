import torch
import torch.nn.functional as F
from torch import nn


class OWLv2LossFunction(nn.Module):
    """
    Custom loss function for OWLv2 scene recognition.
    """

    def __init__(self, weight=None, size_average=True):
        super(OWLv2LossFunction, self).__init__()
        
        self.classification_loss = nn.CrossEntropyLoss(weight=weight, reduction='mean' if size_average else 'sum')
        self.objectness_loss = nn.BCEWithLogitsLoss(reduction='mean' if size_average else 'sum')
        self.box_loss = nn.L1Loss(reduction='mean' if size_average else 'sum')  # Example for L1 loss

    def forward(self, outputs, targets):
        """
        Compute the forward pass of the loss function.
        """
        
        return self.compute_labels_loss(outputs, targets) + 2 * self.compute_objectness_loss(outputs, targets) + 6 * self.compute_box_loss(outputs, targets)

    def compute_labels_loss(self, outputs, targets):
        """
        Compute the forward pass of the loss function.
        """
        # Extract the relevant components from outputs
        logits = outputs['logits'].squeeze(1)
        logits = logits.mean(dim=1)

        # Prepare the targets for loss computation
        class_labels = torch.tensor([t['category_id'] for t in targets], dtype=torch.float, device=logits.device).unsqueeze(1)

        class_loss = self.classification_loss(logits, class_labels)

        return class_loss
    
    def compute_objectness_loss(self, outputs, targets):
        """
        Compute the forward pass of the loss function.
        """
        # Extract the relevant components from outputs
        objectness_logits = outputs['objectness_logits'].squeeze(1)
        objectness_logits = objectness_logits.mean(dim=1)

        # Prepare the targets for loss computation
        objectness_labels = torch.tensor([t['objectness'] for t in targets], dtype=torch.float).to(objectness_logits.device)

        objectness_loss = self.objectness_loss(objectness_logits, objectness_labels)
        
        return objectness_loss
    
    def compute_box_loss(self, outputs, targets):
        """
        Compute the forward pass of the loss function.
        """
        # Extract the relevant components from outputs
        pred_boxes = outputs['pred_boxes'].squeeze(1)
        pred_boxes = pred_boxes.mean(dim=1)
        
        # Prepare the targets for loss computation
        bbox_labels = torch.stack([torch.tensor(t['bbox']) for t in targets]).to(pred_boxes.device)
        box_loss = self.box_loss(pred_boxes, bbox_labels)
        
        return box_loss * 100