import torch
import config 
from torch import nn

import torch.nn.functional as F



class OWLv2LossFunction(nn.Module):
    """
    Custom loss function for OWLv2 scene recognition.
    """

    def __init__(self, weight=None, size_average=True):
        super(OWLv2LossFunction, self).__init__()
        
        # Define the classification loss using CrossEntropyLoss
        self.classification_loss = nn.CrossEntropyLoss(weight=weight, reduction='mean' if size_average else 'sum')
        
        # Define the objectness loss using BCEWithLogitsLoss
        self.objectness_loss = nn.BCEWithLogitsLoss(reduction='mean' if size_average else 'sum')
        
        # Define the box loss using L1Loss
        self.box_loss = nn.L1Loss(reduction='mean' if size_average else 'sum')

    def forward(self, outputs, targets):
        """
        Compute the forward pass of the loss function.

        Args:
            outputs (dict): The output dictionary from the model.
            targets (list): The list of target dictionaries.

        Returns:
            torch.Tensor: The computed loss value.
        """
        if config.IS_IMAGE_QUERY:
            # If it's an image query, only compute the box loss
            return self.compute_box_loss(outputs, targets)
        
        # Compute the labels loss, objectness loss, and box loss
        return self.compute_labels_loss(outputs, targets) + 0 * self.compute_objectness_loss(outputs, targets) + self.compute_box_loss(outputs, targets)

    def compute_labels_loss(self, outputs, targets):
        """
        Compute the labels loss.

        Args:
            outputs (dict): The output dictionary from the model.
            targets (list): The list of target dictionaries.

        Returns:
            torch.Tensor: The computed classification loss value.
        """
        # Extract the relevant components from outputs
        logits = outputs['logits'].squeeze(1)
        logits = logits.mean(dim=1)

        # Prepare the targets for loss computation
        class_labels = torch.tensor([t['category_id'] for t in targets], dtype=torch.float, device=logits.device).unsqueeze(1)

        # Compute the classification loss
        class_loss = self.classification_loss(logits, class_labels)

        return class_loss
    
    def compute_objectness_loss(self, outputs, targets):
        """
        Compute the objectness loss.

        Args:
            outputs (dict): The output dictionary from the model.
            targets (list): The list of target dictionaries.

        Returns:
            torch.Tensor: The computed objectness loss value.
        """
        # Extract the relevant components from outputs
        objectness_logits = outputs['objectness_logits'].squeeze(1)
        objectness_logits = objectness_logits.mean(dim=1)

        # Prepare the targets for loss computation
        objectness_labels = torch.tensor([t['objectness'] for t in targets], dtype=torch.float).to(objectness_logits.device)

        # Compute the objectness loss
        objectness_loss = self.objectness_loss(objectness_logits, objectness_labels)
        
        return objectness_loss
    
    def compute_box_loss(self, outputs, targets):
        """
        Compute the box loss.

        Args:
            outputs (dict): The output dictionary from the model.
            targets (list): The list of target dictionaries.

        Returns:
            torch.Tensor: The computed box loss value.
        """
        # Extract the relevant components from outputs
        if not config.IS_IMAGE_QUERY:
            pred_boxes = outputs['pred_boxes'].squeeze(1)
        else:
            pred_boxes = outputs['target_pred_boxes'].squeeze(1)

        pred_boxes = pred_boxes.mean(dim=1)
        
        # Prepare the targets for loss computation
        bbox_labels = torch.stack([torch.tensor(t['bbox']) for t in targets]).to(pred_boxes.device)
        
        # Compute the box loss
        box_loss = self.box_loss(pred_boxes, bbox_labels)
        
        return box_loss * 100