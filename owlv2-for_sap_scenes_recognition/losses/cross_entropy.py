from losses import *

class CrossEntropyLossForOWLv2(nn.Module):
    """
    Custom loss function for OWLv2 scene recognition.

    Args:
        weight (Tensor, optional): A manual rescaling weight given to each class. Default: None
        size_average (bool, optional): By default, the losses are averaged over each loss element in the batch. Default: True

    Attributes:
        loss (CrossEntropyLoss): The underlying PyTorch CrossEntropyLoss object.

    """

    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLossForOWLv2, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight, size_average)

    def forward(self, outputs, targets):
        """
        Compute the forward pass of the loss function.

        Args:
            outputs (Tensor): The predicted outputs from the model.
            targets (Tensor): The ground truth labels.

        Returns:
            Tensor: The computed loss value.

        """
        return self.loss(outputs, targets)