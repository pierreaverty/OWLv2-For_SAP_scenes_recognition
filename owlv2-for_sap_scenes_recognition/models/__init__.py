"""Module providing models used."""

import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from torch import Tensor, nn

from torch.utils.checkpoint import checkpoint

from transformers import PretrainedConfig
