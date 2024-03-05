"""Module providing models used."""

import torch

import pytorch_lightning as pl

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from torch import Tensor, nn

from torch.utils.checkpoint import checkpoint