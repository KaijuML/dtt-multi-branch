""" Main entry point of the ONMT library """
from __future__ import division, print_function

import onmt.inputters
import onmt.encoders
import onmt.decoders
import onmt.models
import onmt.utils
import onmt.modules
import onmt.metrics
from onmt.trainer import Trainer
from onmt.rl_trainer import RLTrainer
import sys
import onmt.utils.optimizers
onmt.utils.optimizers.Optim = onmt.utils.optimizers.Optimizer
sys.modules["onmt.Optim"] = onmt.utils.optimizers

# For Flake
__all__ = [onmt.inputters, onmt.encoders, onmt.decoders, onmt.models,
           onmt.utils, onmt.modules, onmt.metrics, "Trainer", "RLTrainer"]

__version__ = "1.0.2"
