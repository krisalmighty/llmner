'''
other idea
1. change the neg tokens
2. change different learning rate for contrastive learning
3. instead of normalization on origin_e, neg_e, pos_e, we can use different solutions
'''
import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
import copy
import math

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_mdoel_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from transformer import LlamaForCausalLM, LlamaTokenizer

from transformer import Trainer

from transformer import DataCollatorForSeq2Seq

from datasets import Dataset

from utils.prompter import Prompt

import pandas as pd

import numpy as np