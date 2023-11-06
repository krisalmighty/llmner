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
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from transformers import Trainer

from transformers import DataCollatorForSeq2Seq

from datasets import Dataset


from utils.prompter import Prompter

base_model = "decapoda-research/llama-7b-hf"
model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map = "auto",
)
# load the vocabulary
input_ids = torch.tensor([0, 13866,   338,   385, 15278,   393, 16612,   263,  3414, 29892,
          3300,  2859,   411,   385,  1881,   393,  8128,  4340,  3030, 29889,
         14350,   263,  2933,   393,  7128,  2486,  1614,  2167,   278,  2009,
         29889,    13,    13,  2277, 29937,  2799,  4080, 29901,    13, 12148,
          6597,   278,  7855,   310, 13013,   297,   278,  1881, 10541,  2183,
          2400,  1919,   278,  7855,   310, 13013, 14637,   304,   278,  7855,
           393, 11524,   263,  2702, 13013, 29892, 12666, 29892,   470,  2318,
           297,   278,  1881, 10541,   869,    13,    13,  2277, 29937, 10567,
         29901,    13,  1576,   501, 29889, 29903, 29889, 16738,   284, 14409,
           428, 29353,  4083,   777,  8182,   567,  1122,   505,  6133, 17498,
           411, 11664, 11174,   310, 22004, 20562, 29916,   680,   869,    13,
            13,  2277, 29937, 13291, 29901,    13, 29966,   326, 29918,  2962,
         29958,   306,   508,  6597, 16212,   363,   366, 29892,   278, 23892,
         16212,   526,  3532, 29966,   450,   501, 29889, 29903, 29889, 16738,
           284, 14409,   428, 29353,  8653, 29871,   529,   326, 29918,   355,
         29958,     0])
string= 'extractLlll'
tokenizer = LlamaTokenizer.from_pretrained(base_model)
tokenized = tokenizer.encode(string)
print(tokenized) # [0, 3539, 6597,   278,  7855,   310]
print(tokenizer.decode(tokenized)) #  ?? Please extract the entity of
print(tokenizer.decode(input_ids)) # Please extract the entity of
print(tokenizer.eos_token_id)

