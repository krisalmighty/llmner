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
# LlamaForCausalLM--> llama model LlamaTokenizer: tokenizer
from transformer import LlamaForCausalLM, LlamaTokenizer

from transformer import Trainer

from transformer import DataCollatorForSeq2Seq

from datasets import Dataset

# generate prompt
from utils.prompter import Prompter
# Please extract the entity of 
origin = [12148, 6597, 278, 7855, 310]
origin = [str(item) for item in origin]
#'_<<<'
prefix = [3532, 29966]
prefix = [str(item) for item in prefix]
#'_>>>'
suffix = [8653]
suffix = [str(item) for item in suffix]
# '#_Input:'
input = [29937, 10567, 29901]
input = [str(item) for item in input]

def longest_common_sublist(A, B):
    matrix = [[0 for _ in range(len(B)) for _ in range(len(A))]]
    max_len = 0
    start_A = -1
    end_A = -1
    start_B = -1
    end_B = -1
    



# create a custom dataset class that inherit from Dataset
class CustomCTSDataset(Dataset):
    def __init__(self, dataset):
        super().__init__(self, dataset.data)
        print("----------")
        print(self.column_names)

class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(features, return_tensors=None):
        feat = super().__call__(features, return_tensors=None)
        print("feat:")
        print(feat)
        input_ids_batch = feat["input_ids"].tolist()
        origin_arr = []
        pos_arr = []
        neg_arr = []
        for input_ids in input_ids_batch:
            input_str = "_".join([str(item) for item in input_ids])
            origin_str = "_".join(origin)
            prefix_str = "_".join(prefix)
            suffix_str = "_".join(suffix)
            sentence_str = "_".join(input)

            origin_start = input_str.find(origin_str)
            origin_index = input_str[0: origin_start + len(origin_str)].count("_") + 1
            origin_arr.append(origin_index)

            prefix_start = input_str.find(prefix_str)
            prefix_index = input_str[0: prefix_start + len(prefix_str)].count("_") + 1

            suffix_start = input_str.find(suffix_str)
            suffix_index = input_str[0: suffix_start].count("_") - 1

            input_start = input_str.find(sentence_str)
            input_index = input_str[0: input_start + len(sentence_str)].count("_") + 1

            entity_ids = input_ids[prefix_index: suffix_index + 1]
            overlaps_ids = longest_common_sublist(input_ids[input_index: prefix_index], entity_ids)


