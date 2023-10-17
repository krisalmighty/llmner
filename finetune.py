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
    for i in range(len(A)):
        for j in range(len(B)):
            if A[i] == B[j]:
                if i == 0 or j == 0:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = matrix[i-1][j-1] + 1
                if matrix[i][j] >= max_len:
                    max_len = matrix[i][j]
                    start_A = i - max_len + 1
                    end_A = i
                    start_B = j - max_len + 1
                    end_B = j
                else:
                    matrix[i][j] = 0
    return [start_A, end_A, start_B, end_B]



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
            overlap_ids = longest_common_sublist(input_ids[input_index: prefix_index], entity_ids)

            pos_start = input_index + overlap_ids[0] - overlap_ids[2]
            pos_end = input_index + overlap_ids[0] - overlap_ids[2] + len(entity_ids)
            pos_arr.append([pos_start, pos_end - 1])
            neg_arr.append([pos_start - 1, pos_start - 2, pos_end, pos_end + 1])

        feat["origin"] = torch.tensor(origin_arr)
        feat["pos"] = torch.tensor(pos_arr)
        feat["neg"] = torch.tensor(neg_arr)

        return feat
class ContrastiveTrainer(Trainer):
    def contrastive_loss(self, hidden_states, origin, pos, neg):
        # hidden_states:  B * L * D (batch_size * context_length_of_each_sample(with padding) * token_embedding_length)
        # origin:  B
        # pos : B * P
        # neg : B * N
        # origin_e: B * 1 * D      pos_e:  B * P * D  neg_e: B * N * D
        # P = 2  N = 4
        # embedding comes from transformer
        # origin_e: the embedding for entity in the instruction with number equals to batch
        # pos_e: the embedding for target entities in input
        # neg_e: the embedding for negative tokens in input
        origin_e = torch,gather(hidden_states, 1, origin.unsqueenze(-1).unsqueenze(-1).expend(-1, -1, hidden_states.shape[-1]))
        pos_e = 


    def compute_loss(self, model, inputs, return_outputs=False):
        origin = copy.deepcopy(inputs["origin"])
        pos = copy.deepcopy(inputs["pos"])
        neg = copy.deepcopy(inputs["neg"])

        if "origin" in  inputs:
            inputs.pop("origin")
        if "pos" in inputs:
            inputs.pop("pos")
        if "neg" in inputs:
            inputs.pop("neg")
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs, output_hidden_states=True)
        cts_loss = self.contrastive_loss(outputs["hidden_states"][26], origin, pos, neg)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference,  the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss += 0.001 * cts_loss
        return (loss, outputs) if return_outputs else loss
    