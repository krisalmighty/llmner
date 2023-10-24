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
# 
origin = [12148,  6597,   278,  7855,   310]
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


def find_last_index(A, B):
  B = B[:]
  for i, x in reversed(list(enumerate(A))):
    if x == B[-1]:
      B.pop()
      if not B:
        return i
  return -1
 


def find_first_index(A, B):
  B = B[:]
  for i, x in enumerate(A):
    if x == B[0]:
      B.pop(0)
      if not B:
        return i
  return -1

 
 
def longest_common_sublist(A, B):
  matrix = [[0 for _ in range(len(B))] for _ in range(len(A))]
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

# Create a custom dataset class that inherits from Dataset, actually did nothing
class CustomCTSDataset(Dataset):
    def __init__(self, dataset):
        super().__init__(dataset.data)
        print("-------")
        print(self.column_names)

class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        # call the __call__ function of father class
        feat = super().__call__(features, return_tensors=None) # until this line, we do nothing
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

            origin_index = input_str[0:origin_start + len(origin_str)].count("_") + 1
            
            origin_arr.append(origin_index)

            prefix_start = input_str.find(prefix_str)
            prefix_index = input_str[0:prefix_start + len(prefix_str)].count("_") + 1
            print("The prefix_index is : ", prefix_index)
            input_start = input_str.find(sentence_str)
            input_index = input_str[0:input_start + len(sentence_str)].count("_") + 1


            suffix_start = input_str.find(suffix_str)
            suffix_index = input_str[0:suffix_start].count("_") - 1
            print("The suffix_index is : ", suffix_index)
            entity_ids = input_ids[prefix_index: suffix_index + 1]
            overlap_ids = longest_common_sublist(input_ids[input_index:prefix_index], entity_ids)
            
            sentence_list = input_ids[input_index:prefix_index]
            
            pos_start = input_index + overlap_ids[0] - overlap_ids[2]
            pos_end = input_index + overlap_ids[0] - overlap_ids[2] + len(entity_ids)
            pos_arr.append([pos_start, pos_end - 1])
            neg_arr.append([pos_start - 1, pos_start - 2, pos_end, pos_end + 1])
        feat["origin"] = torch.tensor(origin_arr)
        feat["pos"] = torch.tensor(pos_arr)
        feat["neg"] = torch.tensor(neg_arr)

        # the feat is the 'input' for data collector
        return feat

class ContrastiveTrainer(Trainer): # inherent from trainer function since we need new loss function of contrastive loss
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
        origin_e = torch.gather(hidden_states, 1, origin.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,hidden_states.shape[-1]))
        pos_e = torch.gather(hidden_states, 1, pos.unsqueeze(-1).expand(-1,-1,hidden_states.shape[-1]))
        neg_e = torch.gather(hidden_states, 1, neg.unsqueeze(-1).expand(-1,-1,hidden_states.shape[-1]))
        # L2 normalization
        origin_e = origin_e / origin_e.norm(dim=2, keepdim=True)
        pos_e = pos_e / pos_e.norm(dim=2, keepdim=True)
        neg_e = neg_e / neg_e.norm(dim=2, keepdim=True)

        # the value of pos_score and neg_score are very large which is much better than 1 due to large dimention
        pos_score = (torch.mul(origin_e, pos_e).sum(dim=2)).sum(dim=1)
        neg_score = (torch.mul(origin_e, neg_e).sum(dim=2)).sum(dim=1)
        # if the value in sigmoid is large than 10, then there is gradient vanishing problem
        # three solution: 1.  we do not use sigmoid but this will generate large gradient not good for fine-tune
        # 2. divided by sqrt(d)
        # 3. normalize the embedding,(we use this in this code), can try other ways
        cl_loss = -torch.log(1e-10 + torch.sigmoid(pos_score - neg_score)).mean()

        print("cl_loss:", cl_loss)
        return cl_loss


    def compute_loss(self, model, inputs, return_outputs=False):
        # customized loss function, we need three more features, origin + pos = postive pair, origin + neg = negative pair
        origin = copy.deepcopy(inputs["origin"])
        pos = copy.deepcopy(inputs["pos"])
        neg = copy.deepcopy(inputs["neg"])
        # llama model can only accept 
        if "origin" in inputs:
            inputs.pop("origin")
        if "pos" in inputs:
            inputs.pop("pos")
        if "neg" in inputs:
            inputs.pop("neg")
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # LlamaforCausal input can only have id and mask, 
        # the outputs is output_hidden_states 
        outputs = model(**inputs, output_hidden_states=True)
        # 7b model has 32 layers, apply contrastive learning to 26th layer
        # other idea: add other layer's loss
        cts_loss = self.contrastive_loss(outputs["hidden_states"][26], origin, pos, neg)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # cross_entropy
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
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # lower the weight of cts_loss, since the absolute value of cts_loss is much larger than cross_entropy
        # ensentially multi-task learning, actually it is a weighted sum of gradient
        loss += 0.001 * cts_loss
#        print(0.0015)
        return (loss, outputs) if return_outputs else loss



def train(
    # model/data params
    base_model: str = "decapoda-research/llama-7b-hf",  # the only required argument
    data_path: str = "train_v3.json", # we use train_v3.json dataset
    output_dir: str = "./lora-alpaca_origin",
    # training hyperparams
    batch_size: int = 1,
    micro_batch_size: int = 1,
    num_epochs: int = 5,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 10,
    # lora hyperparams
    lora_r: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    # LOCAL_RANK: the number of GPU, if number = 3, local_number = 0, it means the first GPU of the third process
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    # combine the input, output and instruction
    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    # the number of global process(the number of processes used in distributed training.)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    # some environment set
    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    # for float32 with 7B model, need 7x4 = 28GB memory, if load in 8bit, we only need 7B
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    # load the vocabulary
    tokenizer = LlamaTokenizer.from_pretrained(base_model)



    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    # data preparation convert the prompt to id
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        # 
        # eos_token (str or tokenizers.AddedToken, optional) — A special token representing the end of a sentence. Will be associated to self.eos_token and self.eos_token_id.
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            # the input of transformer consists of input_ids and attention mask(decoder model)
            result["input_ids"].append(tokenizer.eos_token_id)
            # the purpose of attention mask is to mask the padding tokens, because the input token has some padding,
            # we hope to mask the padding
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result
    # data preparation: combine the instruction, input, output to a sentence
    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )

        tokenized_full_prompt = tokenize(full_prompt)
        # whether or not we need to train the instruction and input
        # another choice is to train only on output(start from <im_start>)
        # llama2 only support train on output, but that's when we have enough samples, here we train on input for llama2
        '''
        example for generated prompt
        Below is an instruction that describes a task, paired with an input that provides further context. 
        Write a response that appropriately completes the request.\n\n### Instruction:\nPlease extract the entity 
        of organization in the input sentence given below , the entity of organization refers to the entity that 
        represents a specific organization, institution, or group in the input sentence .\n\n### Input:\nWikinews 
        interviewed one of the owners of a New York City bar about a popular new politically-themed cocktail drink 
        called Santorum .\n\n### Response:\n<im_start> I can extract entities for you, the extracted entities are <<< Wikinews >>>  <im_end>'
        '''
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        tokenized_full_prompt["pos"] = [1111,1111]
        tokenized_full_prompt["neg"] = [0000,0000]
            
        return tokenized_full_prompt

    #model = prepare_model_for_int8_training(model)
    '''
    class TaskType(str, enum.Enum):
        SEQ_CLS = "SEQ_CLS"
        SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
        CAUSAL_LM = "CAUSAL_LM"
        TOKEN_CLS = "TOKEN_CLS"
        QUESTION_ANS = "QUESTION_ANS"
        FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
    '''
    config = LoraConfig(
        r=lora_r,# reduce the r dimention, r = 16, 32
        lora_alpha=lora_alpha, # the weight matrix is scaled by lora_alpha/r,alpha value assign more weight to the LoRA activations
        # the target module of lora,target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", 
        # "fc_in","fc_out", "wte"]
        # out_proj: output classification layer
        # fc_in,fc_out: ffn
        # wte: word_embedding
        target_modules=lora_target_modules, 
        lora_dropout=lora_dropout,# lora_dropout, the dropout probability of the LoRA layers
        bias="none",# Specifies if the bias parameters should be trained. Can be 'none', 'all' or 'lora_only'.
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    # if we have json data in the current directory, we can use it directly, if not we search on the internet
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    # seldom used
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    print("before:")

    if val_set_size > 0:
        '''
        train_val
        DatasetDict({
    train: Dataset({
        features: ['instruction', 'input', 'output'],
        num_rows: 45
    })
    test: Dataset({
        features: ['instruction', 'input', 'output'],
        num_rows: 10
    })
})
        '''
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        
#        print(train_val["train"][0])
#        print(train_val["train"][1])
        # the data is presented in a different order during each epoch when training machine 
        # learning models, which can help the model generalize better.
        '''
        train_data
        Dataset({
    features: ['instruction', 'input', 'output', 'input_ids', 'attention_mask', 'labels', 'pos', 'neg'],
    num_rows: 45
})
        '''
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
        a = 1
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

#    print("after")
#    print(train_data[0])
#    print(train_data[1])
    
    train_data = CustomCTSDataset(train_data) # haven't finished, add customized word to data
    print("train_data")
    print(train_data[0])
    print(train_data[1])
    
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = ContrastiveTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            # The evaluation strategy to adopt during training.
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=CustomDataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

#    old_state_dict = model.state_dict
#    model.state_dict = (
#        lambda self, *_, **__: get_peft_model_state_dict(
#            self, old_state_dict()
#        )
#    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    
    fire.Fire(train)

    