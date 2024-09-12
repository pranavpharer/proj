# module load devel/python/3.11.7_intel_2021.4.0

import json

import re
from pprint import pprint
import pandas as pd
import torch 
from datasets import Dataset, load_dataset

from peft import LoraConfig, PeftModel,AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    GPTQConfig

)
from trl import SFTTrainer, ConstantLengthDataset
from dsfile import Evaldata,Stepdata



DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "/home/kn/kn_kn/kn_pop542099/Project/Llama-2-70b-hf"#/home/kn/kn_kn/kn_pop542099/Project/
lora_r = 16
lora_alpha = 64
lora_dropout = 0.1
lora_target_modules = [
     "q_proj",
     "up_proj",
     "o_proj",
     "k_proj",
     "down_proj",
     "gate_proj",
     "v_proj",
 ]


peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=lora_target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)
training_arguments = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    optim= "paged_adamw_32bit",#"paged_adamw_8bit",
    logging_steps=10,
    learning_rate=1e-4,
    # fp16=True,
    max_grad_norm=0.2,
    num_train_epochs=10,
    # dataloader_num_workers=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="epoch",
    group_by_length=True,
    output_dir = "/content/outputsdir", #directory to store trained model and result
    report_to="tensorboard",
    save_safetensors=True,
    logging_dir="/content/logs",
    lr_scheduler_type= 'polynomial',#'linear',
    seed=42,
)
def create_model_and_tokenizer(MODEL_NAME,DEVICE):
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizer created")
    #tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.padding_side = "right"
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,token = "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")

    # quantization_config = GPTQConfig(
    # bits=8, 
    # group_size=128, 
    # tokenizer= tokenizer,)
    # model = optimum.quantize(model, quantization_config=quantization_config)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type="nf8",
        llm_int8_enable_fp32_cpu_offload=False,
        bnb_4bit_compute_dtype=torch.float16,#torch.float32,
    ),  # 4 bit quantization for loading the weights in 4 bits
        trust_remote_code=True,
        device_map=DEVICE,
        # sep_token=["[<start of hint>]","[<end of hint>]","[<List of Choices>]","[<List of Choices end>]","[<Choice made from List of Choices>]","[<Choice made from List of Choices end>]"]
    )

    
    #specialtks = ["<start of hint>","<end of hint>","<List of Choices>","<List of Choices end>","<Choice made from List of Choices>","<Choice made from List of Choices end>"]

    #tokenizer.add_special_tokens(special_tokens_dict={"additional_special_tokens":specialtks})
    print("model created")
    return model, tokenizer
model, tokenizer = create_model_and_tokenizer(MODEL_NAME, DEVICE)
tokenizer.pad_token = tokenizer.eos_token
model.config.use_cache = False
stepdata = Stepdata(tokenizer,DEVICE)
evaldata = Evaldata(tokenizer,DEVICE)
print(torch.cuda.memory_summary(device=DEVICE, abbreviated=False))
constant_length_dataset = ConstantLengthDataset(
    tokenizer=tokenizer,
    dataset=stepdata,
    seq_length=seq_length,
    num_of_sequences=num_of_sequences,
    shuffle=shuffle
)
model=model.to(DEVICE)
trainer = SFTTrainer(
    model=model,
    train_dataset=constant_length_dataset,
    eval_dataset=evaldata,
    peft_config=peft_config,
    dataset_text_field="input_ids",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,

)
trainer.train()

