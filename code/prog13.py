# module load devel/python/3.11.7_intel_2021.4.0

import json
from memory_profiler import memory_usage
from pprint import pprint
import torch 
from datasets import Dataset
import pandas as pd
from peft import LoraConfig,get_peft_mode#, PeftModel,AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,


)
from trl import SFTTrainer#, ConstantLengthDataset

import os
from accelerate import Accelerator

# ... load model and dataset




def stepData():
    dats = pd.DataFrame(columns=["Scenario","Answer"]) 
    with open("training.json","r") as fi:#
            data = fi.read()
            data = json.loads(data)
            i=1
            while i <= len(data):
                st = "set"+str(i)
                x = data[st]
                
                if i%2 == 1:
                    i+=1
                    for ky, val in x.items():
                        if ky == "Scenario":
                            snval = " Scenario is : " + val
                        if ky == "Steps":
                            c = 1
                            while c<= len(val):
                                    stp = "step"+str(c)
                                    vals = val[stp]
                                    hnt =  "The current hint  is :" +vals.get('The hint')
                                    chcs = 'Choose from one of the following  [CHOICES] : '+vals.get('Choices')
                                    scnvalues ="<|begin_of_text|>"+str(snval) + str(hnt) + str(chcs)+"<|eom_id|>"  +str(vals.get('The Choice made'))+"<|end_of_text|>"
                                    newrole = pd.DataFrame({
                                "Scenario": [str(scnvalues)] 
                            })
                                
                                    dats= pd.concat([dats, newrole],ignore_index = True)
                                    c=c+1
                else:
                     i+=1
    return dats["Scenario"]
def evalData():
    dats = pd.DataFrame(columns=["Scenario","Answer"]) 
    with open("training.json","r") as fi:#
            data = fi.read()
            data = json.loads(data)
            i=1

            while i <= len(data):
                st = "set"+str(i)
                x = data[st]
                
                if i%2 == 0:
                    i+=1
                    for ky, val in x.items():
                        if ky == "Scenario":
                            snval = " Scenario is : " + val
                        if ky == "Steps":
                            c = 1
                            while c<= len(val):
                                    stp = "step"+str(c)
                                    vals = val[stp]
                                    hnt =  "The current hint  is :" +vals.get('The hint')
                                    chcs = 'Choose from one of the following  [CHOICES] : '+vals.get('Choices')
                                    scnvalues ="<|begin_of_text|>"+str(snval) + str(hnt) + str(chcs)+"<|eom_id|>"  +str(vals.get('The Choice made'))+"<|end_of_text|>"
                                    newrole = pd.DataFrame({
                                "Scenario": [str(scnvalues)] 
                            })
                                
                                    dats= pd.concat([dats, newrole],ignore_index = True)
                                    c=c+1
                else:
                     i+=1

    return dats["Scenario"]    
def create_model_and_tokenizer(MODEL_NAME,DEVICE):
    try:
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,device_map="auto")
        print("Tokenizer created")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            use_safetensors=True,
            quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf8",
            llm_int8_enable_fp32_cpu_offload=False,
            # bnb_4bit_compute_dtype=torch.float16,#torch.float16,#
        ),  # 4 bit quantization for loading the weights in 4 bits
            trust_remote_code=True,
            
            # sep_token=["[<start of hint>]","[<end of hint>]","[<List of Choices>]","[<List of Choices end>]","[<Choice made from List of Choices>]","[<Choice made from List of Choices end>]"]
        )
    except Exception as e:
        print(f" The error was {e}")
    print("prog13 model created")
    return model, tokenizer   



if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "/home/kn/kn_kn/kn_pop542099/Project/Llama-2-13b-hf"#"Meta-Llama-3-70B-Instruct"
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
    # accelerator = Accelerator()
#     dataloader_config = DataLoaderConfiguration(
#     dispatch_batches=None, 
#     split_batches=False,     
#     even_batches=True,     
#     use_seedable_sampler=True  
# )

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
        gradient_accumulation_steps=4,
        optim= "paged_adamw_8bit",#"paged_adamw_32bit",
        logging_steps=100,
        learning_rate=1e-4,
        fp16=True,
        max_grad_norm=0.2,
        num_train_epochs=10,
        weight_decay=1e-3,
        label_smoothing_factor =0.2,
        # dataloader_num_workers=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        group_by_length=True,
        output_dir = "outputsdirsft13", #directory to store trained model and result
        report_to="tensorboard",
        save_safetensors=True,
        logging_dir="logsSft13",
        lr_scheduler_type= 'linear', #'polynomial',#
        seed=42,
    )

    model, tokenizer = create_model_and_tokenizer(MODEL_NAME, DEVICE)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|begin_of_text|>","<|end_of_text|>","<|eom_id|>"]})
    model.config.use_cache = False
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(DEVICE)
    # stepdata = Stepdata(tokenizer)
    stepdata= Dataset.from_dict({
        "text": stepData()
    })
    #evaldata = Evaldata(tokenizer,DEVICE)
    evaldata =  Dataset.from_dict({
        "text": evalData()
    })
    # model,optim = accelerator.prepare(model,training_arguments.optim)

    trainer = SFTTrainer(
                model=model,
                train_dataset=stepdata,
                peft_config=peft_config,
                eval_dataset = evaldata,
                dataset_text_field="text",
                max_seq_length=1024,
                tokenizer=tokenizer,
                args=training_arguments,
                packing=True
            )

    trainer.train()
    trainer.save_model(training_arguments.output_dir)
