# module load devel/python/3.11.7_intel_2021.4.0

import json
from memory_profiler import memory_usage
from pprint import pprint
import torch 
from datasets import Dataset
import pandas as pd
from peft import LoraConfig,get_peft_model#, PeftModel,AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,

)
from trl import SFTTrainer#, ConstantLengthDataset

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def stepData():
    dats = pd.DataFrame(columns=["Scenario","Answer"]) 
    with open("training.json","r") as fi:#
            data = fi.read()
            data = json.loads(data)
            i=1
            while i <= len(data):
                st = "set"+str(i)
                x = data[st]
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
                                scnvalues = "<s>"+ str(snval) + str(hnt) + str(chcs) + str(vals.get('The Choice made'))+"</s>"
                                newrole = pd.DataFrame({
                            "Scenario": [str(scnvalues)] 
                            # "Answers": [str(vals.get('The Choice made'))]
                        })
                                # dats = pd.concat([dats,newrole],ignore_index= True)
                               
                                dats= pd.concat([dats, newrole],ignore_index = True)
                                
                                c=c+1

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
                            # "Answers": [str(vals.get('The Choice made'))]
                        })
                                # dats = pd.concat([dats,newrole],ignore_index= True)
                               
                                dats= pd.concat([dats, newrole],ignore_index = True)
                                # abc.append(scnvalues)
                                    #dats["Answer"] = vals.get('The Choice made')
                                # except Exception as e:
                                #      print(f"Execption {e}")
                                c=c+1

    return dats["Scenario"]    
def create_model_and_tokenizer(MODEL_NAME,DEVICE):
    try:
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,device_map='auto',add_special_tokens=True,
   )

        print("Tokenizer created")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            use_safetensors=True,
            quantization_config=BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
),  # 4 bit quantization for loading the weights in 4 bits
            trust_remote_code=True,
            device_map='auto',
            # sep_token=["[<start of hint>]","[<end of hint>]","[<List of Choices>]","[<List of Choices end>]","[<Choice made from List of Choices>]","[<Choice made from List of Choices end>]"]
        )
    except Exception as e:
        print(f" The error was {e}")
    print("prog4b model created")
    return model, tokenizer  



  



if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "/home/kn/kn_kn/kn_pop542099/Project/Llama-2-70b-hf"#"Meta-Llama-3-70B-Instruct"
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
    #   training_arguments = TrainingArguments(
    #       output_dir="./output",
    #       overwrite_output_dir=True,
    #       per_device_train_batch_size=8,
    #       per_device_eval_batch_size=8,
    #       gradient_accumulation_steps=1,
    #       learning_rate=3e-4,
    #       weight_decay=0.01,
    #       adam_epsilon=1e-8,
    #       max_grad_norm=1.0,
    #       num_train_epochs=3,
    #       warmup_steps=0,
    #       logging_dir="./logs",
    #       logging_steps=1000,
    #       save_steps=2000,
    #       evaluation_strategy="steps",
    #       eval_steps=1000,
    #       save_total_limit=1,
    #       save_strategy="steps",
    #       metric_for_best_model="eval_loss",
    #       greater_is_better=False,
    #       load_best_model_at_end=True,
    #       report_to="tensorboard"
    #   )
    training_arguments = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim= "paged_adamw_8bit",#"paged_adamw_32bit",
        logging_steps=100,
        learning_rate=1e-4,
        # fp16=True,
        max_grad_norm=0.2,
        num_train_epochs=3,
        # dataloader_num_workers=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        group_by_length=True,
        output_dir = "outputsdirsft4", #directory to store trained model and result
        report_to="tensorboard",
        save_safetensors=True,
        logging_dir="logsSft4",
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
    # accelerator = accelerate.Accelerator(dataloader_config=dataloader_config)

    trainer = SFTTrainer(
                model=model,
                train_dataset=stepdata,
                peft_config=peft_config,
                eval_dataset = evaldata,
                dataset_text_field="text",
                max_seq_length=1024,
                # tokenizer=tokenizer,
                args=training_arguments,
                DataCollatorForLanguageModeling(tokenizer, mlm=False)
                # packing=True
            )
    trainer.train()
    trainer.save_model(training_arguments.output_dir)