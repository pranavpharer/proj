
# module load devel/python/3.11.7_intel_2021.4.0
# add check point in training
import json
from memory_profiler import memory_usage
from pprint import pprint
import torch 
from datasets import Dataset

from peft import LoraConfig,get_peft_model#, PeftModel,AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    T5Config,
    T5ForConditionalGeneration,
    GPT2Tokenizer, 
    GPT2Model
)
# from accelerate import DataLoaderConfiguration, accelerate
from trl import SFTTrainer#, ConstantLengthDataset
import pandas as pd 
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def stpData():
    dats = pd.DataFrame(columns=["Scenario","Answer"]) 
    with open("/home/kn/kn_kn/kn_pop542099/training.json","r") as fi:#
            data = fi.read()
            data = json.loads(data)
            i=1
            while i <= len(data):
                st = "set"+str(i)
                x = data[st]
                i+=1
                if i%2 ==1:
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
                                    scnvalues = "<|begin_of_text|>"+str(snval) + str(hnt) + str(chcs)+"<|eom_id|>"  +str(vals.get('The Choice made'))+"<|end_of_text|>"
                                
                                    newrole = pd.DataFrame({
                                "Scenario": [str(scnvalues)] 
                                # "Answers": [str(vals.get('The Choice made'))]
                            })
                                
                                    dats= pd.concat([dats, newrole],ignore_index = True)
                                    c=c+1

    return dats["Scenario"]
def evlData():
    dats = pd.DataFrame(columns=["Scenario","Answer"]) 
    with open("/home/kn/kn_kn/kn_pop542099/training.json","r") as fi:#
            data = fi.read()
            data = json.loads(data)
            i=1
            while i <= len(data):
                st = "set"+str(i)
                x = data[st]
                i+=1
                if i%2 == 0:
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
    # try:
        
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME,device_map=DEVICE,add_special_tokens=True)
        print("Tokenizer created")
        model = GPT2Model.from_pretrained(
            MODEL_NAME,
            use_safetensors=True,
            quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf8",
            llm_int8_enable_fp32_cpu_offload=False,
            bnb_4bit_compute_dtype=torch.float32,#torch.float16,#
        ),  # 4 bit quantization for loading the weights in 4 bits
            trust_remote_code=True,
            device_map=DEVICE,
            # sep_token=["[<start of hint>]","[<end of hint>]","[<List of Choices>]","[<List of Choices end>]","[<Choice made from List of Choices>]","[<Choice made from List of Choices end>]"]
        )
        return model, tokenizer
    # except Exception as e:
    #     print(f" The error was {e}")
    # print("prog1 gpt model created")
     



  



if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "/home/kn/kn_kn/kn_pop542099/Project/gpt2-xl"#"Meta-Llama-3-70B-Instruct"
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

    training_arguments = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim= "paged_adamw_8bit",#"paged_adamw_32bit",
        learning_rate=1e-4,
        # fp16=True,
        max_grad_norm=0.2,
        num_train_epochs=3,
        # dataloader_num_workers=10,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_steps=1000,
        save_steps=2000,
        save_strategy="epoch",
        group_by_length=True,
        output_dir = "outputsdirsft", #directory to store trained model and result
        report_to="tensorboard",
        save_safetensors=True,
        do_train= True,
        logging_dir="logsSft",
        lr_scheduler_type= 'linear', #'polynomial',#
        seed=42,
    )

    model, tokenizer = create_model_and_tokenizer(MODEL_NAME, DEVICE)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|begin_of_text|>","<|end_of_text|>","<|eom_id|>"]})
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(DEVICE)
    # stepdata = Stepdata(tokenizer)
    stepdata= Dataset.from_dict({
        "text": stpData()
    })
    #evaldata = Evaldata(tokenizer,DEVICE)
    evaldata =  Dataset.from_dict({
        "text": evlData()
    })
    # accelerator = accelerate.Accelerator(dataloader_config=dataloader_config)

    trainer = SFTTrainer(
                model=model,
                train_dataset=stepdata,
                peft_config=peft_config,
                eval_dataset = evaldata,
                dataset_text_field="text",
                max_seq_length=256,
                tokenizer=tokenizer,
                args=training_arguments,
                packing=True
            )
    trainer.train()
    trainer.save_model(training_arguments.output_dir)