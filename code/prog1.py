# module load devel/python/3.11.7_intel_2021.4.0
# add check point in training
import json
from memory_profiler import memory_usage
from pprint import pprint
import torch 
from datasets import Dataset
from torch.utils.data import DataLoader
from peft import LoraConfig,get_peft_model#, PeftModel,AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling
)
# from accelerate import DataLoaderConfiguration, accelerate
from trl import  SFTTrainer#SFTConfig,
import pandas as pd 
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def stpData():
    dats = pd.DataFrame(columns=["Scenario","Answer"]) 
    with open("training.json","r") as fi:#
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
    with open("raining.json","r") as fi:#
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
        
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,device_map=torch.device('cpu'),add_special_tokens=True)
        # print("Tokenizer created")
        # model = AutoModelForCausalLM.from_pretrained(
        #     MODEL_NAME,
        #     use_safetensors=True,
        #     quantization_config=BitsAndBytesConfig(
        #     load_in_8bit=True,
        #     bnb_8bit_quant_type="nf8",
        #     llm_int8_enable_fp32_cpu_offload=False,
        #     bnb_4bit_compute_dtype=torch.float32,#torch.float16,#
        # ),  # 4 bit quantization for loading the weights in 4 bits
        #     trust_remote_code=True,
        #     device_map=DEVICE,
        #     # sep_token=["[<start of hint>]","[<end of hint>]","[<List of Choices>]","[<List of Choices end>]","[<Choice made from List of Choices>]","[<Choice made from List of Choices end>]"]
        # )
        
        tokenizer = GPT2Tokenizer.from_pretrained('/home/kn/kn_kn/kn_pop542099/Project/Llama-2-70b-hf',token = "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",device_map='auto')
        model = GPT2LMHeadModel.from_pretrained('/home/kn/kn_kn/kn_pop542099/Project/Llama-2-70b-hf',token = "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",device_map='auto')#,load_in_8bit= True
          
        print("Prog1 model created")
        return model, tokenizer  
    # except Exception as e:
    #     print(f" The error was {e}")

def toekn(tokenizer,data):
    
    # MAX_LENGTH = 512  # Adjust based on your data
    # tokenizer.model_max_length = MAX_LENGTH
    data=data["text"]
    input_tokens = tokenizer.encode(data,return_tensors='pt' ,truncation=True )
    # input_ids = input_tokens['input_ids']
    print(input_tokens)
    # input_tokens['labels'] = input_ids.clone()
    # print("ip id", input_ids)
    # print("Text being tokenized:", data['text'])
    # print("Input IDs:", input_ids)
    
    # special_tokens_ids = tokenizer.convert_tokens_to_ids(["<|begin_of_text|>", "<|end_of_text|>", "<|eom_id|>"])
    # print(f"Special tokens IDs: {special_tokens_ids}")

    # if any(id >= tokenizer.vocab_size for id in special_tokens_ids):
    #     raise ValueError(f"Special tokens have IDs outside the vocab size: {special_tokens_ids}")
    # if input_ids.max().item() >= tokenizer.vocab_size:
    #     raise ValueError(f"Found input ID greater than vocab size: {input_ids.max().item()} >= {tokenizer.vocab_size}")

   
    
    return {'input_ids':input_tokens}  
    



  



if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "Project/Llama-2-70b-hf"#"Meta-Llama-3-70B-Instruct"
    # print("--- Torch Start")
    # print("Device available: ",torch.cuda.is_available())
    # print("# of Devices: ",torch.cuda.device_count())
    # print("Index of current Device: ",torch.cuda.current_device())
    # print("Adress Current Device: ",torch.cuda.device(0))
    # print("Name of Device: ",torch.cuda.get_device_name(0))
    # print("Torch End ---")
    lora_r = 16
    lora_alpha = 64
    lora_dropout = 0.1
    # lora_target_modules = [
    #     "q_proj",
    #     "up_proj",
    #     "o_proj",
    #     "k_proj",
    #     "down_proj",
    #     "gate_proj",
    #     "v_proj",
    # ]

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
        # target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

#     training_arguments = SFTConfig(
    
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=4,
#     learning_rate=1e-4,
#     max_grad_norm=0.2,
#     num_train_epochs=3,
#     evaluation_strategy="steps",
#     eval_steps=50,
#     logging_steps=1000,
#     save_steps=2000,
#     save_strategy="epoch",
#     group_by_length=True,
#     output_dir="outputsdirsft",
#     report_to="tensorboard",
#     save_safetensors=True,
#     do_train=True,
#     logging_dir="logsSft",
#     lr_scheduler_type='linear',
#     seed=42,

# )
    training_arguments=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim= "paged_adamw_8bit",#"paged_adamw_32bit",
        learning_rate=1e-4,
        # fp16=True,
        max_grad_norm=0.2,
        num_train_epochs=10,
        # dataloader_num_workers=10,
        evaluation_strategy="steps",
        eval_steps=50,
        weight_decay=1e-3,
        label_smoothing_factor =0.2,
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
    model.resize_token_embeddings(len(tokenizer))
    # model.to(DEVICE)
    # stepdata = Stepdata(tokenizer)
    stepdata= Dataset.from_dict({
        "text": stpData()
    })
    
    stepdat = stepdata.map(
        lambda x: toekn(tokenizer,x)
    )
    #evaldata = Evaldata(tokenizer,DEVICE)
    evaldata =  Dataset.from_dict({
        "text": evlData()
    })
    # stepdata = DataLoader(stepdata,batch_size=4)
    evaldat = evaldata.map(
        lambda x: toekn(tokenizer,x)
    ) 
    print(evaldata)
    # evaldata = evaldata.remove_columns(["text"])
    print(f"eval now {evaldata}")
    # stepdata=stepdata.remove_columns(["text"])
    # accelerator = accelerate.Accelerator(dataloader_config=dataloader_config)

    trainer = SFTTrainer(
                model=model,
                train_dataset=stepdata,
                peft_config=peft_config,
                eval_dataset = evaldata,
                dataset_text_field="text",
                max_seq_length=256,
                # tokenizer=tokenizer,
                args=training_arguments,
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
                # packing=True
            )
    trainer.train()
    trainer.save_model(training_arguments.output_dir)