import json
from memory_profiler import memory_usage
from pprint import pprint
import torch 
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from peft import LoraConfig#, PeftModel,AutoPeftModelForCausalLM
from transformers import (
    # AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    BitsAndBytesConfig,
    T5Config,
    T5ForConditionalGeneration,
    AutoModelForSequenceClassification
    

)
from trl import PPOConfig, PPOTrainer,PreTrainedModelWrapper,AutoModelForCausalLMWithValueHead
from dsfile import Evaldata

def create_model_and_tokenizer(MODEL_NAME,DEVICE):
    
    try: 
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,device_map="auto")
        print("Tokenizer created")
        #tokenizer.pad_token = tokenizer.eos_token
        #tokenizer.padding_side = "right"
        # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,token = "")

        # quantization_config = GPTQConfig(
        # bits=8, 
        # group_size=128, 
        # tokenizer= tokenizer,)
        config = PPOConfig( 
    model_name = MODEL_NAME, 
    batch_size=64,
    mini_batch_size = 32,
    learning_rate=1.41e-5,
    log_with="tensorboard",
    gradient_accumulation_steps=2,
    tracker_project_name="ppoLARGE", 
    project_kwargs={
        "logging_dir": "PPOlog",
        # "logging_steps" : 100,
        # "evaluation_strategy": "epoch"
                    },
)

        config.use_cache = True

    
        
        model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME, 
            use_safetensors=True,
            quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf8",
            llm_int8_enable_fp32_cpu_offload=False,
            bnb_4bit_compute_dtype=torch.float16,#torch.float32,#
            ),
            config=config,
            ignore_mismatched_sizes=True
            )
        #specialtks = ["<start of hint>","<end of hint>","<List of Choices>","<List of Choices end>","<Choice made from List of Choices>","<Choice made from List of Choices end>"]

        #tokenizer.add_special_tokens(special_tokens_dict={"additional_special_tokens":specialtks})
    
        print("model created PPO 13")
    except Exception as e:
        print(f"  Error was {e}")
    return model, tokenizer 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "/home/kn/kn_kn/kn_pop542099/Project/Llama-2-13b-hf" #"Project/Meta-Llama-3-70B"#"Project/Llama-2-70b-hf"
# training_arguments = TrainingArguments(
#     output_dir="outputppo",  # Directory to store checkpoints and logs
#     # per_device_train_batch_size=2,
#     gradient_accumulation_steps=2,
#     learning_rate=1e-5,
#     num_train_epochs=3,  # Adjust based on your needs
#     logging_steps=100,
#     evaluation_strategy="steps",
#     eval_steps=500,
#     save_strategy="epoch",
# )
config = PPOConfig( 
    model_name = MODEL_NAME, 
    batch_size=64,
    mini_batch_size = 32,
    learning_rate=1.41e-5,
    log_with="tensorboard",
    gradient_accumulation_steps=2,
    tracker_project_name="ppoLARGE", 
    project_kwargs={
        "logging_dir": "PPOlog",
        # "logging_steps" : 100,
        # "evaluation_strategy": "epoch"
                    },
)


reward_model = pipeline("text-classification", model="lvwerra/distilbert-imdb")
model, tokenizer = create_model_and_tokenizer(MODEL_NAME, DEVICE)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<|begin_of_text|>","<|eom_id|>"]})#"<|end_of_text|>"

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
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
                    for ky, val in x.items():
                        if ky == "Scenario":
                            snval = " Scenario is : " + val
                        if ky == "Steps":
                            c = 1
                            while c<= len(val):
                                    stp = "step"+str(c)
                                    vals = val[stp]
                                    hnt =  "The current hint  is :" +vals.get('The hint')
                                    chcs = '<|begin_of_text|> Choose from one of the following  [CHOICES] : '+vals.get('Choices')+" <|eom_id|>"
                                    scnvalues = str(snval) + str(hnt) + str(chcs)

                                    newrole = pd.DataFrame({
                            "Scenario": [str(scnvalues)], 
                            "Answers": [str(vals.get('The Choice made'))]
                        })

                                    dats= pd.concat([dats, newrole],ignore_index = True)
                                    c = c+1

        return dats
ds = stpData()

def tokenize(row):
    row["input_ids"] = tokenizer.encode(row["Scenario"], truncation=True, padding='max_length', max_length=512)
    return row

reward_model = pipeline("text-classification", model="roberta-large-mnli")
answerds = pd.DataFrame(ds["Answers"])
answerds=Dataset.from_pandas(answerds)

data_loader = DataLoader(answerds, batch_size=64)
dataset = ds.apply(tokenize, axis=1)
dataset = Dataset.from_pandas(dataset)
epoch = 10
ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer,
)
for epoch in tqdm(range(epoch),"epoch: "):
    c=1
    for batch in tqdm(ppo_trainer.dataloader):
        c= c+1
        print(f"Epoch{c} ")
        query_tensors = batch["input_ids"]
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        #batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        batch["query"] = [tokenizer.decode(q.squeeze(), skip_special_tokens=True) for q in query_tensors]
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts)
        print(pipe_outputs)
        rewards = [torch.tensor(output["score"]) for output in pipe_outputs]
    
        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

#### Save model
ppo_trainer.save_pretrained("my_ppo_model")