from torch.utils.data import DataLoader
import pandas as pd
from datasets import Dataset
import json
import torch
from tqdm import tqdm
from memory_profiler import memory_usage
from pprint import pprint
import torch.nn.functional as F
import torch.optim as optim
from peft import LoraConfig
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoTokenizer,
    pipeline,
    GPT2Tokenizer, 
    TFGPT2Model
    BertTokenizer,
    BertModel,
    AutoModelForSequenceClassification,
)
from trl import PPOConfig, PPOTrainer,AutoModelForCausalLMWithValueHead
from scipy.spatial.distance import cosine

# Function to prepare data
def stpData():
    dats = pd.DataFrame(columns=["Scenario", "Answers"])
    abc = []
    with open("training.json", "r") as fi:
        data = json.loads(fi.read())
        i = 1
        while i <= len(data):
            st = "set" + str(i)
            x = data[st]
            i += 1
            for ky, val in x.items():
                if ky == "Scenario":
                    snval = " Scenario is : " + val
                if ky == "Steps":
                    c = 1
                    while c <= len(val):
                        stp = "step" + str(c)
                        vals = val[stp]
                        hnt = "For " + stp + " The current hint is: " + vals.get('The hint')
                        chcs = ' Choose from one of the following [CHOICES]: ' + vals.get('Choices')
                        scnvalues ="<|begin_of_text|>"+str(snval) + str(hnt) + str(chcs)+"<|eom_id|>" 
                        newrole = pd.DataFrame({"Scenario": [str(scnvalues)], "Answers": [str(vals.get('The Choice made')+"<|end_of_text|>")]})
                        dats = pd.concat([dats, newrole], ignore_index=True)
                        abc.append(scnvalues)
                        c += 1
    return dats

# Function to create model and tokenizer

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
            model_name=MODEL_NAME,
            learning_rate=1.41e-5
            )

        config.use_cache = True

    
        
        model = TFGPT2Model.from_pretrained(MODEL_NAME, 
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
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        return model, tokenizer 
        #specialtks = ["<start of hint>","<end of hint>","<List of Choices>","<List of Choices end>","<Choice made from List of Choices>","<Choice made from List of Choices end>"]

        #tokenizer.add_special_tokens(special_tokens_dict={"additional_special_tokens":specialtks})
    
        print("model created PPO with sim")
    except Exception as e:
        print(f"  Error was { e}")
    
# Function to get sentence embedding
def get_sentence_embedding(sentence):
    inputs = berttokn(sentence, return_tensors='pt', truncation=True, padding=True)
    outputs = bertmodl(**inputs)
    sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1)
    return sentence_embedding.squeeze().detach().numpy()

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    return cosine(vec1, vec2)

# Function to check paraphrasing
def parphrFunc(sentence1, sentence2):
    tokens = tokenizerp.encode_plus(sentence1, sentence2, return_tensors="pt")
    classification_logits = modelp(**tokens)[0]
    results = torch.softmax(classification_logits, dim=1).tolist()[0]
    return results[2]

# Define necessary components
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "/home/kn/kn_kn/kn_pop542099/Project/gpt2-xl"#"Project/Llama-2-70b-hf"
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
berttokn = BertTokenizer.from_pretrained('bert-large-uncased')
bertmodl = BertModel.from_pretrained('bert-large-uncased')
tokenizerp = AutoTokenizer.from_pretrained("roberta-large-mnli")
modelp = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

# Load model and tokenizer
model, tokenizer = create_model_and_tokenizer(MODEL_NAME, DEVICE)
model = get_peft_model(model, peft_config)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<|begin_of_text|>","<|end_of_text|>","<|eom_id|>"]})
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
tokenizer.pad_token = tokenizer.eos_token
ds = stpData()
dsa = pd.DataFrame(ds["Answers"])
dsq = pd.DataFrame(ds["Scenario"])
model.print_trainable_parameters()
model.to(DEVICE)

# Tokenization
def tokenize(row):
    row["input_ids"] = tokenizer.encode(row["Scenario"], padding='max_length', truncation=True, max_length=64)
    return row

def tokenizeans(row):
    row["input_ids"] = tokenizer.encode(row["Answers"], padding='max_length', truncation=True, max_length=90)
    return row

ds = ds.apply(tokenize, axis=1)
answerds = Dataset.from_pandas(dsa)
dataset = Dataset.from_pandas(ds)
model=model.to(DEVICE)
# PPO Configuration
ppo_config = PPOConfig(
    # model_name=MODEL_NAME,
    batch_size=64,
    mini_batch_size=32,
    learning_rate=1.41e-5,
    log_with="tensorboard",
    gradient_accumulation_steps=2,
    tracker_project_name="ppoLARGE",
    project_kwargs={"logging_dir": "PPOlog"},
)

# PPO Trainer
ppo_trainer = PPOTrainer(
    model=model,
    config=ppo_config,
    dataset=dataset,
    tokenizer=tokenizer,
    optimizer=optimizer,
)

# Define generation arguments
generation_kwargs = {
    "max_length": 256,
    "top_k": 5,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

# Define step method
# def step(self, query_tensors, response_tensors, rewards):
#     # Compute logprobs, values, and rewards
#     logprobs, values, rewards = self.model.train_step(query_tensors, response_tensors, rewards)
#     # Compute advantages
#     advantages = rewards - values
#     # Compute policy loss
#     policy_loss = -(advantages * logprobs).mean()
#     # Compute value loss
#     value_loss = F.mse_loss(values, rewards)
#     # Compute total loss
#     total_loss = policy_loss + value_loss
#     # Backpropagate
#     total_loss.backward()
#     # Update parameters
#     self.optimizer.step()
#     self.optimizer.zero_grad()
#     # Log statistics
#     stats = {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "total_loss": total_loss.item()}
#     return stats

# Define compute_rewards method
# def compute_rewards(self, scores, all_logprobs, ref_logprobs, masks):
#     # Calculate KL divergence between the new and old policy
#     kls = F.kl_div(all_logprobs, ref_logprobs, reduction='batchmean')
#     # Calculate rewards
#     rewards = scores - 0.01 * kls
#     non_score_reward = rewards.mean()
#     return rewards, non_score_reward, kls

# # Define log_stats method
# def log_stats(self, stats, batch, rewards):
#     for key, value in stats.items():
#         print(f"{key}: {value}")

# # Attach methods to PPOTrainer instance
# # ppo_trainer.step = step.__get__(ppo_trainer, PPOTrainer)
# ppo_trainer.compute_rewards = compute_rewards.__get__(ppo_trainer, PPOTrainer)
# ppo_trainer.log_stats = log_stats.__get__(ppo_trainer, PPOTrainer)

# Data loader
data_loader1 = DataLoader(answerds, batch_size=64)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
# Training loop
for epoch in tqdm(range(3), "epoch:"):
    c = 1
    for batch, custom_batch_data in zip(ppo_trainer.dataloader, data_loader1):
        print(f" C is {c}")
        query_tensors = batch["input_ids"]
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)

        grdtensors = custom_batch_data["Answers"]
        print(len(grdtensors))
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        batch["query"] = [tokenizer.decode(q.squeeze(), skip_special_tokens=True) for q in query_tensors] 
        rewards = []
        for q, r in zip(batch["response"], custom_batch_data["Answers"]):
            qry = get_sentence_embedding(q)
            rel = get_sentence_embedding(r)
            score = cosine_similarity(qry, rel)
            paraphs = parphrFunc(q, r)
            if score < 0.2 and paraphs > 0.95:
                rewards.append(torch.tensor(float(1) - score))
            else:
                rewards.append(torch.tensor(-1.0))
        print(len(rewards))
        print(rewards)
        
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        c += 1

ppo_trainer.save_pretrained("my_ppo_model")
