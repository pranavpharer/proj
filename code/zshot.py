import json
from memory_profiler import memory_usage
from pprint import pprint
import torch 
from datasets import Dataset
import pandas as pd
from peft import LoraConfig#, PeftModel,AutoPeftModelForCausalLM
from transformers import (
    # AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    BertTokenizer, 
    BertModel,
    AutoModelForCausalLM

)
import evaluate

def stpData():
    dats = pd.DataFrame(columns=["Scenario","Answers"]) 
    abc =[]
    with open("training.json","r") as fi:#
            data = fi.read()
            data = json.loads(data)
            i=1
            while i <= len(data):
                st = "set"+str(i)
                # print(st)
                x = data[st]
                i+=1
                for ky, val in x.items():
                    # print(f"keys - {ky}, val= {val}")
                    if ky == "Scenario":
                        snval = " Scenario is : " + val
                    if ky == "Steps":
                        # print(f"len {len(val)}") 
                        c = 1
                        while c<= len(val):
                                stp = "step"+str(c)
                                vals = val[stp]
                                # print(vals)
                                hnt =  " The current hint  is :" +vals.get('The hint')
                                chcs = ' Choose from one of the following  [CHOICES] : '+vals.get('Choices')
                                chsmd =  ""
                                scnvalues = str(snval) + str(hnt) + str(chcs)
                                    # print(scnvalues)
                                # nwrow =
                                newrole = pd.DataFrame({
                            "Scenario": [str(scnvalues)], 
                            "Answers": [str(vals.get('The Choice made'))]
                        })
                                # dats = pd.concat([dats,newrole],ignore_index= True)
                               
                                dats= pd.concat([dats, newrole],ignore_index = True)
                                abc.append(scnvalues)
                                    #dats["Answer"] = vals.get('The Choice made')
                                # except Exception as e:
                                #      print(f"Execption {e}")
                                c=c+1
    # print(dats["Answers"])
    
    return dats 

def create_model_and_tokenizer(MODEL_NAME):
    try:
        tokenizer = AutoTokenizer.from_pretrained("Project/Llama-2-13b-hf",device_map="auto")
        print("Tokenizer created")
        model = AutoModelForCausalLM.from_pretrained(
           "/home/kn/kn_kn/kn_pop542099/Project/Llama-2-13b-hf",
            use_safetensors=True,
            quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf8",
            llm_int8_enable_fp32_cpu_offload=False,
            bnb_4bit_compute_dtype=torch.float32,#torch.float16,#
        ),  # 4 bit quantization for loading the weights in 4 bits
            trust_remote_code=True,
            device_map="auto",
            # sep_token=["[<start of hint>]","[<end of hint>]","[<List of Choices>]","[<List of Choices end>]","[<Choice made from List of Choices>]","[<Choice made from List of Choices end>]"]
        )
    except Exception as e:
        print(f" The error was {e}")
    print("prog13 model created")
    return model, tokenizer  

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "Project/Llama-2-13b-hf"
    lora_r = 16
    lora_alpha = 64
    lora_dropout = 0.1
    model,tokenizer = create_model_and_tokenizer(MODEL_NAME)
    accuracym = evaluate.load("accuracy")
    berttokn = BertTokenizer.from_pretrained('google-bert/bert-large-uncased')
    bertmodl = BertModel.from_pretrained('google-bert/bert-large-uncased')
    def get_sentence_embedding(sentence):
        inputs = berttokn(sentence, return_tensors='pt', truncation=True, padding=True)
        outputs = bertmodl(**inputs)
        # Take the mean of the token embeddings to get a single sentence embedding
        sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1)
        return sentence_embedding.squeeze().detach().numpy()

    def cosine_similarity(vec1, vec2):
        return cosine(vec1, vec2)
    tokenizerp = AutoTokenizer.from_pretrained("roberta-large-mnli")
    modelp = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    def parphrFunc(sentence1,sentence2):
        tokens = tokenizerp.encode_plus(sentence1, sentence2, return_tensors="pt")
        classification_logits = modelp(**tokens)[0]
        results = torch.softmax(classification_logits, dim=1).tolist()[0]
        return results[2]


    # Set the model to evaluation mode
    # model.eval()


 
    dats = stpData()

    # Zero-shot-like behavior on new da ta
    for sc in (dats.iloc):
        
        #  for _ in  range(len(sc)):
            #   print( f"The len {sc}")
        # question = "What is the capital of Spain?"
        # answer = "Madrid"  # Example answer (optional)

        # formatted_prompt = prompt.format(sc["Scenario"], sc["Answers"])  # Include answer for better guidance
        # print(sc[0])
        # tokenizer = AutoTokenizer.from_pretrained("t5-base")
        input_ids = tokenizer(sc[0], return_tensors="pt")

        # # Generate the response
        # # model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        with torch.no_grad():
            output = model.generate(**input_ids,max_new_tokens=20)

        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        print(f"\nPredicted answer: {decoded_output}")
        print(f"The actual Answer: {sc[1]}")
         # pred = pred.append(output[0])#.tolist()[0]
        q = sc[1]
        r = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        qry = get_sentence_embedding(q)
        rel = get_sentence_embedding(r)
        score =  cosine_similarity(qry,rel)
        paraphs = parphrFunc(q,r)
        if score < 0.2 and paraphs >0.95:
            ref.append(1)
            pred.append(1)
        else:
            ref.append(1)
            pred.append(0) 

    result = accuracym.compute(predictions = pred, references=ref)
    print(f" accuracy score is {result}")

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Seq2SeqTrainer


# # ...

# # Few-shot data preparation
# few_shot_data = [
#     ("What is the capital of France?", "Paris"),
#     ("What is the tallest mountain in the world?", "Mount Everest"),
# ]

# # Tokenization
# tokenizer = AutoTokenizer.from_pretrained("t5-base")
# inputs = tokenizer(few_shot_data, return_tensors="pt", padding="max_length", truncation=True)

# # Fine-tune the LLM
# model_name = "t5-base"  # pre-trained model
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# training_args = TrainingArguments(
#     output_dir="./outputs",
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=8,
#     gradient_accumulation_steps=2,
#     learning_rate=2e-5,
#     num_train_epochs=3,
#     save_steps=1000,
#     eval_steps=500,
# )

# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=inputs,
# )

# trainer.train()

# # Save the fine-tuned model
# trainer.save_model("./fine-tuned_model")

# # Zero-shot-like behavior on new data
# new_question = "What is the largest ocean on Earth?"
# new_input = tokenizer(new_question, return_tensors="pt")

# with torch.no_grad():
#     output = model.generate(**new_input)

# decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
# print(f"Predicted answer: {decoded_output}")
