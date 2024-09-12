from datasets import Dataset
import torch 
from peft import LoraConfig,get_peft_model#, PeftModel,AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    T5Config,
    T5ForConditionalGeneration
)
def create_model_and_tokenizer(MODEL_NAME,DEVICE):
    try:
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,device_map=DEVICE,add_special_tokens=True)
        print("Tokenizer created")
        model = AutoModelForCausalLM.from_pretrained(
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
    except Exception as e:
        print(f" The error was {e}")
    print("LLama model created")
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
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model, tokenizer = create_model_and_tokenizer(MODEL_NAME, DEVICE)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print(model)