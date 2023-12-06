from transformers import AutoTokenizer, AutoModel
import paths
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, GenerationConfig
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

torch.cuda.empty_cache()

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
# )

# save_name = "yu-nomi/llama-wiki-standards_Lora_D0.0001_Bl1024_Ba6_Ga2"
# save_name = "yu-nomi/llama-wiki-standards_Lora_D0.01_Bl128_Ba16_Ga4"
# save_name = "yu-nomi/llama-wiki-standards_Lora_D0.0001_Bl1024"
save_name = "yu-nomi/llama-wiki-standards_Lora_D0.01_Bl1024_Ba2_Ga2"
model = AutoPeftModelForCausalLM.from_pretrained(
    save_name,
    trust_remote_code=True,
    token=paths.nomi_read_token,
    # low_cpu_mem_usage=True,
    # quantization_config=bnb_config,
    use_cache=False,
)
tokenizer = AutoTokenizer.from_pretrained(save_name)
print("Model parameters: ", str(model.num_parameters()))

model = model.merge_and_unload()

merged_name = save_name+"_merged"
print(merged_name)
model.save_pretrained(merged_name)

model.push_to_hub(merged_name, use_auth_token=paths.nomi_write_token)
tokenizer.push_to_hub(merged_name, use_auth_token=paths.nomi_write_token)


