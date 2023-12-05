from transformers import AutoTokenizer, AutoModel
import paths
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, GenerationConfig
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType

tokenizer = AutoTokenizer.from_pretrained(paths.llama_local_checkpoint, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(paths.llama_local_checkpoint, device_map="auto", load_in_4bit=True)

# tokenizer = AutoTokenizer.from_pretrained(paths.llama_local_checkpoint, trust_remote_code=True, token=paths.annie_read_token)
# model = AutoModelForCausalLM.from_pretrained(paths.llama_local_checkpoint, trust_remote_code=True, token=paths.annie_read_token)
model = model.eval()
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                         inference_mode=False,
                         r=8,
                         lora_alpha=32,
                         lora_dropout=0.1)
model = get_peft_model(model, peft_config)

prompt = "Hello"
model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
output = model.generate(**model_inputs)

print(tokenizer.decode(output[0], skip_special_tokens=True))