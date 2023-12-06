from transformers import AutoTokenizer, AutoModel
import paths
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, GenerationConfig
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

torch.cuda.empty_cache()


USE_TRAINED = True
# prompt = "The NASA standard 5002A is"
# prompt = "A method of determining Sound Power Level"
prompt = "At the sending end, the Synchronization and Channel Coding Sublayer accepts"
max_new_tokens = 512



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

if not USE_TRAINED:
    # Default
    print("Using Untrained")
    # tokenizer = AutoTokenizer.from_pretrained(paths.llama_local_checkpoint, trust_remote_code=True, token=paths.annie_read_token)
    # model = AutoModelForCausalLM.from_pretrained(paths.llama_local_checkpoint, trust_remote_code=True, token=paths.annie_read_token)
    model = AutoModelForCausalLM.from_pretrained(paths.llama_local_checkpoint,
                                                 quantization_config=bnb_config,
                                                 low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(paths.llama_local_checkpoint)
    model = model.eval()
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             inference_mode=False,
                             r=8,
                             lora_alpha=32,
                             lora_dropout=0.1)
    model = get_peft_model(model, peft_config)

else:
    print("Using Trained")
    # save_name = "yu-nomi/llama-wiki-standards_Lora_D0.0001_Bl1024_Ba6_Ga2"
    save_name = "yu-nomi/llama-wiki-standards_Lora_D0.01_Bl128_Ba16_Ga4"
    model = AutoPeftModelForCausalLM.from_pretrained(
        save_name,
        trust_remote_code=True,
        token=paths.nomi_read_token,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        use_cache=False,
        # _configuration_file="adapter_config.json"
    )
    tokenizer = AutoTokenizer.from_pretrained(save_name)

output_parameters_list = {
    "precise": {
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.1
    },
    "creative": {
        "temperature": 0.72,
        "top_k": 0,
        "top_p": 0.73
    },
    "sphinx": {
        "temperature": 1.99,
        "top_k": 30,
        "top_p": 0.18
    }
}
output_parameters = output_parameters_list["precise"]


print("Input: ", prompt)
model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

print("Output: ")
output = model.generate(**model_inputs, do_sample=True,
                        top_p=output_parameters["top_p"],
                        temperature=output_parameters["temperature"],
                        top_k=output_parameters["top_k"],
                        max_new_tokens=max_new_tokens)

for i in range(len(output)):
    print(tokenizer.decode(output[i], skip_special_tokens=True))