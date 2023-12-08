from transformers import AutoTokenizer, AutoModel
import paths
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, GenerationConfig
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

torch.cuda.empty_cache()


## Adapter
USE_ADAPTER = True
# save_name = "yu-nomi/llama-wiki-standards_Lora_D0.0001_Bl1024"
# save_name = "yu-nomi/llama-wiki-standards_Lora_D0.01_Bl1024_Ba2_Ga2"
save_name = "yu-nomi/llama-wiki-standards_Lora_D0.01_Bl1024_PAD"

## No Adapter
# USE_ADAPTER = False
# save_name = "yu-nomi/llama-wiki-standards_Lora_D0.01_Bl1024_Ba2_Ga2_merged"
# save_name = paths.llama_local_checkpoint

# prompt = "The NASA standard 5002A is"
prompt = "A method of determining Sound Power Level"
# prompt = "At the sending end, the Synchronization and Channel Coding Sublayer accepts"
# prompt = "When attaching window panels to an aircraft, "
# prompt = "To create a linear rail with minimal backlash, "
# prompt = "Certain cryptographic devices and technical data regarding"
max_new_tokens = 2048



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

if not USE_ADAPTER:
    # Default
    print("Using Untrained")
    model = AutoModelForCausalLM.from_pretrained(save_name,
                                                 quantization_config=bnb_config,
                                                 trust_remote_code=True,
                                                 low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(save_name)
    model = model.eval()

else:
    print("Using Trained")

    model = AutoPeftModelForCausalLM.from_pretrained(
        save_name,
        trust_remote_code=True,
        token=paths.nomi_read_token,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(save_name,
                                              token=paths.nomi_read_token,
                                              trust_remote_code=True)

print("Model parameters: ", str(model.num_parameters()))


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
output_parameters = output_parameters_list["sphinx"]

print("Input: ", prompt)
model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

print("Output: ")
output = model.generate(**model_inputs,
                        do_sample=True,
                        top_p=output_parameters["top_p"],
                        temperature=output_parameters["temperature"],
                        top_k=output_parameters["top_k"],
                        # max_new_tokens=max_new_tokens,
                        num_beams=3,
                        no_repeat_ngram_size=5
                        )


# for i, sample_output in enumerate(output):
#   print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

for i in range(len(output)):
    print(tokenizer.decode(output[i], skip_special_tokens=True))