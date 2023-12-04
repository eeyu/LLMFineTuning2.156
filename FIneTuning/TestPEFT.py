## Demonstrates converting a refular model to a PEFT model


from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
model_name_or_path = "THUDM/chatglm-6b"
tokenizer_name_or_path = "THUDM/chatglm-6b"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

# model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device='cuda')
model = model.eval()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()