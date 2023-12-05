## Working example of fine tuning with PEFT

# import loralib as lora
# import lora_utils.insert_lora
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, concatenate_datasets
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

import paths

personal_access_token = "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"
device = 'cuda'
checkpoint = paths.llama_checkpoint

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                         inference_mode=False,
                         r=8,
                         lora_alpha=32,
                         lora_dropout=0.1)

print("model")
# model = AutoModelForCausalLM.from_pretrained(paths.glm_checkpoint, trust_remote_code=True, device='cuda')
model = AutoModelForCausalLM.from_pretrained(paths.llama_checkpoint, trust_remote_code=True, token=paths.annie_read_token)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print("datasets")
standards_dataset = load_dataset(paths.standards_dataset_checkpoint, split="train", token=paths.nomi_read_token)
wiki_dataset = load_dataset(paths.wikipedia_dataset_checkpoint, split="train", token=paths.nomi_read_token)
dataset = concatenate_datasets([standards_dataset, wiki_dataset])

# dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:5000]")
dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset.flatten()
# print(dataset['train'][0])

print("tokenizer")
# tokenizer = AutoTokenizer.from_pretrained(paths.glm_checkpoint, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(paths.llama_checkpoint, trust_remote_code=True, token=paths.annie_read_token)

# special_tokens_dict = {'pad_token': "<pad>"}
# tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.eos_token
num_proc = 200

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["text"]])

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=dataset["train"].column_names,
)

block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc)


# tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="yu-nomi/llama-wiki-standards",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
    hub_token=paths.nomi_write_token,
    per_device_train_batch_size=16,
    # gradient_checkpointing=True,
    # gradient_accumulation_steps=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()
trainer.push_to_hub()