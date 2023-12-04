from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import paths

dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:5000]")

## Wikitext dataset originally:
# {
#       text:
#           [, , , , ] <- each of this is a paragraph or so



dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset.flatten()
a = dataset['train'][0:100]
print(1)

## Wikitext dataset after flatten:
# {
#   train: (4000,1)
#       text:
#           [, , , , ] <- each of this is a paragraph or so
#   test: (1000,1)

tokenizer = AutoTokenizer.from_pretrained(paths.glm_checkpoint, trust_remote_code=True)

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["text"]])

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset["train"].column_names,
)

## tokenized dataset. At this point it is tokenizer-agnostic
# {
#   train: (4000,3)
#       input_ids
#       attention_mask
#       position_ids
#   test: (1000,3)



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

lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)
