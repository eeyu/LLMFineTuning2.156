import sentencepiece as spm
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import paths

checkpoint = paths.glm_checkpoint
# model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, device='cuda')


# https://stackoverflow.com/questions/72625528/translation-between-different-tokenizers

sp = spm.SentencePieceProcessor(model_file=checkpoint)
tokens = sp.encode('This is a test')
print(tokens)