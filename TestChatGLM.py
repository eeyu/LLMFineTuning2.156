from transformers import AutoTokenizer, AutoModel
import paths
checkpoint = "yu-nomi/test-glm-1"
checkpoint_origin = "THUDM/chatglm3-6b"

tokenizer = AutoTokenizer.from_pretrained(checkpoint_origin, trust_remote_code=True, token=paths.personal_access_token)


from transformers import pipeline
prompt = "Somatic hypermutation allows the immune system to"

# generator = pipeline("text-generation", model=checkpoint, trust_remote_code=True, tokenizer=tokenizer)
# generator(prompt)

model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, device='cuda')
model = model.eval()
response, history = model.chat(tokenizer, "Hello", history=[])
print(response)
response, history = model.chat(tokenizer, "What should I do if I can't sleep at night", history=history)
print(response)