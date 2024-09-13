from toygpt_LSTM import toyGPT
import torch

model = toyGPT(tokenizer_path= "/home/ChenYufan/gpt/tokenizer")
model.load_state_dict(torch.load("/home/ChenYufan/gpt/model_LSTM/model_iter9999.pth")["model"])
prompt = "One day"
print('prompt:',prompt)
print('output:',model.generate(prompt = prompt, max_new_tokens=400))