import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
n = torch.cuda.device_count()
print(f"There are {n} GPUs available for torch.")
for i in range(n):
  name = torch.cuda.get_device_name(i)
  print(f"GPU {i}: {name}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)