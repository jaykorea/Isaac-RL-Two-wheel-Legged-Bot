import torch

# .pt 파일 로드
checkpoint = torch.load('Flamingo.pt')

# checkpoint의 키를 출력하여 구조를 확인
print("Checkpoint keys:", checkpoint.keys())

# model_state_dict의 키를 출력하여 구조를 확인
model_state_dict = checkpoint['model_state_dict']
print("Model state dict keys:", model_state_dict.keys())

# model_state_dict의 키를 출력하여 구조를 확인
model_state_dict = checkpoint['optimizer_state_dict']
print("Optimizer state dict keys:", model_state_dict.keys())
