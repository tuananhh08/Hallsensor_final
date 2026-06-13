import torch
from model import Model

model = Model(out_dim=5)
ckpt  = torch.load("./ckpt/best.pt", map_location="cpu", weights_only=False)

# Xử lý compiled state dict
state = {k.replace("_orig_mod.", ""): v
         for k, v in ckpt["model"].items()}
model.load_state_dict(state)
model.eval()

# Đếm params
n_params = sum(p.numel() for p in model.parameters())
size_kb  = sum(p.numel() * p.element_size()
               for p in model.parameters()) / 1024

print(f"Parameters : {n_params:,}")
print(f"Model size : {size_kb:.1f} KB  (float32)")
print(f"           : {size_kb/2:.1f} KB  (float16 / int8 quantized ÷2)")

# Kiểm tra MACs (số phép tính)
try:
    from thop import profile
    dummy = torch.randn(1, 1, 8, 8)
    macs, _ = profile(model, inputs=(dummy,), verbose=False)
    print(f"MACs       : {macs/1e6:.2f} M")
except ImportError:
    print("pip install thop  để đếm MACs")