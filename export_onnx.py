import torch
from model import Model  # Import class Model từ file model.py của bạn

# Load model
checkpoint = torch.load(r'D:\Downloads\Hallsensor_final\Code\best_orientation.pt', map_location='cpu')
# Lưu ý có chữ r phía trước để tránh lỗi dấu gạch chéo ngược (\)model.eval()

# ── 1. Khởi tạo kiến trúc model ─────────────────────────────────
model = Model(out_dim=5, drop_path_rate=0.035)


new_state_dict = {}
for key, value in checkpoint['model'].items():
    new_key = key.replace('_orig_mod.', '')
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
model.eval()

print("✅ Model loaded thành công!")

# ── Export ONNX với legacy exporter ─────────────────────────────
dummy_input = torch.randn(1, 1, 8, 8)

with torch.no_grad():  # Thêm no_grad để tránh track gradients
    torch.onnx.export(
        model,
        dummy_input,
        'best.onnx',
        export_params=True,
        opset_version=12,          # Legacy hỗ trợ tốt opset 11-13
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None,
        verbose=True,              # Bật verbose để debug
        dynamo=False,              # ← BẮT BUỘC: dùng legacy exporter
    )

print("✅ Export ONNX thành công!")

# ── Verify ONNX ─────────────────────────────────────────────────
import onnx
import onnxruntime as ort
import numpy as np

onnx_model = onnx.load('best.onnx')
onnx.checker.check_model(onnx_model)
print("✅ ONNX model hợp lệ!")

# Test inference
session = ort.InferenceSession('best.onnx')
dummy_np = np.random.randn(1, 1, 8, 8).astype(np.float32)
output = session.run(None, {'input': dummy_np})

print(f"\n📊 Model info:")
print(f"   Input : {session.get_inputs()[0].shape}")
print(f"   Output: {session.get_outputs()[0].shape}")
print(f"   Sample: {output[0][0]}")  # [x, y, z, ang1, ang2]