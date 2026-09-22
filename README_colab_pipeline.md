# Pipeline Training 3 Phase 

**định vị 5DoF của viên nang từ mảng 64 cảm biến Hall (8x8)**.

---

## 1. Tổng quan Kiến trúc và Pipeline

| Thành phần | File | Input | Target/Output | Vai trò |
|---|---|---|---|---|
| **Phase 1: CalibNet** | `calibnet.py` | `Grid_data.csv` (Raw voltage 64) | `Grid_data_computed.csv` (Clean voltage 64) | Học ánh xạ từ điện áp thực tế (nhiễu/phi tuyến) sang điện áp lý thuyết/chuẩn. |
| **Phase 2: LocNet** | `model.py` | `synthetic_grid_data2.csv` | `synthetic_grid_coordinates2.csv` (x,y,z,mx,my,mz) | Học định vị 5DoF trên tập dữ liệu mô phỏng lớn. |
| **Phase 3: Finetune** | `train_mvec.py` | `Grid_data.csv` | `Grid_points_coordinates.csv` + `Grid_data_computed.csv` | Ghép nối CalibNet + LocNet, fine-tune end-to-end trên tập dữ liệu đo thực tế với hàm loss kết hợp (Calib Loss + Pose Loss + Physics Loss). |

---

Định nghĩa đường dẫn thư mục dữ liệu và checkpoint:
```bash
DATA_DIR="/content/drive/MyDrive/Hallsensor_final/Data_8_2026"
CODE_DIR="/content/drive/MyDrive/Hallsensor_final/Code"
CKPT_DIR="/content/drive/MyDrive/Hallsensor_final/ckpt_mvec"
```

---

### Phase 1 — Train CalibNet

Chỉ huấn luyện mạng **CalibNet** để học mapping `raw -> clean`:

```bash
python $CODE_DIR/train_mvec.py \
  --phase calibnet \
  --raw-voltage $DATA_DIR/Grid_data.csv \
  --clean-voltage $DATA_DIR/Grid_data_computed.csv \
  --raw-label $DATA_DIR/Grid_points_coordinates.csv \
  --synthetic-voltage $DATA_DIR/synthetic_grid_data2.csv \
  --synthetic-label $DATA_DIR/synthetic_grid_coordinates2.csv \
  --calib-physical-csv $DATA_DIR/Calibration_Physical_new.csv \
  --calib-alpha-csv $DATA_DIR/Calibration_Alpha_new.csv \
  --ckpt-dir $CKPT_DIR \
  --batch-size 256 \
  --num-epochs 200 \
  --lr-calibnet 2e-4
```
> **Output checkpoint:** `$CKPT_DIR/calibnet_pretrained.pt` và `$CKPT_DIR/scalers.pkl`

---

### Phase 2 — Train LocNet

Chỉ huấn luyện mạng **LocNet** (không dùng CalibNet) trên tập synthetic:

```bash
python $CODE_DIR/train_mvec.py \
  --phase locnet \
  --raw-voltage $DATA_DIR/Grid_data.csv \
  --clean-voltage $DATA_DIR/Grid_data_computed.csv \
  --raw-label $DATA_DIR/Grid_points_coordinates.csv \
  --synthetic-voltage $DATA_DIR/synthetic_grid_data2.csv \
  --synthetic-label $DATA_DIR/synthetic_grid_coordinates2.csv \
  --calib-physical-csv $DATA_DIR/Calibration_Physical_new.csv \
  --calib-alpha-csv $DATA_DIR/Calibration_Alpha_new.csv \
  --ckpt-dir $CKPT_DIR \
  --scaler-file $CKPT_DIR/scalers.pkl \
  --batch-size 256 \
  --num-epochs 200 \
  --lr-locnet 1e-3 \
  --no-physics
```
> **Output checkpoint:** `$CKPT_DIR/locnet_pretrained.pt`

---

### Phase 3 — Fine-tune End-to-End Toàn Bộ Mô Hình

Ghép 2 checkpoint đã train ở Phase 1 và Phase 2 để fine-tune cùng lúc trên tập raw:

```bash
python $CODE_DIR/train_mvec.py \
  --phase finetune \
  --calibnet-checkpoint $CKPT_DIR/calibnet_pretrained.pt \
  --locnet-checkpoint $CKPT_DIR/locnet_pretrained.pt \
  --raw-voltage $DATA_DIR/Grid_data.csv \
  --clean-voltage $DATA_DIR/Grid_data_computed.csv \
  --raw-label $DATA_DIR/Grid_points_coordinates.csv \
  --synthetic-voltage $DATA_DIR/synthetic_grid_data2.csv \
  --synthetic-label $DATA_DIR/synthetic_grid_coordinates2.csv  \
  --calib-physical-csv $DATA_DIR/Calibration_Physical_new.csv  \
  --calib-alpha-csv $DATA_DIR/Calibration_Alpha_new.csv \
  --ckpt-dir $CKPT_DIR \
  --scaler-file $CKPT_DIR/scalers.pkl \
  --batch-size 256 \
  --num-epochs 200 \
  --lr-calibnet 5e-5 \
  --lr-locnet 2e-4 \
  --lambda-calib 0.1 \
  --lambda-physics 1e-4
```
> **Output checkpoint:** `$CKPT_DIR/full_model_best.pt`

---

### Test & Đánh Giá Mô Hình

Đánh giá độ chính xác định vị vị trí (mm) và hướng (độ) trên tập test:

```bash
python $CODE_DIR/test_mvec.py \
  --test_voltage $DATA_DIR/Helix_data.csv \
  --test_label $DATA_DIR/Helix_points_coordinates.csv \
  --ckpt_dir $CKPT_DIR \
  --checkpoint $CKPT_DIR/full_model_best.pt \
  --code_dir $CODE_DIR \
  --out $CKPT_DIR/test_result_3d.png
```
> **Kết quả xuất ra:**
> - `testresult.csv`: Chi tiết tọa độ dự đoán và sai số từng điểm.
> - `test_result_3d.png`: Biểu đồ quỹ đạo 3D thực tế vs dự đoán.
> - `position_error.png`: Đồ thị sai số vị trí theo từng sample.
> - `orientation_error.png`: Đồ thị sai số góc theo từng sample.
