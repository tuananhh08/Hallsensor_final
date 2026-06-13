import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def convert_angles_to_cosine(input_file: Path, output_file: Path) -> None:
    df = pd.read_csv(input_file)

    missing_cols = [col for col in ["pitch", "yaw"] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu cột bắt buộc: {', '.join(missing_cols)}")

    cos_pitch = np.cos(np.deg2rad(df["pitch"]))
    cos_yaw = np.cos(np.deg2rad(df["yaw"]))

    for col in ["x", "y", "z"]:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột bắt buộc: {col}")

    output_df = df[["x", "y", "z"]].copy()
    output_df["cos_pitch"] = cos_pitch
    output_df["cos_yaw"] = cos_yaw

    output_df.to_csv(output_file, index=False)
    print(f"Đã tạo file: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Chuyển giá trị góc độ ở cột pitch/yaw thành cos_pitch/cos_yaw, "
            "thay thế dữ liệu cũ và chỉ giữ 5 cột."
        )
    )
    parser.add_argument(
        "--input",
        default=r"D:\Downloads\Hallsensor_final\Data set\Coordinate\Helix_points.csv",
        help="Đường dẫn file CSV đầu vào (mặc định theo yêu cầu bài toán).",
    )
    parser.add_argument(
        "--output",
        default="D:\\Downloads\\Hallsensor_final\\Data set\\Coordinate\\Helix.csv",
        help="Tên/đường dẫn file CSV đầu ra.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {input_path}")

    convert_angles_to_cosine(input_path, output_path)


if __name__ == "__main__":
    main()
