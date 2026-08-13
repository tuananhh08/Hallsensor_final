from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_SENSOR_COUNT = 64

BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026")
INPUT_CSV = BASE_DIR / "Grid_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "sensor_voltage_plots"


def load_voltage_data(csv_path: Path) -> pd.DataFrame:
    """Read and validate a voltage CSV with one column per sensor."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"File not found: {csv_path}")

    data = pd.read_csv(csv_path)
    if data.shape[1] != EXPECTED_SENSOR_COUNT:
        raise ValueError(
            f"File phai co {EXPECTED_SENSOR_COUNT} cot sensor, "
            f"nhung nhan duoc {data.shape[1]} cot."
        )
    if data.empty:
        raise ValueError("File input khong co dong du lieu nao.")

    non_numeric = data.columns[data.apply(lambda col: not pd.api.types.is_numeric_dtype(col))]
    if len(non_numeric):
        raise ValueError(f"Cot khong phai du lieu so: {', '.join(non_numeric)}")
    return data


def plot_sensor(data: pd.DataFrame, sensor_index: int, output_dir: Path) -> None:
    """Draw and save the voltage curve of one sensor column."""
    column = data.columns[sensor_index]
    fig, ax = plt.subplots(figsize=(12, 5))
    sample_index = data.index.to_numpy()

    ax.plot(sample_index, data[column].to_numpy(), linewidth=0.95, color="tab:orange")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"Voltage {column}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = output_dir / f"Sensor {sensor_index + 1:02d}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    plt.close(fig)


def main() -> None:
    """Create one voltage plot for each of the 64 sensor columns."""
    data = load_voltage_data(INPUT_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sensor_index in range(EXPECTED_SENSOR_COUNT):
        plot_sensor(data, sensor_index, OUTPUT_DIR)

    print(f"Da luu {EXPECTED_SENSOR_COUNT} bieu do vao: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

