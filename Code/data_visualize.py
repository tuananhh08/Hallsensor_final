from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_SENSOR_COUNT = 64

# BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026") #WINDOWS
BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data_8_2026") #MAC

# Hai file dien ap can so sanh. Moi file phai co 64 cot sensor.
# Thay doi hai duong dan nay khi can ve cap du lieu khac.
INPUT_CSV_1 = BASE_DIR / "Helix_data_2.csv"
INPUT_CSV_2 = BASE_DIR / "Helix_voltage_computed.csv"
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


def plot_sensor(
        data_1: pd.DataFrame,
        sensor_index: int,
        output_dir: Path,
        data_2: pd.DataFrame,
        label_1: str = "Input 1",
        label_2: str = "Input 2") -> None:
    """Overlay the two input signals for one sensor and save one PNG."""
    column_1 = data_1.columns[sensor_index]
    column_2 = data_2.columns[sensor_index]
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        data_1.index.to_numpy(), data_1[column_1].to_numpy(),
        linewidth=0.95, color="tab:orange", label=label_1,
    )
    ax.plot(
        data_2.index.to_numpy(), data_2[column_2].to_numpy(),
        linewidth=0.95, color="tab:blue", label=label_2,
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"Voltage {column_1}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / f"Sensor {sensor_index + 1:02d}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    plt.close(fig)


def main() -> None:
    """Create 64 per-sensor PNGs, each overlaying the two input files."""
    data_1 = load_voltage_data(INPUT_CSV_1)
    data_2 = load_voltage_data(INPUT_CSV_2)
    if list(data_1.columns) != list(data_2.columns):
        raise ValueError("Hai file input phai co cung ten va thu tu 64 cot sensor.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sensor_index in range(EXPECTED_SENSOR_COUNT):
        plot_sensor(
            data_1, sensor_index, OUTPUT_DIR, data_2=data_2,
            label_1=INPUT_CSV_1.stem,
            label_2=INPUT_CSV_2.stem,
        )

    print(f"Da luu {EXPECTED_SENSOR_COUNT} bieu do (2 duong/sensor) vao: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
