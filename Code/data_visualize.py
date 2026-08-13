from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_SENSOR_COUNT = 64

# BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026") #WINDOWS
BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data_8_2026") #MAC

INPUT_CSV = BASE_DIR / "Grid_data.csv"
# Set to a second 64-column voltage CSV to create a second subplot per sensor.
# Keep None to preserve the original one-input-file behaviour.
INPUT_CSV_2 = BASE_DIR / "Helix_data_2.csv"
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
        data: pd.DataFrame,
        sensor_index: int,
        output_dir: Path,
        data_2: pd.DataFrame | None = None,
        label_1: str = "Input 1",
        label_2: str = "Input 2") -> None:
    """Draw one or two independent subplots for the same sensor and save one PNG."""
    column = data.columns[sensor_index]
    n_plots = 2 if data_2 is not None else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), squeeze=False)
    ax_1 = axes[0, 0]

    ax_1.plot(
        data.index.to_numpy(), data[column].to_numpy(),
        linewidth=0.95, color="tab:orange", label=label_1,
    )
    ax_1.set_xlabel("Sample index")
    ax_1.set_ylabel("Voltage (V)")
    ax_1.set_title(f"{label_1}: {column}" if data_2 is not None else f"Voltage {column}")
    ax_1.grid(True, alpha=0.3)

    if data_2 is not None:
        column_2 = data_2.columns[sensor_index]
        ax_2 = axes[1, 0]
        ax_2.plot(
            data_2.index.to_numpy(), data_2[column_2].to_numpy(),
            linewidth=0.95, color="tab:blue", label=label_2,
        )
        ax_2.set_xlabel("Sample index")
        ax_2.set_ylabel("Voltage (V)")
        ax_2.set_title(f"{label_2}: {column_2}")
        ax_2.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = output_dir / f"Sensor {sensor_index + 1:02d}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    plt.close(fig)


def main() -> None:
    """Create 64 per-sensor PNGs, with an optional second independent subplot."""
    data = load_voltage_data(INPUT_CSV)
    data_2 = load_voltage_data(INPUT_CSV_2) if INPUT_CSV_2 is not None else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sensor_index in range(EXPECTED_SENSOR_COUNT):
        plot_sensor(
            data, sensor_index, OUTPUT_DIR, data_2=data_2,
            label_1=INPUT_CSV.stem,
            label_2=INPUT_CSV_2.stem if INPUT_CSV_2 is not None else "",
        )

    n_subplots = 2 if data_2 is not None else 1
    print(f"Da luu {EXPECTED_SENSOR_COUNT} bieu do ({n_subplots} subplot/sensor) vao: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
