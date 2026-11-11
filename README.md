# BHCast Pipeline Release

This release contains the necessary files to run the BHCast pipeline for GRMHD movie forecasting and plasma feature extraction.

## Contents

- `checkpoints/`: Directory containing all model files:
  - `model.pt`: Pre-trained UNet model checkpoint (118MB)
  - `data_processor_sgra.pkl`: Data processor for Sgr A* dataset normalization
  - `sgra_Ma_forecast.joblib`: XGBoost model for spin/inclination classification
  - `sgra_Ma_forecast.pkl`: Label map for XGBoost classification
- `images/`: Directory containing input images as numpy arrays (.npy files)
- `run_pipeline.sh`: Bash script to run the pipeline on all images
- `bhcast_pipeline.py`: Main pipeline script

## Installation

1. Ensure you have Python 3.7+ installed

2. Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```
3. CUDA-capable GPU (recommended) or CPU

4. XGBoost model and label map files are included in `checkpoints/` directory

## UV Installation

Clone the repository and sync the environment. uv will automatically download the required Python version and install all dependencies into a local .venv folder.

```uv sync```

You do not need to manually activate the virtual environment. Simply prefix your commands with uv run.

Open Jupyter Lab/Notebook:
uv run jupyter lab

## Download Checkpoints

- [Forecast model checkpoint (`model.pt`)](https://github.com/pi-vision/bhcast-release/releases/download/v1.0.1/model.pt) ([Google Drive mirror](https://drive.google.com/file/d/1pd8dYdACCvBe84eCsMWd2HLX1GM1usnM/view?usp=sharing))
- [XGBoost checkpoint (`sgra_Ma_forecast.joblib`)](https://drive.google.com/file/d/1ixg4iolxifYOSSCu3LnkmmETt0mGkmPW/view?usp=sharing)

Save the downloads in `checkpoints/`. The forecast checkpoint uses the original
trained weights (SHA-256 `96449f1d1a6acc38b59e329196f2541447b67f40052de404a11322c9dd5bc46e`).
Use `GRMHDInference.predict` to match the paper's single-image rollouts. Each
BatchNorm layer computes per-channel statistics from the current frame's feature
maps at each forecast step.

```bash
export XGB_MODEL=/path/to/custom_model.joblib
export LABEL_MAP=/path/to/custom_label_map.pkl
```

## Usage

### Jupyter Notebook

An interactive Jupyter Notebook (`bhcast_pipeline.ipynb`) is provided to run the pipeline step-by-step. The notebook is divided into three main sections:
1. **Frame Autoregressive Generation**: Generates future frames from an initial image using the GRMHD UNet model.
2. **Feature Computation**: Extracts plasma features (e.g., pattern speed, pitch angle, asymmetry) and visualizes the cylinder plot.
3. **Parameter Inference**: Uses the extracted features and a trained XGBoost model to predict the black hole's spin and inclination.


### Full Pipeline

Run the pipeline on all images in the `images/` directory:

```bash
./run_pipeline.sh
```

With uv the command would be:
```
uv run bash run_pipeline.sh
```

## Input Images

The script expects `.npy` files in the `images/` directory. Each file should contain a 2D numpy array representing a single frame.

The script will:
1. Convert each `.npy` file to a temporary PNG image
2. Run the pipeline to generate forecasts
3. Extract plasma features
4. (Optional) Run XGBoost classification if model/label_map are provided

## Output

Results are saved in the `output/` directory, with a subdirectory for each processed image:

```
output/
  sgra_spin-0.5_inc10/
    temp/              # Temporary HDF5 files
    pipeline_output.log # Pipeline log
  sgra_spin-0.5_inc110/
    ...
```

## Troubleshooting

1. **Pipeline script not found**: Update the `PIPELINE_SCRIPT` path in `run_pipeline.sh` to point to `bhcast_pipeline.py`

2. **CUDA out of memory**: Reduce `FORECAST_HORIZON`

3. **Missing XGBoost files**: The script will automatically look for them in the `checkpoints/` directory. If not found, it will prompt you to continue without classification, or you can provide custom paths via environment variables

4. **Image conversion errors**: Ensure numpy arrays are 2D (H, W) or can be squeezed to 2D

## Notes

- The script automatically extracts spin and inclination hints from filenames (format: `sgra_spin-0.5_inc10.npy`)
- Temporary files are kept by default (use `--keep_temp` flag in the script if you want to remove them)
- Each image is processed independently, so failures on one image won't stop processing of others

