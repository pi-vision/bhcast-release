#!/bin/bash
# BHCast Pipeline Runner Script
# This script runs the BHCast pipeline on all images in the images/ directory
# using the model checkpoint and data processor from the release folder.

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$SCRIPT_DIR"

# Paths to model and data processor (all in checkpoints directory)
CHECKPOINTS_DIR="$RELEASE_DIR/checkpoints"
CHECKPOINT_PATH="$CHECKPOINTS_DIR/model.pt"
DATA_PROCESSOR_PATH="$CHECKPOINTS_DIR/data_processor_sgra.pth"
IMAGES_DIR="$RELEASE_DIR/images"

# Pipeline script path (adjust if needed)
PIPELINE_SCRIPT="${SCRIPT_DIR}/bhcast_pipeline.py"

# Check if pipeline script exists
if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "Error: Pipeline script not found at $PIPELINE_SCRIPT"
    echo "Please update PIPELINE_SCRIPT path in this script."
    exit 1
fi

# Check if checkpoints directory exists
if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "Error: Checkpoints directory not found at $CHECKPOINTS_DIR"
    exit 1
fi

# Check if required files exist
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Model checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -f "$DATA_PROCESSOR_PATH" ]; then
    echo "Error: Data processor not found at $DATA_PROCESSOR_PATH"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory not found at $IMAGES_DIR"
    exit 1
fi

# XGBoost model and label map (default to checkpoints directory, can be overridden)
XGB_MODEL="${XGB_MODEL:-$CHECKPOINTS_DIR/sgra_Ma_forecast.joblib}"
LABEL_MAP="${LABEL_MAP:-$CHECKPOINTS_DIR/sgra_Ma_forecast.pkl}"

# Check if XGBoost files exist
if [ ! -f "$XGB_MODEL" ] || [ ! -f "$LABEL_MAP" ]; then
    echo "Warning: XGBoost model or label map not found."
    echo "  XGB Model: $XGB_MODEL"
    echo "  Label Map: $LABEL_MAP"
    echo ""
    echo "Set XGB_MODEL and LABEL_MAP environment variables to override, or:"
    echo "  export XGB_MODEL=/path/to/model.joblib"
    echo "  export LABEL_MAP=/path/to/label_map.pkl"
    echo ""
    read -p "Continue without XGBoost inference? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    # Unset so they won't be passed to pipeline
    XGB_MODEL=""
    LABEL_MAP=""
fi

# Create output directory
OUTPUT_DIR="$RELEASE_DIR/output"
mkdir -p "$OUTPUT_DIR"

# Default parameters
DEVICE="${DEVICE:-cuda}"
DATASET_NAME="${DATASET_NAME:-sgra}"
FORECAST_HORIZON="${FORECAST_HORIZON:-60}"
IN_CHANNELS="${IN_CHANNELS:-1}"
OUT_CHANNELS="${OUT_CHANNELS:-1}"

# Process each .npy file in the images directory
echo "=========================================="
echo "BHCast Pipeline Runner"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data Processor: $DATA_PROCESSOR_PATH"
if [ -n "$XGB_MODEL" ] && [ -n "$LABEL_MAP" ]; then
    echo "XGBoost Model: $XGB_MODEL"
    echo "Label Map: $LABEL_MAP"
fi
echo "Images Directory: $IMAGES_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "=========================================="
echo ""

# Find all .npy files
NPY_FILES=("$IMAGES_DIR"/*.npy)

if [ ${#NPY_FILES[@]} -eq 0 ] || [ ! -f "${NPY_FILES[0]}" ]; then
    echo "Error: No .npy files found in $IMAGES_DIR"
    exit 1
fi

echo "Found ${#NPY_FILES[@]} image file(s) to process"
echo ""

# Process each file
for npy_file in "${NPY_FILES[@]}"; do
    if [ ! -f "$npy_file" ]; then
        continue
    fi
    
    filename=$(basename "$npy_file" .npy)
    echo "Processing: $filename"
    
    # Extract spin and inclination from filename if possible
    # Format: sgra_spin-0.5_inc10.npy
    SPIN_HINT=0.0
    INCLINATION_HINT=0
    
    if [[ "$filename" =~ spin([+-]?[0-9.]+) ]]; then
        SPIN_HINT="${BASH_REMATCH[1]}"
    fi
    
    if [[ "$filename" =~ inc([0-9]+) ]]; then
        INCLINATION_HINT="${BASH_REMATCH[1]}"
    fi
    
    # Create output subdirectory for this image
    IMAGE_OUTPUT_DIR="$OUTPUT_DIR/$filename"
    mkdir -p "$IMAGE_OUTPUT_DIR"
    
    # Run the pipeline (pass .npy file directly - pipeline will load it and use first frame)
    echo "  Running pipeline..."
    python3 "$PIPELINE_SCRIPT" \
        --input_image "$npy_file" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --data_processor_path "$DATA_PROCESSOR_PATH" \
        --dataset_name "$DATASET_NAME" \
        --device "$DEVICE" \
        --in_channels "$IN_CHANNELS" \
        --out_channels "$OUT_CHANNELS" \
        --forecast_horizon "$FORECAST_HORIZON" \
        --spin_hint "$SPIN_HINT" \
        --inclination_hint "$INCLINATION_HINT" \
        --temp_dir "$IMAGE_OUTPUT_DIR/temp" \
        --keep_temp \
        ${XGB_MODEL:+--xgb_model "$XGB_MODEL"} \
        ${LABEL_MAP:+--label_map "$LABEL_MAP"} \
        > "$IMAGE_OUTPUT_DIR/pipeline_output.log" 2>&1 || {
        echo "  Error: Pipeline failed for $filename. Check log: $IMAGE_OUTPUT_DIR/pipeline_output.log"
        continue
    }
    
    echo "  Completed: $filename"
    echo "  Output directory: $IMAGE_OUTPUT_DIR"
    echo ""
done

echo "=========================================="
echo "Pipeline processing complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

