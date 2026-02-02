#!/bin/bash

# Configuration
# Load .env variables
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Ensure DOCKER_IMAGE is set
if [ -z "$DOCKER_IMAGE" ]; then
    echo "Error: DOCKER_IMAGE is not set in .env"
    exit 1
fi
WORKSPACE_DIR=$(pwd)
CONTAINER_WORKSPACE="/workspace"

# Function to build engine
build_model() {
    local dataset=$1
    local onnx_path=$2
    local plan_path=$3
    local input_name=$4
    local input_shape=$5  # e.g. "1x3x384x128" or "1x77"

    if [ ! -f "$onnx_path" ]; then
        echo "Skipping ${dataset} - ${input_name}: ONNX file not found at ${onnx_path}"
        return
    fi

    echo "=================================================="
    echo "Building Engine for ${dataset} - ${input_name}"
    echo "ONNX: ${onnx_path}"
    echo "PLAN: ${plan_path}"
    echo "=================================================="

    set -f
    
    if [[ "$input_name" == "image" ]]; then
        MIN_SHAPE="image:1x3x384x128"
        OPT_SHAPE="image:16x3x384x128"
        MAX_SHAPE="image:32x3x384x128"
        EXTRA_FLAGS="--fp16 --outputIOFormats=fp32:chw"
    else
        MIN_SHAPE="text:1x77"
        OPT_SHAPE="text:16x77"
        MAX_SHAPE="text:32x77"
        EXTRA_FLAGS="--fp16 --inputIOFormats=int32:chw --outputIOFormats=fp32:chw"
    fi

    # Run trtexec inside docker
    docker run --rm --gpus all \
        -v "${WORKSPACE_DIR}:${CONTAINER_WORKSPACE}" \
        -w "${CONTAINER_WORKSPACE}" \
        ${DOCKER_IMAGE} \
        trtexec \
        --onnx=${onnx_path} \
        --saveEngine=${plan_path} \
        ${EXTRA_FLAGS} \
        --minShapes=${MIN_SHAPE} \
        --optShapes=${OPT_SHAPE} \
        --maxShapes=${MAX_SHAPE} \
        --verbose
    
    set +f
}

# 1. CUHK-PEDES
build_model "CUHK-PEDES" \
    "irra_cuhk/irra_cuhk_image_encoder.onnx" \
    "irra_cuhk/irra_cuhk_image_encoder.plan" \
    "image"

build_model "CUHK-PEDES" \
    "irra_cuhk/irra_cuhk_text_encoder.onnx" \
    "irra_cuhk/irra_cuhk_text_encoder.plan" \
    "text"

# 2. ICFG-PEDES
build_model "ICFG-PEDES" \
    "irra_icfg/irra_icfg_image_encoder.onnx" \
    "irra_icfg/irra_icfg_image_encoder.plan" \
    "image"

build_model "ICFG-PEDES" \
    "irra_icfg/irra_icfg_text_encoder.onnx" \
    "irra_icfg/irra_icfg_text_encoder.plan" \
    "text"

# 3. RSTPReid
build_model "RSTPReid" \
    "irra_rstp/irra_rstp_image_encoder.onnx" \
    "irra_rstp/irra_rstp_image_encoder.plan" \
    "image"

build_model "RSTPReid" \
    "irra_rstp/irra_rstp_text_encoder.onnx" \
    "irra_rstp/irra_rstp_text_encoder.plan" \
    "text"

echo "All builds completed."
