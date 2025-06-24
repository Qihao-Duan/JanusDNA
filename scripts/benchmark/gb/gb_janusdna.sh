#!/bin/bash

# environment variables
export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH=<path to conda env>/janusdna/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

source /etc/profile.d/conda.sh
conda activate janusdna

cd <root path of the repo>

PROJECT_ROOT_DIR="<root path of the repo>"

# datasets
DATASETS=(
    "dummy_mouse_enhancers_ensembl"
    "demo_coding_vs_intergenomic_seqs"
    "demo_human_or_worm"
    "human_enhancers_cohn"
    "human_enhancers_ensembl"
    "human_ensembl_regulatory"
    "human_nontata_promoters"
    "human_ocr_ensembl"
)

LRS=("1e-3" "2e-3")
BATCH_SIZES=(128 256 512)
SEEDS=(1 2 3 4 5)
NUM_GPUS=8

# pretrained model path and config
CONFIG_PATH=$(realpath "${PROJECT_ROOT_DIR}/outputs/pretrain/hg38/janusdna_len-1k_d_model-32_inter_dim-128_n_layer-8_lr-8e-3_step-10K_moeloss-true_1head_midattn/model_config.json")
PRETRAINED_PATH=$(realpath "${PROJECT_ROOT_DIR}/outputs/pretrain/hg38/janusdna_len-1k_d_model-32_inter_dim-128_n_layer-8_lr-8e-3_step-10K_moeloss-true_1head_midattn/checkpoints/last.ckpt")

# model parameters
# name should be same as the pre-trained one
DISPLAY_NAME="janusdna_len-1k_d_model-32_inter_dim-128_n_layer-8_lr-8e-3_step-10K_moeloss-true_1head_midattn"
MODEL="janusdna"
MODEL_NAME="dna_embedding_janusdna"
RC_AUG="false"
CONJOIN_TRAIN_DECODER="false"
CONJOIN_TEST="true"

HYDRA_RUN_DIR="${PROJECT_ROOT_DIR}/outputs/gb"

LOG_DIR="${PROJECT_ROOT_DIR}/watch_folder/gb/${DISPLAY_NAME}"
mkdir -p "${LOG_DIR}"

# task queue
declare -A GPU_TASKS


run_task() {
    local task=$1
    local gpu_id=$2
    local lr=$3
    local batch_size=$4
    local seed=$5

    local WANDB_NAME="${DISPLAY_NAME}_LR-${lr}_BATCH_SIZE-${batch_size}"
    local hydra_run_dir="${HYDRA_RUN_DIR}/${WANDB_NAME}/${task}/seed-${seed}"
    mkdir -p "${hydra_run_dir}"

    local LOG_FILE="${LOG_DIR}/${task}_gpu-${gpu_id}_lr-${lr}_batch-${batch_size}_seed-${seed}.log"

    local WANDBID=$(python -c "import wandb; print(wandb.util.generate_id())")

    echo "Running Task: ${task}, GPU: ${gpu_id}, LR: ${lr}, BATCH_SIZE: ${batch_size}, SEED: ${seed}"
    echo "Logging to: ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=${gpu_id} nohup python -m train \
        experiment=hg38/genomic_benchmark \
        dataset.dataset_name="${task}" \
        dataset.train_val_split_seed="${seed}" \
        dataset.batch_size=${batch_size} \
        dataset.rc_aug="${RC_AUG}" \
        +dataset.conjoin_test="${CONJOIN_TEST}" \
        optimizer.lr="${lr}" \
        model="${MODEL}" \
        model._name_="${MODEL_NAME}" \
        +model.config_path="${CONFIG_PATH}" \
        +model.conjoin_test="${CONJOIN_TEST}" \
        +decoder.conjoin_train="${CONJOIN_TRAIN_DECODER}" \
        +decoder.conjoin_test="${CONJOIN_TEST}" \
        train.pretrained_model_path="${PRETRAINED_PATH}" \
        trainer.max_epochs=10 \
        trainer.devices=1 \
        trainer.precision=bf16 \
        wandb.mode="offline" \
        wandb.group="downstream/gb/${task}" \
        wandb.job_type="${task}" \
        wandb.name="${WANDB_NAME}" \
        wandb.id="${WANDBID}" \
        +wandb.tags=\["seed-${seed}"\]  \
        hydra.run.dir="${HYDRA_RUN_DIR}" \
        > "${LOG_FILE}" 2>&1 &

    GPU_TASKS[${gpu_id}]=$!
    echo "Started task on GPU ${gpu_id}, PID: ${GPU_TASKS[${gpu_id}]}"
    sleep 5  # Reduce contention
}


schedule_tasks() {
    local tasks=()

    for dataset in "${DATASETS[@]}"; do
        for lr in "${LRS[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    tasks+=("${dataset} ${lr} ${batch_size} ${seed}")
                done
            done
        done
    done

    local total_tasks=${#tasks[@]}
    local task_index=0

    # initially launch tasks on available GPUs
    for ((gpu_id=0; gpu_id<NUM_GPUS && task_index<total_tasks; gpu_id++)); do
        IFS=' ' read -r dataset lr batch_size seed <<< "${tasks[$task_index]}"
        run_task "${dataset}" "${gpu_id}" "${lr}" "${batch_size}" "${seed}"
        ((task_index++))
    done

    # dynamically schedule tasks
    while (( task_index < total_tasks )); do
        for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
            if ! kill -0 ${GPU_TASKS[$gpu_id]} 2>/dev/null; then
                echo "GPU ${gpu_id} finished, launching new task..."
                IFS=' ' read -r dataset lr batch_size seed <<< "${tasks[$task_index]}"
                run_task "${dataset}" "${gpu_id}" "${lr}" "${batch_size}" "${seed}"
                ((task_index++))
            fi
        done
        sleep 5  # polling interval
    done

    wait
    echo "All tasks completed."
}

# Start task scheduling
schedule_tasks


