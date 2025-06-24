#!/bin/bash
#SBATCH --job-name=eqtl_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=s-sc-gpu[002-028]
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<email>
#SBATCH --output=<sbatch log dir>/my_job_%j_%t.out
#SBATCH --error=<sbatch log dir>/my_job_%j_%t.err

# env params
export LD_LIBRARY_PATH=<path to conda env dir>/janusdna/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HYDRA_FULL_ERROR=1

source /etc/profile.d/conda.sh

conda activate janusdna

# work dir
cd <root path of this rep>

ls train.py || echo "train.py not found!"

# mission params
CELL_TYPE="Whole_Blood" 
LR="4e-4" # same as the pre-trained to locate the pre-trained dir
BATCH_SIZE=8
SEED=1
NUM_GPUS=1
FINETUNED_EPOCH=3
OUTPUT_ROUTER_LOGITS="true" 

# model params
MODEL="janusdna"
MODEL_NAME="dna_embedding_janusdna"
RC_AUG="false"
CONJOIN_TRAIN_DECODER="false"
CONJOIN_TEST="true"
FREEZE_BACKBONE="false"

MODEL_PRETRAINED_DIRNAME="janusdna_len-131k_d_model-144_inter_dim-576_n_layer-8_lr-8e-3_step-50K_moeloss-true_1head_onlymoe"


WANDB_NAME="${CELL_TYPE}_lr-${LR}_cjtrain_${CONJOIN_TRAIN_DECODER}_batch_${BATCH_SIZE}_seed_${SEED}"
ROOT_DIR="<root path of the repo>"
HYDRA_RUN_DIR="${ROOT_DIR}/outputs/downstream/longrange_benchmark/eqtl/${MODEL_PRETRAINED_DIRNAME}/${CELL_TYPE}/${WANDB_NAME}_cjtest_${CONJOIN_TEST}"
LOG_BASE_DIR="${ROOT_DIR}/watch_folder/DNALong/eQTL"
LOG_DIR="${LOG_BASE_DIR}/${MODEL_PRETRAINED_DIRNAME}/${CELL_TYPE}" 

LOG_FILE="${LOG_DIR}/${WANDB_NAME}_cjtest_${CONJOIN_TEST}.log" 
EVAL_OUTPUT_FILE="${LOG_DIR}/${WANDB_NAME}_cjtest_${CONJOIN_TEST}_output.txt"

mkdir -p "${HYDRA_RUN_DIR}"
mkdir -p "${LOG_DIR}"

FINETUNE_BASE_DIR="${ROOT_DIR}/outputs/downstream/longrange_benchmark/eqtl/${MODEL_PRETRAINED_DIRNAME}/${CELL_TYPE}/${WANDB_NAME}" 
CONFIG_PATH="${FINETUNE_BASE_DIR}/model_config.json"
PRETRAINED_WEIGHT_PATH="${FINETUNE_BASE_DIR}/checkpoints/last.ckpt"

srun python -m evaluation wandb=null experiment=hg38/eqtl \
        dataset.batch_size=1 \
        dataset.cell_type=${CELL_TYPE} \
        dataset.dest_path="${ROOT_DIR}/data" \
        +dataset.conjoin_test="${CONJOIN_TEST}" \
        model="${MODEL}" \
        model._name_="${MODEL_NAME}" \
        +model.config_path=${CONFIG_PATH} \
        +model.conjoin_test="${CONJOIN_TEST}" \
        +model.config.output_router_logits="${OUTPUT_ROUTER_LOGITS}" \
        +model.config.router_aux_loss_coef=0.02 \
        decoder.mode="pool" \
        train.pretrained_model_path=${PRETRAINED_WEIGHT_PATH} \
        train.pretrained_model_strict_load=True \
        +train.eval_log_path="${EVAL_OUTPUT_FILE}" \
        train.pretrained_model_state_hook._name_=null \
        train.test=True \
        +train.remove_val_loader_in_eval=True \
        train.remove_test_loader_in_eval=False \
        trainer.precision=32 \
        +decoder.conjoin_test="${CONJOIN_TEST}" \
        hydra.run.dir="${HYDRA_RUN_DIR}" \
        > ${LOG_FILE} 2>&1 
