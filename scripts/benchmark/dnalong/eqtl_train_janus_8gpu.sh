# !/bin/bash
# SBATCH --job-name=AS_w/o_midattn
# SBATCH --partition=pgpu
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=8
# SBATCH --gres=gpu:8
# SBATCH --cpus-per-task=12
# SBATCH --mem=128G
# SBATCH --exclusive
# SBATCH --requeue
# SBATCH --time=2-00:00:00
# SBATCH --nodelist=s-sc-pgpu[08],s-sc-dgx[01-02]
# SBATCH --mail-type=ALL
# SBATCH --mail-user=<email>
# SBATCH --output=<path to sbatch log dir>/my_job%j.out
# SBATCH --error=<path to sbatch log dir>/my_job%j.err


export LD_LIBRARY_PATH=<path to conda env dir>/janusdna/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HYDRA_FULL_ERROR=1

source /etc/profile.d/conda.sh

conda activate janusdna

# cd working directory
cd <root path of the repo>



# make sure train.py exists
ls train.py || echo "train.py not found!"

# CELL_TYPES=( 
#     # "Adipose_Subcutaneous"
#     "Artery_Tibial"
#     "Cells_Cultured_fibroblasts"
#     "Muscle_Skeletal"
#     "Nerve_Tibial"
#     "Skin_Not_Sun_Exposed_Suprapubic"
#     "Skin_Sun_Exposed_Lower_leg"
#     "Thyroid"
#     "Whole_Blood"
# )


# task params
CELL_TYPE="Whole_Blood"  # pich one from above list, e.g., Cells_Cultured_fibroblasts, Adipose_Subcutaneous
LR="4e-4"
BATCH_SIZE=8
SEED=1
NUM_GPUS=8
FINETUNED_EPOCH=3
OUTPUT_ROUTER_LOGITS="true"  # define missing variable

# model config
MODEL="janusdna"
MODEL_NAME="dna_embedding_janusdna"
RC_AUG="false"
CONJOIN_TRAIN_DECODER="false"
CONJOIN_TEST="false"
FREEZE_BACKBONE="false"

# pretrained model path
ROOT_DIR="<root path of the repo>"
PRETRAINED_DIR="${ROOT_DIR}/outputs/pretrain/hg38"
MODEL_PRETRAINED_DIRNAME="janusdna_len-131k_d_model-144_inter_dim-576_n_layer-8_lr-8e-3_step-50K_moeloss-true_1head_onlymoe" # copy the pre-trained model directory name
PRETRAINED_CONFIG_PATH="${PRETRAINED_DIR}/${MODEL_PRETRAINED_DIRNAME}/model_config.json"
PRETRAINED_WEIGHT_PATH="${PRETRAINED_DIR}/${MODEL_PRETRAINED_DIRNAME}/checkpoints/last.ckpt"

# finetuned model path
FINETUNED_BASE_DIR="${ROOT_DIR}/outputs/downstream/longrange_benchmark/eqtl"
MODEL_FINETUNED_DIRNAME=${MODEL_PRETRAINED_DIRNAME}

# log path
LOG_BASE_DIR="${ROOT_DIR}/watch_folder/DNALong/eQTL"

# name and log setting
WANDB_NAME="${CELL_TYPE}_lr-${LR}_cjtrain_${CONJOIN_TRAIN_DECODER}_batch_${BATCH_SIZE}_seed_${SEED}"
HYDRA_RUN_DIR="${FINETUNED_BASE_DIR}/${MODEL_FINETUNED_DIRNAME}/${CELL_TYPE}/${WANDB_NAME}"
LOG_DIR="${LOG_BASE_DIR}/${MODEL_FINETUNED_DIRNAME}/${CELL_TYPE}"
LOG_FILE="${LOG_DIR}/${WANDB_NAME}.log"

mkdir -p "${HYDRA_RUN_DIR}"
mkdir -p "${LOG_DIR}"

echo "Running cell_type: ${CELL_TYPE}, Pretrained_model: ${MODEL_PRETRAINED_DIRNAME}, LR: ${LR}, BATCH_SIZE: ${BATCH_SIZE}, SEED: ${SEED}"
echo "Logging to: ${LOG_FILE}"

# wandb id
export WANDBID=$(python -c "import wandb; print(wandb.util.generate_id())" 2>/dev/null || echo "wandb_id_error")

srun python -m train \
    experiment=hg38/eqtl \
    dataset.batch_size=$((BATCH_SIZE / NUM_GPUS)) \
    dataset.dest_path="${ROOT_DIR}/data" \
    dataset.cell_type=${CELL_TYPE} \
    loader.num_workers=0 \
    model="${MODEL}" \
    model._name_="${MODEL_NAME}" \
    +model.config_path=${PRETRAINED_CONFIG_PATH} \
    +model.config.output_router_logits="${OUTPUT_ROUTER_LOGITS}" \
    +model.config.router_aux_loss_coef=0.02 \
    decoder.mode="pool" \
    optimizer.lr=${LR} \
    train.pretrained_model_path="${PRETRAINED_WEIGHT_PATH}" \
    train.pretrained_model_state_hook.freeze_backbone=${FREEZE_BACKBONE} \
    train.monitor=val/main_loss_epoch \
    train.global_batch_size=${BATCH_SIZE} \
    trainer.num_sanity_val_steps=1 \
    trainer.max_epochs=${FINETUNED_EPOCH} \
    trainer.precision=32 \
    trainer.devices=${NUM_GPUS} \
    +trainer.strategy="ddp_find_unused_parameters_true" \
    wandb.mode=online \
    wandb.name="${WANDB_NAME}" \
    wandb.id=${WANDBID} \
    wandb.group=DNALong/eQTL/janus-onlymoe/131k_50k \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    > "${LOG_FILE}" 2>&1




