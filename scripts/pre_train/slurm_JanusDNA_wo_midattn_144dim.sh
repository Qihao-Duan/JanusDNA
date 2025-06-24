#!/bin/bash
####SBATCH --job-name=janus_w/o_midattn      # Specify job name
####SBATCH --partition=pgpu        # Specify partition name
####SBATCH --nodes=1              # Specify number of nodes
####SBATCH --gres=gpu:8           # Generic resources; 1 GPU
####SBATCH --ntasks-per-node=8    # each gpu is a task, a 4 gpu mission requires 4 tasks
####SBATCH --cpus-per-task=12    
####SBATCH --mem=128G              # Request memory
####SBATCH --exclusive
####SBATCH --requeue
####SBATCH --time=2-00:00:00             # Set a limit on the total run time
####SBATCH --nodelist=s-sc-pgpu[01-08]  
####SBATCH --mail-type=ALL       # Notify user by email in case of job failure
####SBATCH --mail-user=<personal email for notification>
#### SBATCH --output=<path to sbatch report dir>/my_job%j    # File name for standard output
#### SBATCH --error=<path to sbatch report dir>/my_job%j     # File name for standard error output





export LD_LIBRARY_PATH=<path to conda env>/janusdna/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

source /etc/profile.d/conda.sh
conda activate janusdna
cd <path to the root of the repo>

full_path_to_root="<path to the root of the repo>"


export HYDRA_FULL_ERROR=1

NUM_DEVICES=8

# Run script
SEQLEN=131072
MAX_STEPS=50000
GRADIENT_UPDATE_NUM="$((MAX_STEPS / 1000))K"

D_MODEL=144
FLEX_ATTN_MODEL=256 # should be multiple of 2, and 64 is the minimum.
INTER_FFN_MODEL=576 # 4x is the best

N_LAYER=8
LR="8e-3"

RCPS="false"
RC_AUG="false"
BIDIRECTIONAL_WEIGHT_TIE="true"
BIDIRECTIONAL_ATTN_TIE="false"
ROUTER_AUX_LOSS_COEF=0.2
OUTPUT_ROUTER_LOGITS="true"

BATCH_SIZE=$(( 1048576 / SEQLEN ))

SEQLEN_DIS="$((SEQLEN / 1000))k"
WANDB_NAME="janusdna_len-${SEQLEN_DIS}_d_model-${D_MODEL}_inter_dim-${INTER_FFN_MODEL}_n_layer-${N_LAYER}_lr-${LR}_step-${GRADIENT_UPDATE_NUM}_moeloss-${OUTPUT_ROUTER_LOGITS}_1head_onlymoe"
HYDRA_RUN_DIR="${full_path_to_root}/outputs/pretrain/hg38/${WANDB_NAME}"
WATCH_DIR="${full_path_to_root}/watch_folder/pretrain"

export WANDBID=$(python -c "import wandb; print(wandb.util.generate_id())")

mkdir -p "${HYDRA_RUN_DIR}"
mkdir -p "${WATCH_DIR}"
srun python -m train \
  experiment=hg38/hg38 \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  dataset.max_length=${SEQLEN} \
  dataset.batch_size=$(( BATCH_SIZE / NUM_DEVICES )) \
  dataset.batch_size_eval=$(( BATCH_SIZE / NUM_DEVICES )) \
  dataset.mlm=False \
  dataset.rc_aug="${RC_AUG}" \
  dataset.add_eos=false \
  loader.num_workers=0 \
  model="janusdna" \
  +model.config.output_router_logits="${OUTPUT_ROUTER_LOGITS}" \
  +model.config.router_aux_loss_coef="${ROUTER_AUX_LOSS_COEF}" \
  model.config.bidirectional_weight_tie="${BIDIRECTIONAL_WEIGHT_TIE}" \
  model.config.bidirectional_attn_tie="${BIDIRECTIONAL_ATTN_TIE}" \
  model.config.num_hidden_layers=${N_LAYER} \
  model.config.hidden_size=${D_MODEL} \
  model.config.flex_attn_n_embd=${FLEX_ATTN_MODEL} \
  +model.config.intermediate_size=${INTER_FFN_MODEL} \
  model.config.expert_layer_period=2 \
  model.config.expert_layer_offset=1 \
  model.config.intermediate_factor=4 \
  model.config.num_attention_heads=4 \
  model.config.attn_implementation="flash_attention_2" \
  model.config.attn_layer_period=8 \
  model.config.attn_layer_offset=100 \
  optimizer.lr="${LR}" \
  train.global_batch_size=${BATCH_SIZE} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.precision=bf16-mixed \
  trainer.devices=${NUM_DEVICES} \
  +trainer.val_check_interval=$(( MAX_STEPS / 5 )) \
  +trainer.strategy="ddp_find_unused_parameters_true" \
  wandb.group=pretrain_hg38 \
  wandb.name="${WANDB_NAME}" \
  wandb.mode=online \
  wandb.id=${WANDBID} \
  hydra.run.dir="${HYDRA_RUN_DIR}" \
  > ${WATCH_DIR}/${WANDB_NAME}.log 2>&1
  



