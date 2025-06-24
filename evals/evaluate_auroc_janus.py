import os
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# automatically go through all the output for all cell types, and 

def fprint(*args, **kwargs):
    """Print to file and stdout"""
    print(*args, **kwargs)
    conclude_output_file.write(" ".join(map(str, args)) + "\n")
    conclude_output_file.flush()


CELL_TYPES=[
    "Adipose_Subcutaneous",
    "Artery_Tibial",
    "Cells_Cultured_fibroblasts",
    "Muscle_Skeletal",
    "Nerve_Tibial",
    "Skin_Not_Sun_Exposed_Suprapubic",
    "Skin_Sun_Exposed_Lower_leg",
    "Thyroid",
    "Whole_Blood",
]

WATCH_FOLDER_BASE_PATH = "TO THE LOG FOLDER PATH"
MODEL_NAME = "janusdna_len-131k_d_model-128_inter_dim-512_n_layer-8_lr-8e-3_step-50K_moeloss-true_1head_onlymoe"
END_SUFFIX = "lr-5e-5_ftepoch-3_cjtrain_false_cjtest_true_batch_2_withpretrainedweight_bf16mix_output.log" 


CONCLUDE_OUTPUT_FILE_DIR_PATH = os.path.join(WATCH_FOLDER_BASE_PATH, MODEL_NAME)
conclude_output_file_path = os.path.join(CONCLUDE_OUTPUT_FILE_DIR_PATH, END_SUFFIX)
print("conclude_output_file_path", conclude_output_file_path)

conclude_output_file = open(conclude_output_file_path, "w", encoding="utf-8")

# print("CONCLUDE_FILE_PATH", os.path.join(CONCLUDE_OUTPUT_FILE_DIR_PATH, END_SUFFIX))
fprint("original_watch_folder: ", os.path.join(WATCH_FOLDER_BASE_PATH, MODEL_NAME))
fprint("MODEL_NAME: ", MODEL_NAME)
fprint("end_suffix: ", END_SUFFIX)

fprint("\n")
fprint("metric_value", "cell_type", "average_precision_score", "metric_type")
for cell_type in CELL_TYPES:
    file_path = os.path.join(WATCH_FOLDER_BASE_PATH, MODEL_NAME, cell_type, cell_type + "_" + END_SUFFIX) 
    if not os.path.exists(file_path):
        fprint(f"File not found: {file_path}")
        continue
    preds, targets = [], []
    lines = open(file_path).readlines()
    for idx, line in enumerate(lines):
        items = line.strip().split()
        preds.append(float(items[0]))
        targets.append(float(items[1]))

    # fprint()
    
    fprint(roc_auc_score(targets, preds), cell_type,  ": ", average_precision_score(targets, preds), "auroc")

