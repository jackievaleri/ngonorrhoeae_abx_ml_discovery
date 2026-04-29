# Train Baseline Models on Full Data with Random (Non-Scaffold) Split
#
# Overall Goal
# - Retrain the best RFC, SVM, and FFN configurations on the full PK + 37K dataset
#   using a random 80 / 10 / 10 split, in parallel to the scaffold-balanced finals
#   in 2D_train_final_models_all_data.sh.
#
# Relation to Manuscript
# - Provides random-split counterparts to the scaffold-balanced baseline models
#   in Figure 2D.
#
# Configurations Trained
# - RFC rfc_RANDOM_final_122: num_bits 4096, radius 2, num_trees 250, class_weight balanced
# - SVM svm_RANDOM_final_15: num_bits 4096, radius 2, class_weight balanced
# - FFN ffn_RANDOM_final_20_1: hidden_size 500, ffn_num_layers 3, dropout 0.4 (Morgan FPs, seed 4)
#
# Run Configuration
# - Data: data_prep_for_ml_pk_37k_screen/FULL_03_19_2022.csv (+ matching .npz)
# - Split: random, 80 / 10 / 10
# - Folds: RFC 30, SVM 30, FFN 6
# - Primary metric: prc-auc; extra: auc
#
# Prerequisites
# - Run 2D_generate_rdkit_fts.sh first to produce the .npz feature files.

cd ../chemprop-master/

export DATA_PATH=../data/data_prep_for_ml/data_prep_for_ml_pk_37k_screen/

# RFC on fingerprints
export MODEL_PATH=../models/other_models/rfc_hyperopt_pk_37k/
export MODEL_NAME=rfc_RANDOM_final_122
mkdir "$MODEL_PATH""$MODEL_NAME"; python sklearn_train.py --num_bits 4096 --radius 2 --class_weight balanced --num_trees 250 --save_dir "$MODEL_PATH""$MODEL_NAME" --data_path "$DATA_PATH"FULL_03_19_2022.csv --num_folds 30 --dataset_type classification --features_path "$DATA_PATH"FULL_03_19_2022.npz  --no_features_scaling --split_type random --split_sizes 0.8 0.1 0.1 --smiles_columns SMILES --target_columns hit --model_type random_forest --metric prc-auc --extra_metrics auc

# SVM on fingerprints
export MODEL_PATH=../models/other_models/svm_hyperopt_pk_37k/
export MODEL_NAME=svm_RANDOM_final_15
mkdir "$MODEL_PATH""$MODEL_NAME"; python sklearn_train.py --num_bits 4096 --radius 2 --class_weight balanced --save_dir "$MODEL_PATH""$MODEL_NAME" --data_path "$DATA_PATH"FULL_03_19_2022.csv --num_folds 30 --dataset_type classification --features_path "$DATA_PATH"FULL_03_19_2022.npz  --no_features_scaling --split_type random --split_sizes 0.8 0.1 0.1 --smiles_columns SMILES --target_columns hit --model_type svm --metric prc-auc --extra_metrics auc

# FFN on fingerprints
export MODEL_PATH=../models/other_models/ffn_hyperopt_pk_37k/
export MODEL_NAME=ffn_RANDOM_final_20_1
mkdir "$MODEL_PATH""$MODEL_NAME"; chemprop_train  --hidden_size 500 --ffn_num_layers 3 --dropout 0.4 --save_dir "$MODEL_PATH""$MODEL_NAME" --data_path "$DATA_PATH"FULL_03_19_2022.csv --num_folds 6 --dataset_type classification --features_generator morgan --no_features_scaling --split_type random --split_sizes 0.8 0.1 0.1 --smiles_columns SMILES --target_columns hit --depth 0 --features_only --metric prc-auc --extra_metrics auc --seed 4
