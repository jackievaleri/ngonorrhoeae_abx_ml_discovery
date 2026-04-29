cd ../../chemprop/

EXPORT MODEL_PATH=../ngonorrhoeae_abx_ml_discovery/models/other_models/
EXPORT MODEL_NAME=rfc_final_122
EXPORT DATA_PATH=../ngonorrhoeae_abx_ml_discovery/data/data_prep_for_ml/data_prep_for_ml_pk_37k_screen/

# RFC on fingerprints
mkdir "$MODEL_PATH""$MODEL_NAME"; python sklearn_train.py --num_bits 4096 --radius 2 --class_weight balanced --num_trees 250 --save_dir "$MODEL_PATH""$MODEL_NAME" --data_path "$DATA_PATH"FULL_03_19_2022.csv --num_folds 30 --dataset_type classification --features_path "$DATA_PATH"FULL_03_19_2022.npz  --no_features_scaling --split_type scaffold_balanced --split_sizes 0.8 0.1 0.1 --smiles_columns SMILES --target_columns hit --model_type random_forest --metric prc-auc --extra_metrics auc

# SVM on fingerprints
EXPORT MODEL_NAME=svm_final_15
mkdir "$MODEL_PATH""$MODEL_NAME"; python sklearn_train.py --num_bits 4096 --radius 2 --class_weight balanced --save_dir "$MODEL_PATH""$MODEL_NAME" --data_path "$DATA_PATH"FULL_03_19_2022.csv --num_folds 30 --dataset_type classification --features_path "$DATA_PATH"FULL_03_19_2022.npz  --no_features_scaling --split_type scaffold_balanced --split_sizes 0.8 0.1 0.1 --smiles_columns SMILES --target_columns hit --model_type svm --metric prc-auc --extra_metrics auc

# FFN on fingerprints
EXPORT MODEL_NAME=ffn_final_20
mkdir "$MODEL_PATH""$MODEL_NAME"; python train.py  --hidden_size 500 --ffn_num_layers 3 --dropout 0.4 --save_dir "$MODEL_PATH""$MODEL_NAME" --data_path "$DATA_PATH"FULL_03_19_2022.csv --num_folds 30 --dataset_type classification --features_generator morgan --no_features_scaling --split_type scaffold_balanced --split_sizes 0.8 0.1 0.1 --smiles_columns SMILES --target_columns hit --depth 0 --features_only --metric prc-auc --extra_metrics auc

# GNN on PK screen alone
EXPORT MODEL_PATH=../ngonorrhoeae_abx_ml_discovery/pk_screen_models_11152021/
EXPORT MODEL_NAME=FINAL151
EXPORT DATA_PATH=../ngonorrhoeae_abx_ml_discovery/data/data_prep_for_ml/data_prep_for_ml_pk_screen/
mkdir "$MODEL_PATH""$MODEL_NAME"; chemprop_train --init_lr 0.001 --dropout 0.3 --hidden_size 1200 --ffn_num_layers 3 --depth 4 --save_dir "$MODEL_PATH""$MODEL_NAME" --data_path "$DATA_PATH"FULL_11_15_2021.csv --dataset_type classification --features_path "$DATA_PATH"FULL_11_15_2021.npz --no_features_scaling --num_folds 50 --ensemble_size 1 --split_type scaffold_balanced --split_sizes 0.8 0.1 0.1 --smiles_columns SMILES --target_columns hit --metric prc-auc --extra_metrics auc

# GNN on PK+37K screen
EXPORT MODEL_PATH=../ngonorrhoeae_abx_ml_discovery/models/pk_37k_screen_models_03192022/
EXPORT MODEL_NAME=FINALbayHO04052022
EXPORT DATA_PATH=../ngonorrhoeae_abx_ml_discovery/data/data_prep_for_ml/data_prep_for_ml_pk_37k_screen/
mkdir "$MODEL_PATH""$MODEL_NAME"; chemprop_train --dropout 0.15 --hidden_size 2300 --ffn_num_layers 3 --depth 3 --metric prc-auc --extra_metrics auc --save_dir "$MODEL_PATH""$MODEL_NAME" --data_path "$DATA_PATH"FULL_03_19_2022.csv --dataset_type classification --features_path "$DATA_PATH"FULL_03_19_2022.npz --no_features_scaling --num_folds 5 --ensemble_size 10 --split_sizes 0.8 0.1 0.1 --smiles_columns SMILES --target_columns hit --gpu 0

EXPORT MODEL_NAME=FINALbayHO04052022_with_scaffold_split
mkdir "$MODEL_PATH""$MODEL_NAME"; chemprop_train --dropout 0.15 --hidden_size 2300 --ffn_num_layers 3 --depth 3 --metric prc-auc --extra_metrics auc --save_dir "$MODEL_PATH""$MODEL_NAME" --data_path "$DATA_PATH"FULL_03_19_2022.csv --dataset_type classification --features_path "$DATA_PATH"FULL_03_19_2022.npz --no_features_scaling --num_folds 5 --ensemble_size 10 --split_type scaffold_balanced --split_sizes 0.8 0.1 0.1 --smiles_columns SMILES --target_columns hit --gpu 0

# GNN on PK+37K screen + 1st round validation
EXPORT MODEL_PATH=../ngonorrhoeae_abx_ml_discovery/models/pk_37k_first_round_val_screen_models_10262022/
EXPORT MODEL_NAME=FINALbayHO11152022
EXPORT DATA_PATH=../ngonorrhoeae_abx_ml_discovery/data/data_prep_for_ml/data_prep_for_ml_pk_37k_first_round_val_screen/
mkdir "$MODEL_PATH""$MODEL_NAME"; chemprop_train --dropout 0.25 --hidden_size 800 --ffn_num_layers 2 --depth 5 --metric prc-auc --extra_metrics auc --save_dir "$MODEL_PATH""$MODEL_NAME" --data_path "$DATA_PATH"FULL_10_26_2022.csv --dataset_type classification --features_path "$DATA_PATH"FULL_10_26_2022.npz --no_features_scaling --num_folds 5 --ensemble_size 10 --split_type scaffold_balanced --split_sizes 0.8 0.1 0.1 --smiles_columns SMILES --target_columns hit --gpu 0

# GNN on PK+37K screen + 3 rounds validation
EXPORT MODEL_PATH=../ngonorrhoeae_abx_ml_discovery/models/pk_37k_three_rounds_val_models_03312023/
EXPORT MODEL_NAME=FINALbayHO04112023
EXPORT DATA_PATH=../ngonorrhoeae_abx_ml_discovery/data/data_prep_for_ml/data_prep_for_ml_pk_37k_three_rounds_val/
mkdir "$MODEL_PATH""$MODEL_NAME"; chemprop_train --dropout 0.25 --hidden_size 400 --ffn_num_layers 3 --depth 6 --metric prc-auc --extra_metrics auc --save_dir "$MODEL_PATH""$MODEL_NAME" --data_path "$DATA_PATH"FULL_03_31_2023.csv --dataset_type classification --features_path "$DATA_PATH"FULL_03_31_2023.npz --no_features_scaling --num_folds 5 --ensemble_size 10 --split_type scaffold_balanced --split_sizes 0.8 0.1 0.1 --smiles_columns SMILES --target_columns hit --gpu 0
