# Bayesian Hyperparameter Optimization: D-MPNN GNN
#
# Overall Goal
# - Run chemprop's Bayesian hyperparameter optimization (chemprop_hyperopt) for the
#   D-MPNN GNN on each successive round of training data.
#
# Relation to Manuscript
# - Selects the D-MPNN architectures used in Figure 2D, Figure 4A, and Figure 5A;
#   selected configurations are reported in Table S2.
#
# Rounds
# - Round 1: PK + 37K screen (TRAIN_03_19_2022); 10 search iterations, ensemble_size 4
# - Round 2: PK + 37K + 1st-round validation (TRAIN_10_26_2022); 20 iterations, ensemble_size 4
# - Round 3: PK + 37K + 3-round validation (TRAIN_03_31_2023); 20 iterations, ensemble_size 2
#       - Note: Not used in the final paper but provided for interested readers.
# 
# Run Configuration
# - Tool: chemprop_hyperopt
# - Split: scaffold_balanced 80 / 10 / 10, 5-fold CV
# - Primary metric: prc-auc; extra: auc
# - GPU 0
#
# Notes
# - Best configurations from each round are subsequently retrained in
#   2D_train_final_models_all_data.sh.

conda activate chemprop

# PK + 37K model - 03/19/2022
EXPORT DATA_PATH=../data/data_prep_for_ml/data_prep_for_ml_pk_37k_screen/
EXPORT MODEL_PATH=models/pk_37k_screen_models_03192022/
chemprop_hyperopt --save_dir $MODEL_PATH --dropout 0.35 --hidden_size 1600 --ffn_num_layers 1 --depth 5 --data_path "$DATA_PATH"TRAIN_03_19_2022.csv --dataset_type classification --features_path "$DATA_PATH"TRAIN_03_19_2022.npz --no_features_scaling --num_folds 5 --ensemble_size 4 --split_sizes 0.8 0.1 0.1 --smiles_columns SMILES --target_columns hit --gpu 0 --split_type scaffold_balanced --metric prc-auc --extra_metrics auc --num_iters 10 --config_save_path $MODEL_PATH

# PK + 37K + 1st round validation model - 10/26/2022
EXPORT DATA_PATH=../data/data_prep_for_ml/data_prep_for_ml_pk_37k_first_round_val_screen/
EXPORT MODEL_PATH=models/pk_37k_first_round_val_screen_models_10262022/
chemprop_hyperopt --save_dir $MODEL_PATH --dropout 0.15 --hidden_size 2300 --ffn_num_layers 3 --depth 3 --data_path "$DATA_PATH"TRAIN_10_26_2022.csv --dataset_type classification --features_path "$DATA_PATH"TRAIN_10_26_2022.npz --no_features_scaling --num_folds 5 --ensemble_size 4 --split_sizes 0.8 0.1 0.1 --smiles_columns SMILES --target_columns hit --gpu 0 --split_type scaffold_balanced --metric prc-auc --extra_metrics auc --num_iters 20 --config_save_path $MODEL_PATH

# PK + 37K + 3 rounds validation model - 03/31/2023
# NOTE: Round 3 GNN — not used in the final manuscript, left for interested readers.
EXPORT DATA_PATH=../data/data_prep_for_ml/data_prep_for_ml_pk_37k_three_rounds_val/
EXPORT MODEL_PATH=models/pk_37k_three_rounds_val_models_03312023/
chemprop_hyperopt --save_dir $MODEL_PATH --data_path "$DATA_PATH"TRAIN_03_31_2023.csv --dataset_type classification --features_path "$DATA_PATH"TRAIN_03_31_2023.npz --no_features_scaling --num_folds 5 --ensemble_size 2 --split_sizes 0.8 0.1 0.1 --smiles_columns SMILES --target_columns hit --gpu 0 --split_type scaffold_balanced --metric prc-auc --extra_metrics auc --num_iters 20 --config_save_path $MODEL_PATH
