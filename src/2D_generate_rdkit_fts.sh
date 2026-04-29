# Generate RDKit 2D Normalized Features
#
# Overall Goal
# - Precompute RDKit 2D normalized molecular descriptors as .npz feature files for
#   each TRAIN / TEST / FULL split, across all four screen rounds.
#
# Relation to Manuscript
# - Prerequisite step for all D-MPNN, RFC, SVM, and FFN training and hyperparameter
#   optimization scripts that pass --features_path; supports models reported in
#   Figure 2D and Figures 4A / 5A.
#
# Datasets Featurized
# - Dataset 0: PK screen (11_15_2021)  -- *feeds the Round 0 PK-only GNN, which is not used
#   in the final manuscript; left for interested readers*
# - Dataset 1: PK + 37K screen (03_19_2022)
# - Dataset 2: PK + 37K + 1st-round validation (10_26_2022)
# - Dataset 3: PK + 37K + 3-round validation (03_31_2023)  -- *feeds the Round 3 GNN, which is
#   not used in the final manuscript; left for interested readers*
#
# Tooling
# - chemprop scripts/save_features.py with --features_generator rdkit_2d_normalized.
#
# Prerequisites
# - chemprop installed (https://chemprop.readthedocs.io/en/latest/installation.html);
#   update the cd path to point to your local chemprop/scripts folder.

# need to first install chemprop (I recommend option 2): https://chemprop.readthedocs.io/en/latest/installation.html

# activate venv
source activate chemprop

# navigate into the scripts folder to use features generator
# replace with the path to your chemprop/scripts folder
cd ../../37K_screens/models/chemprop/scripts

# PK screen only
# NOTE: Features for the Round 0 (PK-only) GNN — not used in the final manuscript, left for interested readers.
EXPORT DATA_PATH=../../../../ngonorrhoeae_abx_ml_discovery/data/data_prep_for_ml/data_prep_for_ml_pk_screen/
python save_features.py --data_path "$DATA_PATH"TRAIN_11_15_2021.csv --features_generator rdkit_2d_normalized --save_path "$DATA_PATH"TRAIN_11_15_2021  --smiles_column SMILES
python save_features.py --data_path "$DATA_PATH"TEST_11_15_2021.csv --features_generator rdkit_2d_normalized --save_path "$DATA_PATH"TEST_11_15_2021  --smiles_column SMILES
python save_features.py --data_path "$DATA_PATH"FULL_11_15_2021.csv --features_generator rdkit_2d_normalized --save_path "$DATA_PATH"FULL_11_15_2021  --smiles_column SMILES

# PK + 37K screen
EXPORT DATA_PATH=../../../../ngonorrhoeae_abx_ml_discovery/data/data_prep_for_ml/data_prep_for_ml_pk_37k_screen/
python save_features.py --data_path "$DATA_PATH"TRAIN_03_19_2022.csv --features_generator rdkit_2d_normalized --save_path "$DATA_PATH"TRAIN_03_19_2022  --smiles_column SMILES
python save_features.py --data_path "$DATA_PATH"TEST_03_19_2022.csv --features_generator rdkit_2d_normalized --save_path "$DATA_PATH"TEST_03_19_2022  --smiles_column SMILES
python save_features.py --data_path "$DATA_PATH"FULL_03_19_2022.csv --features_generator rdkit_2d_normalized --save_path "$DATA_PATH"FULL_03_19_2022  --smiles_column SMILES

# PK + 37K screen + 1st round validation
EXPORT DATA_PATH=../../../../ngonorrhoeae_abx_ml_discovery/data/data_prep_for_ml/data_prep_for_ml_pk_37k_first_round_val_screen/
python save_features.py --data_path "$DATA_PATH"TRAIN_10_26_2022.csv --features_generator rdkit_2d_normalized --save_path "$DATA_PATH"TRAIN_10_26_2022  --smiles_column SMILES
python save_features.py --data_path "$DATA_PATH"TEST_10_26_2022.csv --features_generator rdkit_2d_normalized --save_path "$DATA_PATH"TEST_10_26_2022  --smiles_column SMILES
python save_features.py --data_path "$DATA_PATH"FULL_10_26_2022.csv --features_generator rdkit_2d_normalized --save_path "$DATA_PATH"FULL_10_26_2022  --smiles_column SMILES

# PK + 37K screen + 3 rounds validation
# NOTE: Features for the Round 3 GNN — not used in the final manuscript, left for interested readers.
EXPORT DATA_PATH=../../../../ngonorrhoeae_abx_ml_discovery/data/data_prep_for_ml/data_prep_for_ml_pk_37k_three_rounds_val/
python save_features.py --data_path "$DATA_PATH"TRAIN_03_31_2023.csv --features_generator rdkit_2d_normalized --save_path "$DATA_PATH"TRAIN_03_31_2023  --smiles_column SMILES
python save_features.py --data_path "$DATA_PATH"TEST_03_31_2023.csv --features_generator rdkit_2d_normalized --save_path "$DATA_PATH"TEST_03_31_2023  --smiles_column SMILES
python save_features.py --data_path "$DATA_PATH"FULL_03_31_2023.csv --features_generator rdkit_2d_normalized --save_path "$DATA_PATH"FULL_03_31_2023  --smiles_column SMILES
