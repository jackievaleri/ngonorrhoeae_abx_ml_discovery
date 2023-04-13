# need to first install chemprop (I recommend option 2): https://chemprop.readthedocs.io/en/latest/installation.html

# activate venv
source activate chemprop

# navigate into the scripts folder to use features generator
# replace with the path to your chemprop/scripts folder
cd ../../37K_screens/models/chemprop/scripts

# PK screen only
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
EXPORT DATA_PATH=../../../../ngonorrhoeae_abx_ml_discovery/data/data_prep_for_ml/data_prep_for_ml_pk_37k_three_rounds_val/
python save_features.py --data_path "$DATA_PATH"TRAIN_03_31_2023.csv --features_generator rdkit_2d_normalized --save_path "$DATA_PATH"TRAIN_03_31_2023  --smiles_column SMILES
python save_features.py --data_path "$DATA_PATH"TEST_03_31_2023.csv --features_generator rdkit_2d_normalized --save_path "$DATA_PATH"TEST_03_31_2023  --smiles_column SMILES
python save_features.py --data_path "$DATA_PATH"FULL_03_31_2023.csv --features_generator rdkit_2d_normalized --save_path "$DATA_PATH"FULL_03_31_2023  --smiles_column SMILES
