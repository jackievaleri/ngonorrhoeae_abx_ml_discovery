# need to first install chemprop (I recommend option 2): https://chemprop.readthedocs.io/en/latest/installation.html

# activate venv
source activate chemprop

# navigate into the scripts folder to use features generator
# replace with the path to your chemprop/scripts folder
cd ../../37K_screens/models/chemprop/scripts

# save features for all the ~2300 drugs in screen
#for original model 11/15/2021
#python save_features.py --data_path ../../../../melis_gonorrhea/data/TRAIN_11_15_2021.csv --features_generator rdkit_2d_normalized --save_path ../../../../melis_gonorrhea/data/TRAIN_11_15_2021  --smiles_column SMILES
#python save_features.py --data_path ../../../../melis_gonorrhea/data/TEST_11_15_2021.csv --features_generator rdkit_2d_normalized --save_path ../../../../melis_gonorrhea/data/TEST_11_15_2021  --smiles_column SMILES
#python save_features.py --data_path ../../../../melis_gonorrhea/data/FULL_11_15_2021.csv --features_generator rdkit_2d_normalized --save_path ../../../../melis_gonorrhea/data/FULL_11_15_2021  --smiles_column SMILES

#for retraining model 01/09/2022
#python save_features.py --data_path ../../../../melis_gonorrhea/data/TRAIN_01_09_2022.csv --features_generator rdkit_2d_normalized --save_path ../../../../melis_gonorrhea/data/TRAIN_01_09_2022  --smiles_column SMILES
#python save_features.py --data_path ../../../../melis_gonorrhea/data/TEST_01_09_2022.csv --features_generator rdkit_2d_normalized --save_path ../../../../melis_gonorrhea/data/TEST_01_09_2022  --smiles_column SMILES
#python save_features.py --data_path ../../../../melis_gonorrhea/data/FULL_01_09_2022.csv --features_generator rdkit_2d_normalized --save_path ../../../../melis_gonorrhea/data/FULL_01_09_2022  --smiles_column SMILES

# for training models 03/18/2022
#python save_features.py --data_path ../../../../melis_gonorrhea/data/TRAIN_03_19_2022.csv --features_generator rdkit_2d_normalized --save_path ../../../../melis_gonorrhea/data/TRAIN_03_19_2022  --smiles_column SMILES
#python save_features.py --data_path ../../../../melis_gonorrhea/data/TEST_03_19_2022.csv --features_generator rdkit_2d_normalized --save_path ../../../../melis_gonorrhea/data/TEST_03_19_2022  --smiles_column SMILES
#python save_features.py --data_path ../../../../melis_gonorrhea/data/FULL_03_19_2022.csv --features_generator rdkit_2d_normalized --save_path ../../../../melis_gonorrhea/data/FULL_03_19_2022  --smiles_column SMILES

# for extended screen set 04/18/2022
#python save_features.py --data_path ../../../../melis_gonorrhea/data/cleaned_extended_screens/cleaned_full_all_dbs_04_19_2022.csv --features_generator rdkit_2d_normalized --save_path ../../../../melis_gonorrhea/data/cleaned_extended_screens/cleaned_full_all_dbs_04_19_2022  --smiles_column SMILES

# for training models 10/26/2022
# python save_features.py --data_path ../../../../melis_gonorrhea/data/TRAIN_10_26_2022.csv --features_generator rdkit_2d_normalized --save_path ../../../../melis_gonorrhea/data/TRAIN_10_26_2022  --smiles_column SMILES
# python save_features.py --data_path ../../../../melis_gonorrhea/data/TEST_10_26_2022.csv --features_generator rdkit_2d_normalized --save_path ../../../../melis_gonorrhea/data/TEST_10_26_2022  --smiles_column SMILES
# python save_features.py --data_path ../../../../melis_gonorrhea/data/FULL_10_26_2022.csv --features_generator rdkit_2d_normalized --save_path ../../../../melis_gonorrhea/data/FULL_10_26_2022  --smiles_column SMILES

# for training models 03/31/2023 - do on cloud
python save_features.py --data_path ../../../../ngonorrhoeae_abx_ml_discovery/data/data_prep_for_ml/data_prep_for_ml_pk_37k_three_rounds_val/TRAIN_03_31_2023.csv --features_generator rdkit_2d_normalized --save_path ../../../../ngonorrhoeae_abx_ml_discovery/data/data_prep_for_ml/data_prep_for_ml_pk_37k_three_rounds_val/TRAIN_03_31_2023  --smiles_column SMILES
python save_features.py --data_path ../../../../ngonorrhoeae_abx_ml_discovery/data/data_prep_for_ml/data_prep_for_ml_pk_37k_three_rounds_val/TEST_03_31_2023.csv --features_generator rdkit_2d_normalized --save_path ../../../../ngonorrhoeae_abx_ml_discovery/data/data_prep_for_ml/data_prep_for_ml_pk_37k_three_rounds_val/TEST_03_31_2023  --smiles_column SMILES
python save_features.py --data_path ../../../../ngonorrhoeae_abx_ml_discovery/data/data_prep_for_ml/data_prep_for_ml_pk_37k_three_rounds_val/FULL_03_31_2023.csv --features_generator rdkit_2d_normalized --save_path ../../../../ngonorrhoeae_abx_ml_discovery/data/data_prep_for_ml/data_prep_for_ml_pk_37k_three_rounds_val/FULL_03_31_2023  --smiles_column SMILES
