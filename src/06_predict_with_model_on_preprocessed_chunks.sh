# for enamine frags
export DATA_PATH=../ngonorrhoeae_abx_ml_discovery/data/library_info/temp_dir_enamine_cleaned_chunks/
export MODEL_PATH=../ngonorrhoeae_abx_ml_discovery/models/pk_37k_three_rounds_val_models_03312023/FINALbayHO04112023/

cd ../chemprop/

for ((i=0; i<=1833; i=i+1));
do

    cd scripts/
    
    python save_features.py --data_path ../"$DATA_PATH"$i.csv --features_generator rdkit_2d_normalized --save_path ../"$DATA_PATH"$i  --smiles_column smiles
    
    cd ../
    
    python predict.py --test_path "$DATA_PATH"$i.csv --features_path "$DATA_PATH"$i.npz --no_features_scaling --checkpoint_dir $MODEL_PATH --preds_path "$DATA_PATH"$i.csv --smiles_column smiles --ensemble_variance --gpu 0
    
    rm "$DATA_PATH"$i.npz
    echo $i
done
