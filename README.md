# N. gonorrhoeae Antibiotics Discovery
Discovery of antibiotics active against N. gonorrhoeae using experimental and machine learning screening

# DONE
* 01 - biochem comparison of PK hits + non-hits; biochem comparison of 37K+PK hits + non-hits; biochem comparison of model incorrect predictions
* 02 - tSNE of 37K, abx, PK; tSNE of hits, nonhits, abx

# IN PROGRESS
* 03: data prep for ml: pk, pk+37k, pk+37k+val

# TODO
* model training - GNN for PK, GNN for pk+37K
* shallow model training - RFC, SVM, FFN on pk+37k
* other neural model training - attention model, new GNNs on pk+37K
* using models for prediction
* hyperopt of all models
* curating predictions for 800k, molport
* substruct interpret of nitro groups
