import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MCS, Descriptors, PandasTools, AllChem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.ML.Cluster import Butina
from sklearn.cluster import AgglomerativeClustering
from adme_pred import ADME # pip install ADME_predict is NOT the right one

# SECTION 1: MOLECULAR FILTERS
def filter_for_logp_less_than(df, mols, thresh = 5):
    keep_indices = [Descriptors.MolLogP(mol) < thresh for mol in mols]
    df = df[keep_indices]
    print('length of df with logP < ' + str(thresh) + ': ', len(df))
    mols = [m for i,m in enumerate(mols) if keep_indices[i]]
    return(df, mols)

def check_pains_brenk(df, mols, method = 'both', thresh = 0):
    # initialize filter
    params = FilterCatalogParams()
    if method == 'both' or method == 'pains':
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    if method == 'both' or method == 'brenk':
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog(params)

    def search_for_pains_or_brenk(mol):
        entry = catalog.GetMatches(mol)  # Get all matching PAINS or Brenk
        if entry is None:
            return(True)
        else:
            return(len(entry) <= thresh)

    keep_indices = [search_for_pains_or_brenk(m) for m in mols]
    df = df[keep_indices]
    mols = [m for i,m in enumerate(mols) if keep_indices[i]]
    print('length of all preds with clean (no PAINS or Brenk) mols: ', len(df))
    return(df, mols)

nitrofuran = 'O=[N+](O)c1ccco1'
nitro_mol = Chem.MolFromSmiles(nitrofuran)

sulfonamide = 'NS(=O)=O'
sulfa_mol = Chem.MolFromSmiles(sulfonamide)

quinolone = 'O=c1cc[nH]c2ccccc12'
quino_mol = Chem.MolFromSmiles(quinolone)

def is_pattern(mol, pattern_mol):
    if mol is None:
        return(False)
    num_atoms_frag = pattern_mol.GetNumAtoms()
    mcs = MCS.FindMCS([mol, pattern_mol], atomCompare='elements',completeRingsOnly = True)
    if mcs.smarts is None:
        return(False)
    mcs_mol = Chem.MolFromSmarts(mcs.smarts)
    num_atoms_mcs = mcs_mol.GetNumAtoms()
    if num_atoms_frag == num_atoms_mcs:
        return(True)
    else:
        return(False)

def filter_for_nitrofuran(df, mols):
    keep_indices = [not is_pattern(mol, nitro_mol) for mol in mols]
    df = df[keep_indices]
    print('length of df with no nitrofurans: ', len(df))
    mols = [m for i,m in enumerate(mols) if keep_indices[i]]
    return(df, mols)   
    
def filter_for_sulfonamide(df, mols):
    keep_indices = [not is_pattern(mol, sulfa_mol) for mol in mols]
    df = df[keep_indices]
    print('length of df with no sulfonamides: ', len(df))
    mols = [m for i,m in enumerate(mols) if keep_indices[i]]
    return(df, mols)

def filter_for_quinolone(df, mols):
    keep_indices = [not is_pattern(mol, quino_mol) for mol in mols]
    df = df[keep_indices]
    print('length of df with no quinolones: ', len(df))
    mols = [m for i,m in enumerate(mols) if keep_indices[i]]
    return(df, mols)  

def filter_for_mw_bounds(df, mols, lower_bound = 100, upper_bound = 600):
    keep_indices = [Descriptors.MolWt(mol) <= upper_bound and Descriptors.MolWt(mol) >= lower_bound for mol in mols]
    df = df[keep_indices]
    print('length of df with ' + str(lower_bound) + ' < MW < ' + str(upper_bound) + ': ', len(df))
    mols = [m for i,m in enumerate(mols) if keep_indices[i]]
    return(df, mols)

def filter_for_rotatable_bonds(df, mols, smiles_column, thresh = 5):
    smis = list(df[smiles_column])
    bonds = [ADME(smi)._n_rot_bonds() for smi in smis]
    keep_indices = [bond < thresh for bond in bonds]
    df = df[keep_indices]
    print('length of df with num rotatable bonds < ' + str(thresh) + ': ', len(df))
    mols = [m for i,m in enumerate(mols) if keep_indices[i]]
    return(df, mols)

def keep_valid_molecules(df, smiles_column):
    smis = list(df[smiles_column])
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    keep_indices = [m is not None for m in mols]
    df = df[keep_indices]
    print('length of df with valid mols: ', len(df))
    mols = [m for i,m in enumerate(mols) if keep_indices[i]]
    smis = list(df[smiles_column])
    for m, smi in zip(mols, smis):
        m.SetProp('SMILES', smi)
    return(df, mols)

def deduplicate_on_tan_sim(df, mols):
    # deduplicate based on tanimoto similarity
    fps = [Chem.RDKFingerprint(m) for m in mols]
    keep_indices = [True] * len(fps)
    for i, f1 in enumerate(fps):
        tans = DataStructs.BulkTanimotoSimilarity(f1, fps)
        for j, tan in enumerate(tans):
            if tan < 1.0 or i == j:
                continue
            else:
                if keep_indices[i]:
                    keep_indices[j] = False

    df = df[keep_indices]
    print('length of all preds deduplicated: ', len(df))
    return(df)

# SECTION 2: PROCESSING FUNCTIONS
def rank_order_preds(smis, scos, path):
    new_scos = []
    new_smis = []
    for smi,sco in zip(smis,scos):
        try: 
            new_scos.append(float(sco))
            new_smis.append(smi)
        except:
            continue
    df = pd.DataFrame(zip(new_smis,new_scos), columns = ['smiles','scores'])
    df = df.sort_values(by = 'scores', ascending = False)
    df['rank'] = range(0,len(df))
    
    fig, ax = plt.subplots(figsize = (3,2), dpi = 300)
    plt.plot(df['rank'], df['scores'], color = 'slategrey')
    plt.xlabel('Rank')
    plt.ylabel('Score')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(path + 'score_wrt_rank.svg')
    plt.show()
    
    return(df)

# for every molecule, get similarity to closest drug in a set of drug fps
# this is simpler than the later tan sim code because there is no need to save the specific drug name/SMILES here
def get_closest_tanimoto_from_drug_set(new_set, drug_fps):
    # new_set is list of smiles
    # drug_fps is list of fingerprints of things you want to compare to
    try:
        mols = [Chem.MolFromSmiles(x) for x in new_set]
        # more info on RDK Fingerprints (similar to Morgan fingerprints) here - https://stackoverflow.com/questions/67811388/how-can-i-interpret-the-features-obtained-from-chem-rdkfingerprintmol
        fps = [Chem.RDKFingerprint(x) if x is not None else '' for x in mols]
    except Exception as e:
        print(e)
        return([1000])

    best_similarity = []
   
    i = 0
    for mol in fps: # corresponds to the 'new_set' molecules
        curr_highest_sim = 0
        if mol == '':
            best_similarity.append('NaN')
            continue
        j = 0
        for drug in drug_fps: # corresponds to the list of things you want to compare to
            try: 
                sim = DataStructs.FingerprintSimilarity(mol,drug)
            except Exception as e:
                continue
            if sim > curr_highest_sim and i !=j: # make sure they are not the same molecule
                curr_highest_sim = sim
            j = j + 1
        best_similarity.append(curr_highest_sim)
        i = i + 1 
    return(best_similarity)

def compute_highest_tan_sim_intraset(smis):
    smis = [smi for smi in smis if type(smi) != float]
    mols = [Chem.MolFromSmiles(x) for x in smis]
    fps = [Chem.RDKFingerprint(x) if x is not None else '' for x in mols ]
    tans = get_closest_tanimoto_from_drug_set(smis, fps)
    return(tans)

def plot_tan_wrt_rank(df, path, score_col = 'scores', tan_col = 'tan'):
    df = df.sort_values(by = score_col, ascending = False)
    
    fig, ax = plt.subplots(figsize = (3,2), dpi = 300)
    plt.scatter(df[score_col], df[tan_col], s = 0.5, color = 'slategrey')
    plt.xlabel('Rank')
    plt.ylabel('Highest Inter-Set Tanimoto')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(path + 'tan_wrt_rank.svg')
    plt.show()
    return

def plot_tan_wrt_score(df, path, score_col = 'scores', tan_col = 'tan'):
    fig, ax = plt.subplots(figsize = (3,2), dpi = 300)
    plt.scatter(df[score_col], df[tan_col], s = 0.5, color = 'slategrey')
    plt.xlabel('Score')
    plt.ylabel('Highest Inter-Set Tanimoto')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(path + 'tan_wrt_score.svg')
    plt.show()

def process_preds(df, smiles_col, hit_col, path):
    smis = list(df[smiles_col])
    scos = list(df[hit_col])

    # get ranks
    ranks = rank_order_preds(smis, scos, path)

    # get tanimotos
    ranks = ranks.iloc[0:10000,:] # use just the first 10K molecules by score to do tan analysis - else it takes forever
    tans = compute_highest_tan_sim_intraset(ranks['smiles']) # standardized to lowercase so can do this
    ranks['tan'] = tans

    # additional plots
    plot_tan_wrt_rank(ranks, path)
    plot_tan_wrt_score(ranks, path)

def evaluate_scores(df, hit_col):
    df[hit_col] = [float(x) for x in df[hit_col]]
    print('Total number in df: ', len(df))

    for i in np.arange(0.1, 1, 0.1):
        print('>' + str(i) + ': ' + str(len(df[df[hit_col] > i])))
    return(df)

# some hardcoded stuff in here - not ideal
def print_drug_dict(drugdict, smiles_list, col_list, names_list, merging_df, left_smiles = 'SMILES', right_smiles = 'SMILES'):

    ret = pd.DataFrame(columns = col_list)
    for key, item in drugdict.items():
        tans, indexes = item
        for tan, index in zip(tans, indexes):
            if names_list == None:
                ret = pd.concat([ret, pd.DataFrame([[key, tan, smiles_list[index], index]], columns = col_list)])
            else:
                ret = pd.concat([ret, pd.DataFrame([[key, tan, smiles_list[index], names_list[index]]], columns = col_list)])

    ret = ret.merge(merging_df, left_on = left_smiles, right_on = right_smiles)
    return(ret)

# for every molecule, get similarity to closest antibiotic
def get_lowest_tanimoto_from_drug_set(new_set, abx_fps, smiles_list, col_list, merging_df, names_list = None):
    mols = [Chem.MolFromSmiles(x) for x in new_set]
    
    best_similarity = {}
    index = 0
   
    for m in mols:
        try:
            mol = Chem.RDKFingerprint(m)
        except Exception as e:
            index = index + 1
            continue
        set_index = 0
        curr_highest_sim = 0
        curr_highest_drug = None
        for abx in abx_fps:
            sim = DataStructs.FingerprintSimilarity(mol,abx) # default metric is tanimoto similarity
            if sim > curr_highest_sim:
                curr_highest_sim = sim
                curr_highest_drug = set_index
            set_index = set_index + 1
        info_to_include = [[curr_highest_sim], [curr_highest_drug]]
        best_similarity[new_set[index]] = info_to_include
        
        index = index + 1
    # kind of convoluted b/c I like seeing the actual pandas df and it's helpful to keep track of names, smiles, and values
    ret = print_drug_dict(best_similarity, smiles_list, col_list, names_list, merging_df)
    return(ret)

# unfortunately sometimes SMILES can be NaN, or create invalid molecules, or fingerprints cannot be created from mols - 
def clean_up_names_and_smiles(df, name_col = 'Name', smiles_col = 'SMILES'):
    not_nans = [type(smi) != float for smi in list(df[smiles_col])]
    df = df[not_nans]

    smiles = list(df[smiles_col])
    names = list(df[name_col])
    mols = [Chem.MolFromSmiles(x) for x in smiles]
    
    new_smis = []
    new_names = []
    new_mols = []
    for smi, na, mo in zip(smiles, names, mols):
        try: 
            fp = Chem.RDKFingerprint(mo)
            new_smis.append(smi)
            new_names.append(na)
            new_mols.append(fp)
        except Exception as e:
            print(e)
            continue
    return(new_mols, new_smis, new_names)

def compute_tanimoto_against_abx(smis, merging_df):
    df = pd.read_csv('../data/04052022_CLEANED_v5_antibiotics_across_many_classes.csv')
    clean_mols, clean_smis, clean_names = clean_up_names_and_smiles(df, name_col = 'Name', smiles_col = 'Smiles')
    col_list = ['SMILES', 'tanimoto similarity to closest abx', 'closest abx smiles', 'closest abx name']
    gen_mols = get_lowest_tanimoto_from_drug_set(smis, clean_mols, clean_smis, col_list, merging_df, clean_names)
    return(gen_mols)

def compute_tanimoto_against_training_set(smis, merging_df, train_set_path, hit_col = 'hit', just_hits = False):
    df = pd.read_csv(train_set_path)
    if just_hits:
        df = df[df[hit_col] == 1]
    clean_mols, clean_smis, clean_names = clean_up_names_and_smiles(df, name_col = 'Name', smiles_col = 'SMILES')
    col_list = ['SMILES', 'tanimoto similarity to closest train set', 'closest train set smiles', 'closest train set name']
    gen_mols = get_lowest_tanimoto_from_drug_set(smis, clean_mols, clean_smis, col_list, merging_df, clean_names)
    return(gen_mols)

def evaluate_similarities(df, smiles_col, path, train_set_path, thresh_name = '04', hit_col = 'hit', just_hits = False):

    def plot_tanimoto_score_plot(df, tanimoto_col, path, name):
        plt.figure(figsize = (5,3), dpi = 300)
        plt.hist(df[tanimoto_col], bins = 50, edgecolor = 'black', linewidth = 0.5)
        plt.xlabel(tanimoto_col)
        plt.ylabel("Number")
        plt.savefig(path + name + '.png')
        plt.show()
    
    print("Computing tanimoto scores against abx...")
    smis = list(df[smiles_col])
    df = compute_tanimoto_against_abx(smis, df)
    plot_tanimoto_score_plot(df, 'tanimoto similarity to closest abx', path, 'inh' + thresh_name + '_tan_to_closest_abx')
    
    print("Computing tanimoto scores against training set...")
    df = compute_tanimoto_against_training_set(smis, df, train_set_path = train_set_path, hit_col = hit_col, just_hits = just_hits)
    plot_tanimoto_score_plot(df, 'tanimoto similarity to closest train set', path, 'inh' + thresh_name + '_tan_to_closest_train_set')
    return(df)

# SECTION 3: Clustering

# code adapted from https://www.macinchem.org/reviews/clustering/clustering.php
def clusterFps(fps,num_clusters):

    tan_array = [DataStructs.BulkTanimotoSimilarity(i, fps) for i in fps]
    tan_array = np.array(tan_array)
    clusterer= AgglomerativeClustering(n_clusters = num_clusters, compute_full_tree = True).fit(tan_array)
    final_clusters = {}
    for ix, m in enumerate(clusterer.labels_):
        if m in final_clusters:
            curr_list = final_clusters[m]
            curr_list.append(ix)
            final_clusters[m] = curr_list
        else:
            final_clusters[m] = [ix]
    
    return clusterer.labels_, final_clusters

def make_legends(mols, smis, df, smiles_col, name_col, hit_col, tans):
    row_num = 0
    legends = []
    for smi in smis:
        try:
            row = df.iloc[row_num,:]
            try:
                actualrowname = str(row.loc[name_col])
            except:
                actualrowname = str(row.loc[smiles_col])
            if len(actualrowname) > 20:
                actualrowname = actualrowname[0:20] + '...'
            hit = str(np.round(float(row.loc[hit_col]),3))
            actual_row_num = str(row.loc['row_num'])

            if tans:
                tan_train = str(np.round(float(row.loc['tanimoto similarity to closest train set']),3))
                tan_abx = str(np.round(float(row.loc['tanimoto similarity to closest abx']),3))
                legend = actualrowname + '\n' + 'tantrain: ' + tan_train + '\n' + 'tanabx: ' + tan_abx + '\n score: ' + hit
            else:
                legend = actualrowname + '\n' + '\n score: ' + hit

        except Exception as e:
            print(e)
            actual_row_num = str(row.loc['row_num'])
            legend = 'row: ' + actual_row_num

        mols[row_num].SetProp('legend', legend)
        row_num = row_num + 1
    return(mols) 

def extract_legends_and_plot(df, name, folder, num_clusters = 30, smiles_col = 'smiles', name_col = 'Name', hit_col = 'hit', tans = True):
    df['row_num'] = list(range(len(df)))
    df = df.drop_duplicates(subset=smiles_col)
    smis = list(df[smiles_col])
    
    mols = [Chem.MolFromSmiles(mol) for mol in smis]
    mols = make_legends(mols, smis, df, smiles_col = smiles_col, name_col = name_col, hit_col = hit_col, tans = tans)
    
    # code with help from the OG greg landrum: https://gist.github.com/greglandrum/d5f12058682f6b336905450e278d3399
    molsPerRow = 5
    subImgSize= (500,500)
    
    murcks = [MurckoScaffold.GetScaffoldForMol(mol) for mol in mols]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x,2,1024) for x in murcks]
    raw_cluster_labels, final_clusters=clusterFps(fps,num_clusters=num_clusters)
    
    #show clusters
    name_index = 0
    for cluster_key in final_clusters:
        cluster_mols = final_clusters[cluster_key]
        cluster_mols = [mols[i] for i in cluster_mols]

        nRows = len(cluster_mols) // molsPerRow
        if len(cluster_mols) % molsPerRow:
            nRows += 1
        fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
        d2d = rdMolDraw2D.MolDraw2DCairo(fullSize[0],fullSize[1], subImgSize[0], subImgSize[1])
        d2d.drawOptions().legendFontSize=100
        #d2d.drawOptions().useBWAtomPalette()
        d2d.DrawMolecules(cluster_mols,legends=[mol.GetProp('legend') for mol in cluster_mols])
        d2d.FinishDrawing()
        new_name = folder + str(name_index) + '_' + name
        open(new_name,'wb+').write(d2d.GetDrawingText())
        name_index = name_index + 1
    df['cluster'] = [str(i) for i in raw_cluster_labels] # so it gets interpreted by plotly in a good way
    return(df, mols)

def determine_optimal_clustering_number(df, max_num_clusters, smiles_col):

    df['row_num'] = list(range(len(df)))
    df = df.drop_duplicates(subset=smiles_col)
    smis = list(df[smiles_col])
    mols = [Chem.MolFromSmiles(mol) for mol in smis]
    murcks = [MurckoScaffold.GetScaffoldForMol(mol) for mol in mols]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x,2,1024) for x in murcks]
    
    max_dists = []
    avg_dists = []
    print(max_num_clusters)
    for number_of_clusters in range(1, max_num_clusters):
        raw_cluster_labels, final_clusters=clusterFps(fps,num_clusters=number_of_clusters)
        max_dist = []
        avg_dist = []
        for cluster_key in final_clusters:
            cluster_mols = final_clusters[cluster_key]
            cluster_mols = [mols[i] for i in cluster_mols]
            
            # get similarities
            #cluster_murcks = [MurckoScaffold.GetScaffoldForMol(mol) for mol in cluster_mols]
            cluster_fps = [AllChem.GetMorganFingerprintAsBitVect(x,2,1024) for x in cluster_mols]
            tan_array = [DataStructs.BulkTanimotoSimilarity(i, cluster_fps) for i in cluster_fps]
            flattened_tan_array = [item for sublist in tan_array for item in sublist]
            avg_dist.append(np.mean(flattened_tan_array))
            max_dist.append(np.min(flattened_tan_array))
        max_dists.append(np.average(max_dist))
        avg_dists.append(np.average(avg_dist))
    plt.scatter(list(range(1,max_num_clusters)), max_dists)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average of Minimum Similarity Within Cluster')
    plt.show()
    plt.scatter(list(range(1,max_num_clusters)), avg_dists)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average of Mean Similarity Within Cluster')
    plt.show()

    return(df, max_dists)
        
