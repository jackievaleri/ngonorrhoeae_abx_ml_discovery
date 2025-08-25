import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MCS, AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import AgglomerativeClustering

from adme_pred import ADME  # pip install ADME_predict is NOT the right one

# SECTION 1: MOLECULAR FILTERS

nitrofuran = "O=[N+](O)c1ccco1"
nitro_mol = Chem.MolFromSmiles(nitrofuran)

sulfonamide = "NS(=O)=O"
sulfa_mol = Chem.MolFromSmiles(sulfonamide)

quinolone = "O=c1cc[nH]c2ccccc12"
quino_mol = Chem.MolFromSmiles(quinolone)


def filter_for_logp_less_than(df, mols, thresh=5):
    """
    Filters molecules based on LogP values being below a specified threshold.

    Parameters:
    df (pd.DataFrame): DataFrame containing molecular information.
    mols (list): List of RDKit Mol objects corresponding to the DataFrame rows.
    thresh (float, optional): Maximum LogP value allowed. Default is 5.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: Filtered DataFrame with molecules passing the LogP filter.
        - list: List of RDKit Mol objects corresponding to the filtered DataFrame.
    """
    keep_indices = [Descriptors.MolLogP(mol) < thresh for mol in mols]
    df = df[keep_indices]
    print("length of df with logP < " + str(thresh) + ": ", len(df))
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    return (df, mols)


def check_pains_brenk(df, mols, method="both", thresh=0):
    """
    Filters molecules based on PAINS and/or Brenk structural alerts.

    Parameters:
    df (pd.DataFrame): DataFrame containing molecular information.
    mols (list): List of RDKit Mol objects corresponding to the DataFrame rows.
    method (str, optional): Filter method. Options are 'pains', 'brenk', or 'both'. Default is 'both'.
    thresh (int, optional): Maximum number of allowed matches before exclusion. Default is 0.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: Filtered DataFrame with molecules that passed the structural alert filter.
        - list: List of RDKit Mol objects corresponding to the filtered DataFrame.
    """
    # initialize filters
    params = FilterCatalogParams()
    if method == "both" or method == "pains":
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    if method == "both" or method == "brenk":
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog(params)

    def search_for_pains_or_brenk(mol):
        entry = catalog.GetMatches(mol)  # Get all matching PAINS or Brenk
        if entry is None:
            return True
        else:
            return len(entry) <= thresh

    keep_indices = [search_for_pains_or_brenk(m) for m in mols]
    df = df[keep_indices]
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    print("length of all preds with clean (no PAINS or Brenk) mols: ", len(df))
    return (df, mols)


def is_pattern(mol, pattern_mol):
    """
    Checks if a molecule contains a complete structural pattern (fragment match).

    Parameters:
    mol (RDKit Mol): The molecule to test.
    pattern_mol (RDKit Mol): The structural fragment to search for.

    Returns:
    bool: True if the molecule fully matches the pattern, False otherwise.
    """
    if mol is None:
        return False
    num_atoms_frag = pattern_mol.GetNumAtoms()
    mcs = MCS.FindMCS([mol, pattern_mol], atomCompare="elements", completeRingsOnly=True)
    if mcs.smarts is None:
        return False
    mcs_mol = Chem.MolFromSmarts(mcs.smarts)
    num_atoms_mcs = mcs_mol.GetNumAtoms()
    if num_atoms_frag == num_atoms_mcs:
        return True
    else:
        return False


def filter_for_nitrofuran(df, mols):
    """
    Removes molecules containing nitrofuran substructures.

    Parameters:
    df (pd.DataFrame): DataFrame containing molecular information.
    mols (list): List of RDKit Mol objects corresponding to the DataFrame rows.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: Filtered DataFrame without nitrofuran-containing molecules.
        - list: List of RDKit Mol objects corresponding to the filtered DataFrame.
    """
    keep_indices = [not is_pattern(mol, nitro_mol) for mol in mols]
    df = df[keep_indices]
    print("length of df with no nitrofurans: ", len(df))
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    return (df, mols)


def filter_for_sulfonamide(df, mols):
    """
    Removes molecules containing sulfonamide substructures.

    Parameters:
    df (pd.DataFrame): DataFrame containing molecular information.
    mols (list): List of RDKit Mol objects corresponding to the DataFrame rows.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: Filtered DataFrame without sulfonamide-containing molecules.
        - list: List of RDKit Mol objects corresponding to the filtered DataFrame.
    """
    keep_indices = [not is_pattern(mol, sulfa_mol) for mol in mols]
    df = df[keep_indices]
    print("length of df with no sulfonamides: ", len(df))
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    return (df, mols)


def filter_for_quinolone(df, mols):
    """
    Removes molecules containing quinolone motifs.

    Parameters:
    df (pd.DataFrame): DataFrame containing molecular information.
    mols (list): List of RDKit Mol objects corresponding to the DataFrame rows.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: Filtered DataFrame without quinolone-containing molecules.
        - list: List of RDKit Mol objects corresponding to the filtered DataFrame.
    """
    keep_indices = [not is_pattern(mol, quino_mol) for mol in mols]
    df = df[keep_indices]
    print("length of df with no quinolones: ", len(df))
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    return (df, mols)


def filter_for_mw_bounds(df, mols, lower_bound=100, upper_bound=600):
    """
    Filters molecules based on molecular weight (MW) within specified bounds.

    Parameters:
    df (pd.DataFrame): DataFrame containing molecular information.
    mols (list): List of RDKit Mol objects corresponding to the DataFrame rows.
    lower_bound (float, optional): Minimum molecular weight allowed. Default is 100.
    upper_bound (float, optional): Maximum molecular weight allowed. Default is 600.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: Filtered DataFrame with molecules in the specified MW range.
        - list: List of RDKit Mol objects corresponding to the filtered DataFrame.
    """
    keep_indices = [Descriptors.MolWt(mol) <= upper_bound and Descriptors.MolWt(mol) >= lower_bound for mol in mols]
    df = df[keep_indices]
    print(
        "length of df with " + str(lower_bound) + " < MW < " + str(upper_bound) + ": ",
        len(df),
    )
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    return (df, mols)


def filter_for_rotatable_bonds(df, mols, smiles_column, thresh=5):
    """
    Filters molecules based on the number of rotatable bonds.

    Parameters:
    df (pd.DataFrame): DataFrame containing SMILES strings.
    mols (list): List of RDKit Mol objects corresponding to the DataFrame rows.
    smiles_column (str): Column name in the DataFrame that contains SMILES strings.
    thresh (int, optional): Maximum number of rotatable bonds allowed. Default is 5.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: Filtered DataFrame with molecules below the rotatable bond threshold.
        - list: List of RDKit Mol objects corresponding to the filtered DataFrame.
    """
    smis = list(df[smiles_column])
    bonds = [ADME(smi)._n_rot_bonds() for smi in smis]
    keep_indices = [bond < thresh for bond in bonds]
    df = df[keep_indices]
    print("length of df with num rotatable bonds < " + str(thresh) + ": ", len(df))
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    return (df, mols)


def keep_valid_molecules(df, smiles_column):
    """
    Keeps only valid molecules that can be parsed from SMILES strings.

    Parameters:
    df (pd.DataFrame): DataFrame containing SMILES strings.
    smiles_column (str): Column name in the DataFrame containing SMILES.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: Filtered DataFrame with valid SMILES.
        - list: List of RDKit Mol objects corresponding to the filtered DataFrame.
    """
    smis = list(df[smiles_column])
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    keep_indices = [m is not None for m in mols]
    df = df[keep_indices]
    print("length of df with valid mols: ", len(df))
    mols = [m for i, m in enumerate(mols) if keep_indices[i]]
    smis = list(df[smiles_column])
    for m, smi in zip(mols, smis):
        m.SetProp("SMILES", smi)
    return (df, mols)


def deduplicate_on_tan_sim(df, mols):
    """
    Deduplicates molecules based on Tanimoto similarity of fingerprints.

    Parameters:
    df (pd.DataFrame): DataFrame containing molecular information.
    mols (list): List of RDKit Mol objects corresponding to the DataFrame rows.

    Returns:
    pd.DataFrame: DataFrame with duplicate molecules (Tanimoto similarity = 1) removed.
    """
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
    print("length of all preds deduplicated: ", len(df))
    return df


# SECTION 2: PROCESSING FUNCTIONS


def rank_order_preds(smis, scos, path):
    """
    Ranks molecules by their prediction scores and plots score vs. rank.

    Parameters:
    smis (list): List of SMILES strings.
    scos (list): List of scores corresponding to the SMILES.
    path (str): Directory path where the rank plot (SVG) will be saved.

    Returns:
    pd.DataFrame: DataFrame with columns:
        - 'smiles': SMILES strings.
        - 'scores': Prediction scores (floats).
        - 'rank': Rank of each molecule based on score.
    """
    new_scos = []
    new_smis = []
    for smi, sco in zip(smis, scos):
        try:
            new_scos.append(float(sco))
            new_smis.append(smi)
        except Exception:
            continue
    df = pd.DataFrame(zip(new_smis, new_scos), columns=["smiles", "scores"])
    df = df.sort_values(by="scores", ascending=False)
    df["rank"] = range(0, len(df))

    fig, ax = plt.subplots(figsize=(3, 2), dpi=300)
    plt.plot(df["rank"], df["scores"], color="slategrey")
    plt.xlabel("Rank")
    plt.ylabel("Score")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path + "score_wrt_rank.svg")
    plt.show()

    return df


def get_closest_tanimoto_from_drug_set(new_set, drug_fps):
    """
    Computes the closest Tanimoto similarity between a set of molecules and a reference drug set.

    For every molecule, get similarity to closest drug in a set of drug fingerprints.
    This is simpler than other TS code because there is no need to save the specific drug name/SMILES here.

    Parameters:
    new_set (list): List of SMILES strings to evaluate.
    drug_fps (list): List of RDKit fingerprint objects for the reference drug set.

    Returns:
    list: List of maximum Tanimoto similarities (floats) for each molecule in new_set.
          Returns 'NaN' for invalid SMILES or failed fingerprint generation.
    """
    # new_set is list of smiles
    # drug_fps is list of fingerprints of things you want to compare to
    try:
        mols = [Chem.MolFromSmiles(x) for x in new_set]
        # more info on RDK Fingerprints (similar to Morgan fingerprints) here
        # https://stackoverflow.com/questions/67811388/how-can-i-interpret-the-features-obtained-from-chem-rdkfingerprintmol # noqa: E501
        fps = [Chem.RDKFingerprint(x) if x is not None else "" for x in mols]
    except Exception as e:
        print(e)
        return [1000]

    best_similarity = []

    i = 0
    for mol in fps:  # corresponds to the 'new_set' molecules
        curr_highest_sim = 0
        if mol == "":
            best_similarity.append("NaN")
            continue
        j = 0
        for drug in drug_fps:  # corresponds to the list of things you want to compare to
            try:
                sim = DataStructs.FingerprintSimilarity(mol, drug)
            except Exception:
                continue
            if sim > curr_highest_sim and i != j:  # make sure they are not the same molecule
                curr_highest_sim = sim
            j = j + 1
        best_similarity.append(curr_highest_sim)
        i = i + 1
    return best_similarity


def compute_highest_tan_sim_intraset(smis):
    """
    Computes the highest Tanimoto similarity of each molecule against others in the same set.

    Parameters:
    smis (list): List of SMILES strings.

    Returns:
    list: List of maximum intra-set Tanimoto similarities (floats).
    """
    smis = [smi for smi in smis if not isinstance(smi, float)]
    mols = [Chem.MolFromSmiles(x) for x in smis]
    fps = [Chem.RDKFingerprint(x) if x is not None else "" for x in mols]
    tans = get_closest_tanimoto_from_drug_set(smis, fps)
    return tans


def plot_tan_wrt_rank(df, path, score_col="scores", tan_col="tan"):
    """
    Generates a scatter plot of Tanimoto similarity versus rank.

    Parameters:
    df (pd.DataFrame): DataFrame containing scores and Tanimoto similarities.
    path (str): Directory path where the plot will be saved.
    score_col (str, optional): Column name containing scores. Default is 'scores'.
    tan_col (str, optional): Column name containing Tanimoto similarities. Default is 'tan'.

    Returns:
    None: Saves the scatter plot as an SVG file and displays it.
    """
    df = df.sort_values(by=score_col, ascending=False)

    fig, ax = plt.subplots(figsize=(3, 2), dpi=300)
    plt.scatter(df[score_col], df[tan_col], s=0.5, color="slategrey")
    plt.xlabel("Rank")
    plt.ylabel("Highest Inter-Set Tanimoto")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path + "tan_wrt_rank.svg")
    plt.show()
    return


def plot_tan_wrt_score(df, path, score_col="scores", tan_col="tan"):
    """
    Generates a scatter plot of Tanimoto similarity versus score.

    Parameters:
    df (pd.DataFrame): DataFrame containing scores and Tanimoto similarities.
    path (str): Directory path where the plot will be saved.
    score_col (str, optional): Column name containing scores. Default is 'scores'.
    tan_col (str, optional): Column name containing Tanimoto similarities. Default is 'tan'.

    Returns:
    None: Saves the scatter plot as an SVG file and displays it.
    """
    fig, ax = plt.subplots(figsize=(3, 2), dpi=300)
    plt.scatter(df[score_col], df[tan_col], s=0.5, color="slategrey")
    plt.xlabel("Score")
    plt.ylabel("Highest Inter-Set Tanimoto")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path + "tan_wrt_score.svg")
    plt.show()


def process_preds(df, smiles_col, hit_col, path):
    """
    Processes prediction results by ranking, computing Tanimoto similarities, and plotting.

    Parameters:
    df (pd.DataFrame): DataFrame containing SMILES and hit scores.
    smiles_col (str): Column name containing SMILES strings.
    hit_col (str): Column name containing prediction scores.
    path (str): Directory path where plots will be saved.

    Returns:
    None: Generates ranked DataFrame and plots for score vs. rank and Tanimoto similarity.
    """
    smis = list(df[smiles_col])
    scos = list(df[hit_col])

    # get ranks
    ranks = rank_order_preds(smis, scos, path)

    # get tanimotos
    ranks = ranks.iloc[
        0:10000, :
    ]  # use just the first 10K molecules by score to do tan analysis - else it takes forever
    tans = compute_highest_tan_sim_intraset(ranks["smiles"])  # standardized to lowercase so can do this
    ranks["tan"] = tans

    # additional plots
    plot_tan_wrt_rank(ranks, path)
    plot_tan_wrt_score(ranks, path)


def evaluate_scores(df, hit_col):
    """
    Evaluates prediction scores by converting them to floats and reporting distribution.

    Parameters:
    df (pd.DataFrame): DataFrame containing prediction scores.
    hit_col (str): Column name containing prediction scores.

    Returns:
    pd.DataFrame: DataFrame with scores converted to float and descriptive counts printed.
    """
    df[hit_col] = [float(x) for x in df[hit_col]]
    print("Total number in df: ", len(df))

    for i in np.arange(0.1, 1, 0.1):
        print(">" + str(i) + ": " + str(len(df[df[hit_col] > i])))
    return df


def print_drug_dict(
    drugdict,
    smiles_list,
    col_list,
    names_list,
    merging_df,
    left_smiles="SMILES",
    right_smiles="SMILES",
):
    """
    Builds a DataFrame summarizing drug similarity results from a dictionary.

    There is some hardcoded stuff in here, which is not ideal.

    Parameters:
    drugdict (dict): Dictionary with SMILES as keys and similarity/index lists as values.
    smiles_list (list): List of SMILES strings corresponding to the drug set.
    col_list (list): List of column names for the resulting DataFrame.
    names_list (list or None): List of drug names corresponding to the SMILES. If None, names are omitted.
    merging_df (pd.DataFrame): DataFrame to merge similarity results with.
    left_smiles (str, optional): Column in results for left merge. Default is 'SMILES'.
    right_smiles (str, optional): Column in merging_df for right merge. Default is 'SMILES'.

    Returns:
    pd.DataFrame: DataFrame with similarity results merged with metadata.
    """

    ret = pd.DataFrame(columns=col_list)
    for key, item in drugdict.items():
        tans, indexes = item
        for tan, index in zip(tans, indexes):
            if names_list is None:
                ret = pd.concat(
                    [
                        ret,
                        pd.DataFrame([[key, tan, smiles_list[index], index]], columns=col_list),
                    ]
                )
            else:
                ret = pd.concat(
                    [
                        ret,
                        pd.DataFrame(
                            [[key, tan, smiles_list[index], names_list[index]]],
                            columns=col_list,
                        ),
                    ]
                )

    ret = ret.merge(merging_df, left_on=left_smiles, right_on=right_smiles)
    return ret


def get_lowest_tanimoto_from_drug_set(
    new_set,
    abx_fps,
    smiles_list,
    col_list,
    merging_df,
    names_list=None,
    smiles_col="SMILES",
):
    """
    Computes closest Tanimoto similarity between a new set of molecules and another set of molecules.

    For every molecule, get similarity to closest antibiotic or molecule in another dataset.
    The default is assumed to be that we are comparing to a list of antibiotics,
    but this can be used for any set of molecules.

    Parameters:
    new_set (list): List of SMILES strings to compare.
    abx_fps (list): List of RDKit fingerprint objects for the antibiotic set.
    smiles_list (list): List of antibiotic SMILES strings.
    col_list (list): Column names for the output DataFrame.
    merging_df (pd.DataFrame): DataFrame to merge results into.
    names_list (list, optional): List of antibiotic names. Default is None.
    smiles_col (str, optional): Column in merging_df containing SMILES. Default is 'SMILES'.

    Returns:
    pd.DataFrame: DataFrame containing each SMILES, similarity to closest antibiotic, and metadata.
    """
    mols = [Chem.MolFromSmiles(x) for x in new_set]

    best_similarity = {}
    index = 0

    for m in mols:
        try:
            mol = Chem.RDKFingerprint(m)
        except Exception:
            index = index + 1
            continue
        set_index = 0
        curr_highest_sim = 0
        curr_highest_drug = None
        for abx in abx_fps:
            sim = DataStructs.FingerprintSimilarity(mol, abx)  # default metric is tanimoto similarity
            if sim > curr_highest_sim:
                curr_highest_sim = sim
                curr_highest_drug = set_index
            set_index = set_index + 1
        info_to_include = [[curr_highest_sim], [curr_highest_drug]]
        best_similarity[new_set[index]] = info_to_include

        index = index + 1
    # kind of convoluted b/c I like seeing the actual pandas df and
    # it's helpful to keep track of names, smiles, and values
    ret = print_drug_dict(
        best_similarity,
        smiles_list,
        col_list,
        names_list,
        merging_df,
        right_smiles=smiles_col,
    )
    return ret


def clean_up_names_and_smiles(df, name_col="Name", smiles_col="SMILES"):
    """
    Cleans up a DataFrame by removing invalid SMILES and molecules that cannot generate fingerprints.

    Unfortunately, sometimes SMILES can be NaN, or create invalid molecules,
    or fingerprints cannot be created from mols.

    Parameters:
    df (pd.DataFrame): DataFrame containing SMILES and names.
    name_col (str, optional): Column containing molecule names. Default is 'Name'.
    smiles_col (str, optional): Column containing SMILES strings. Default is 'SMILES'.

    Returns:
    tuple: A tuple containing:
        - list: List of RDKit fingerprint objects.
        - list: List of valid SMILES strings.
        - list: List of corresponding molecule names.
    """
    not_nans = [not isinstance(smi, float) for smi in df[smiles_col]]
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
    return (new_mols, new_smis, new_names)


def compute_tanimoto_against_abx(smis, merging_df, smiles_col="SMILES"):
    """
    Computes Tanimoto similarity of molecules against a curated antibiotic set.

    Parameters:
    smis (list): List of SMILES strings to evaluate.
    merging_df (pd.DataFrame): DataFrame to merge results with.
    smiles_col (str, optional): Column name in merging_df containing SMILES. Default is 'SMILES'.

    Returns:
    pd.DataFrame: DataFrame with Tanimoto similarities to closest antibiotics and metadata.
    """
    df = pd.read_csv("../data/04052022_CLEANED_v5_antibiotics_across_many_classes.csv")
    clean_mols, clean_smis, clean_names = clean_up_names_and_smiles(df, name_col="Name", smiles_col="Smiles")
    col_list = [
        "SMILES",
        "tanimoto similarity to closest abx",
        "closest abx smiles",
        "closest abx name",
    ]
    gen_mols = get_lowest_tanimoto_from_drug_set(
        smis,
        clean_mols,
        clean_smis,
        col_list,
        merging_df,
        clean_names,
        smiles_col=smiles_col,
    )
    return gen_mols


def compute_tanimoto_against_training_set(
    smis,
    merging_df,
    train_set_path,
    hit_col="hit",
    just_hits=False,
    smiles_col="SMILES",
):
    """
    Computes Tanimoto similarity of molecules against a training set.

    Parameters:
    smis (list): List of SMILES strings to evaluate.
    merging_df (pd.DataFrame): DataFrame to merge results with.
    train_set_path (str): Path to CSV file containing the training set.
    hit_col (str, optional): Column in training set indicating hits. Default is 'hit'.
    just_hits (bool, optional): If True, restricts training set to hits only. Default is False.
    smiles_col (str, optional): Column name in merging_df containing SMILES. Default is 'SMILES'.

    Returns:
    pd.DataFrame: DataFrame with Tanimoto similarities to closest training set molecules.
    """
    df = pd.read_csv(train_set_path)
    if just_hits:
        df = df[df[hit_col] == 1]
    clean_mols, clean_smis, clean_names = clean_up_names_and_smiles(df, name_col="Name", smiles_col="SMILES")
    col_list = [
        "SMILES",
        "tanimoto similarity to closest train set",
        "closest train set smiles",
        "closest train set name",
    ]
    gen_mols = get_lowest_tanimoto_from_drug_set(
        smis,
        clean_mols,
        clean_smis,
        col_list,
        merging_df,
        clean_names,
        smiles_col=smiles_col,
    )
    return gen_mols


def evaluate_similarities(
    df,
    smiles_col,
    path,
    train_set_path,
    thresh_name="04",
    hit_col="hit",
    just_hits=False,
):
    """
    Evaluates molecule similarity against antibiotics and training sets, generating histograms.

    Parameters:
    df (pd.DataFrame): DataFrame containing SMILES strings.
    smiles_col (str): Column name containing SMILES.
    path (str): Directory path where plots will be saved.
    train_set_path (str): Path to CSV file containing the training set.
    thresh_name (str, optional): Identifier for saving plots. Default is '04'.
    hit_col (str, optional): Column in training set indicating hits. Default is 'hit'.
    just_hits (bool, optional): If True, restricts training set to hits only. Default is False.

    Returns:
    pd.DataFrame: DataFrame with similarity scores against both antibiotic and training sets.
    """

    def plot_tanimoto_score_plot(df, tanimoto_col, path, name):
        """
        Plots a histogram of Tanimoto similarity scores.

        Parameters:
        df (pd.DataFrame): DataFrame containing Tanimoto similarity values.
        tanimoto_col (str): Column name in the DataFrame with Tanimoto similarities.
        path (str): Directory path where the plot will be saved.
        name (str): Identifier for the plot file name.

        Returns:
        None: Saves the histogram as a PNG file and displays it.
        """
        plt.figure(figsize=(5, 3), dpi=300)
        plt.hist(df[tanimoto_col], bins=50, edgecolor="black", linewidth=0.5)
        plt.xlabel(tanimoto_col)
        plt.ylabel("Number")
        plt.savefig(path + name + ".png")
        plt.show()

    print("Computing tanimoto scores against abx...")
    smis = list(df[smiles_col])
    df = compute_tanimoto_against_abx(smis, df, smiles_col=smiles_col)
    plot_tanimoto_score_plot(
        df,
        "tanimoto similarity to closest abx",
        path,
        "inh" + thresh_name + "_tan_to_closest_abx",
    )

    print("Computing tanimoto scores against training set...")
    df = compute_tanimoto_against_training_set(
        smis,
        df,
        train_set_path=train_set_path,
        hit_col=hit_col,
        just_hits=just_hits,
        smiles_col=smiles_col,
    )
    plot_tanimoto_score_plot(
        df,
        "tanimoto similarity to closest train set",
        path,
        "inh" + thresh_name + "_tan_to_closest_train_set",
    )
    return df


# SECTION 3: Clustering


def clusterFps(fps, num_clusters):
    """
    Clusters molecular fingerprints using agglomerative clustering.

    Code adapted from https://www.macinchem.org/reviews/clustering/clustering.php

    Parameters:
    fps (list): List of RDKit fingerprint objects.
    num_clusters (int): Number of clusters to generate.

    Returns:
    tuple: A tuple containing:
        - np.ndarray: Cluster labels for each molecule.
        - dict: Dictionary mapping cluster IDs to lists of molecule indices.
    """
    tan_array = [DataStructs.BulkTanimotoSimilarity(i, fps) for i in fps]
    tan_array = np.array(tan_array)
    clusterer = AgglomerativeClustering(n_clusters=num_clusters, compute_full_tree=True).fit(tan_array)
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
    """
    Attaches legends to molecules for visualization, including similarity and score info.

    Parameters:
    mols (list): List of RDKit Mol objects.
    smis (list): List of SMILES strings corresponding to the molecules.
    df (pd.DataFrame): DataFrame containing metadata (names, scores, similarities).
    smiles_col (str): Column containing SMILES strings.
    name_col (str): Column containing molecule names.
    hit_col (str): Column containing prediction scores.
    tans (bool): Whether to include Tanimoto similarity values in legends.

    Returns:
    list: List of RDKit Mol objects with legend properties set.
    """
    row_num = 0
    for _smi in smis:
        try:
            row = df.iloc[row_num, :]
            try:
                actualrowname = str(row.loc[name_col])
            except Exception as e:
                print(e)
                actualrowname = str(row.loc[smiles_col])
            if len(actualrowname) > 20:
                actualrowname = actualrowname[0:20] + "..."
            hit = str(np.round(float(row.loc[hit_col]), 3))
            actual_row_num = str(row.loc["row_num"])

            if tans:
                tan_train = str(np.round(float(row.loc["tanimoto similarity to closest train set"]), 3))
                tan_abx = str(np.round(float(row.loc["tanimoto similarity to closest abx"]), 3))
                legend = (
                    actualrowname + "\n" + "tantrain: " + tan_train + "\n" + "tanabx: " + tan_abx + "\n score: " + hit
                )
            else:
                legend = actualrowname + "\n" + "\n score: " + hit

        except Exception as e:
            print(e)
            actual_row_num = str(row.loc["row_num"])
            legend = "row: " + actual_row_num

        mols[row_num].SetProp("legend", legend)
        row_num = row_num + 1
    return mols


def extract_legends_and_plot(
    df,
    name,
    folder,
    num_clusters=30,
    smiles_col="smiles",
    name_col="Name",
    hit_col="hit",
    tans=True,
):
    """
    Generates molecule cluster plots with legends using Murcko scaffolds and Tanimoto clustering.

    Parameters:
    df (pd.DataFrame): DataFrame containing SMILES and associated metadata.
    name (str): Identifier for plot naming.
    folder (str): Output folder for cluster images.
    num_clusters (int, optional): Number of clusters to generate. Default is 30.
    smiles_col (str, optional): Column containing SMILES strings. Default is 'smiles'.
    name_col (str, optional): Column containing molecule names. Default is 'Name'.
    hit_col (str, optional): Column containing prediction scores. Default is 'hit'.
    tans (bool, optional): Whether to include Tanimoto similarity values in legends. Default is True.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: Input DataFrame with cluster labels added.
        - list: List of RDKit Mol objects with legend properties set.
    """
    df["row_num"] = list(range(len(df)))
    df = df.drop_duplicates(subset=smiles_col)
    smis = list(df[smiles_col])

    mols = [Chem.MolFromSmiles(mol) for mol in smis]
    mols = make_legends(
        mols,
        smis,
        df,
        smiles_col=smiles_col,
        name_col=name_col,
        hit_col=hit_col,
        tans=tans,
    )

    # code with help from the OG greg landrum: https://gist.github.com/greglandrum/d5f12058682f6b336905450e278d3399
    molsPerRow = 5
    subImgSize = (500, 500)

    murcks = [MurckoScaffold.GetScaffoldForMol(mol) for mol in mols]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in murcks]
    raw_cluster_labels, final_clusters = clusterFps(fps, num_clusters=num_clusters)

    # show clusters
    name_index = 0
    for cluster_key in final_clusters:
        cluster_mols = final_clusters[cluster_key]
        cluster_mols = [mols[i] for i in cluster_mols]

        nRows = len(cluster_mols) // molsPerRow
        if len(cluster_mols) % molsPerRow:
            nRows += 1
        fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
        d2d = rdMolDraw2D.MolDraw2DCairo(fullSize[0], fullSize[1], subImgSize[0], subImgSize[1])
        d2d.drawOptions().legendFontSize = 100
        # d2d.drawOptions().useBWAtomPalette()
        d2d.DrawMolecules(cluster_mols, legends=[mol.GetProp("legend") for mol in cluster_mols])
        d2d.FinishDrawing()
        new_name = folder + str(name_index) + "_" + name
        open(new_name, "wb+").write(d2d.GetDrawingText())
        name_index = name_index + 1
    df["cluster"] = [str(i) for i in raw_cluster_labels]  # so it gets interpreted by plotly in a good way
    return (df, mols)


def determine_optimal_clustering_number(df, max_num_clusters, smiles_col):
    """
    Determines the optimal number of clusters by computing intra-cluster similarity metrics.

    Parameters:
    df (pd.DataFrame): DataFrame containing SMILES strings.
    max_num_clusters (int): Maximum number of clusters to evaluate.
    smiles_col (str): Column containing SMILES strings.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: Input DataFrame (with duplicates dropped).
        - list: List of average minimum and mean Tanimoto similarities for different cluster counts.
    """
    df["row_num"] = list(range(len(df)))
    df = df.drop_duplicates(subset=smiles_col)
    smis = list(df[smiles_col])
    mols = [Chem.MolFromSmiles(mol) for mol in smis]
    murcks = [MurckoScaffold.GetScaffoldForMol(mol) for mol in mols]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in murcks]

    max_dists = []
    avg_dists = []
    print(max_num_clusters)
    for number_of_clusters in range(1, max_num_clusters):
        raw_cluster_labels, final_clusters = clusterFps(fps, num_clusters=number_of_clusters)
        max_dist = []
        avg_dist = []
        for cluster_key in final_clusters:
            cluster_mols = final_clusters[cluster_key]
            cluster_mols = [mols[i] for i in cluster_mols]

            # get similarities
            # cluster_murcks = [MurckoScaffold.GetScaffoldForMol(mol) for mol in cluster_mols]
            cluster_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in cluster_mols]
            tan_array = [DataStructs.BulkTanimotoSimilarity(i, cluster_fps) for i in cluster_fps]
            flattened_tan_array = [item for sublist in tan_array for item in sublist]
            avg_dist.append(np.mean(flattened_tan_array))
            max_dist.append(np.min(flattened_tan_array))
        max_dists.append(np.average(max_dist))
        avg_dists.append(np.average(avg_dist))
    plt.scatter(list(range(1, max_num_clusters)), max_dists)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Average of Minimum Similarity Within Cluster")
    plt.show()
    plt.scatter(list(range(1, max_num_clusters)), avg_dists)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Average of Mean Similarity Within Cluster")
    plt.show()

    return (df, max_dists)
