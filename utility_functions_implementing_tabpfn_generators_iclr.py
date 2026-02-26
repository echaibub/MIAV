
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata
from tqdm import tqdm

from tabpfn import TabPFNClassifier
from tabpfn import TabPFNRegressor


def get_variable_types(df: pd.DataFrame) -> Dict[str, Optional[List]]:
    """
    Classify DataFrame columns into numeric vs. categorical,
    similar to the R GetVariableTypes() function.

    Rules (to match your R code):
      - numeric_vars: columns with numeric/integer types
      - categorical_vars: columns with factor/character/logical types
        (pandas equivalents: category, string/object, boolean)

    Returns a dict with:
      - num_variables:      list of 0-based indexes for numeric columns (or None if empty)
      - cat_variables:      list of 0-based indexes for categorical columns (or None if empty)
      - num_variable_names: list of column names for numeric columns
      - cat_variable_names: list of column names for categorical columns
    """
    # Make sure df is a pandas dataframe
    df = pd.DataFrame(df)
    
    # Numeric columns (int, float, nullable Int/Float, etc.)
    num_names = df.select_dtypes(include=["number"]).columns.tolist()

    # Categorical-like columns: category, string/object, boolean (incl. nullable)
    cat_names = df.select_dtypes(
        include=["category", "object", "string", "bool", "boolean"]
    ).columns.tolist()

    # Build 0-based indexes
    all_cols = df.columns.tolist()
    num_idx = [all_cols.index(c) for c in num_names]
    cat_idx = [all_cols.index(c) for c in cat_names]

    # Match R's NULL semantics by using None when empty
    if len(num_idx) == 0:
        num_idx = None
    if len(cat_idx) == 0:
        cat_idx = None

    return {
        "num_variables": num_idx,
        "cat_variables": cat_idx,
        "num_variable_names": num_names,
        "cat_variable_names": cat_names,
    }


def round_integer_variables(df_ori: pd.DataFrame, df_syn: pd.DataFrame, atol: float = 1e-8) -> pd.DataFrame:
    """
    This function uses the original data to determine which variables
    are integer-valued and then rounds the values of the corresponding
    variables in the synthetic data to the nearest integer.

    Parameters
    ----------
    df_ori : pd.DataFrame
        DataFrame containing the original data.
    df_syn : pd.DataFrame
        DataFrame containing the synthetic data.
        Assumes same schema/columns as `df_ori`.
    atol : float, optional
        Absolute tolerance when testing whether original data is
        integer-valued (default = 1e-8). This mimics R's all.equal().

    Returns
    -------
    pd.DataFrame
        Synthetic data with rounded values for integer-valued variables.
    """
    # Make sure df_ori and df_syn are pandas dataframes
    df_ori = pd.DataFrame(df_ori)
    df_syn = pd.DataFrame(df_syn)
    
    # Make a copy so original synthetic data is not modified
    syn = df_syn.copy()

    # Iterate only over numeric columns in the original data
    for col in df_ori.select_dtypes(include="number").columns:
        # Drop missing values to avoid issues when checking integer-ness
        s = df_ori[col].dropna()

        # If column is not empty, check if all values are "integer-like"
        # i.e., equal to their rounded version (within tolerance)
        if len(s) and np.isclose(s.to_numpy(), np.round(s.to_numpy()), atol=atol).all():
            # Round the synthetic data for this column to the nearest integer
            #syn[col] = np.round(syn[col])
            syn[col] = np.rint(pd.to_numeric(syn[col], errors="coerce"))

    return syn



def ranking_with_random_tie_breaking(x):
    """
    Rank data with random tie-breaking by shuffling indices.
    
    Args:
        x (array-like): Input data to rank.
        
    Returns:
        np.ndarray: Ranks with random tie-breaking.
    """
    x = np.array(x)
    
    # Shuffle the indices to break ties randomly
    shuffled_indices = np.random.permutation(len(x))
    shuffled_x = x[shuffled_indices]
    
    # Rank the shuffled data
    ranks = rankdata(shuffled_x, method='ordinal')

    # Reorder ranks back to original order
    return ranks[np.argsort(shuffled_indices)]


    
def match_ranks(synthetic_marginals_matrix: pd.DataFrame, rank_matrix: pd.DataFrame):
    """
    Matches the ranks between synthetic marginals and a rank matrix.

    Parameters:
        synthetic_marginals_matrix (pd.DataFrame): Synthetic marginals matrix.
        rank_matrix (pd.DataFrame): Rank matrix.

    Returns:
        pd.DataFrame: Matrix with matched ranks.
    """
    dat_s = synthetic_marginals_matrix.copy()
    p = rank_matrix.shape[1]
    n = rank_matrix.shape[0]
    
    for j in range(p):
        # Sort the synthetic data column
        sorted_syn_dat = np.sort(synthetic_marginals_matrix.iloc[:, j].values)
        
        # Match sorted values to the provided ranks
        # Subtract 1 from the ranks because ranks are 1-based (like in R) 
        dat_s.iloc[:, j] = sorted_syn_dat[rank_matrix.iloc[:, j].values.astype(int) - 1]
    
    return dat_s



def numeric_rank_encoding(x):
    r = np.full(len(x), np.nan)
    unique_levels, counts = np.unique(x, return_counts=True)
    cumulative_counts = np.cumsum(np.concatenate(([0], counts)))
    n_levels = len(unique_levels)
     
    for j in range(n_levels):
        level = unique_levels[j]
        idx = np.where(x == level)[0]  # Get indices of this level
        lower_bound = cumulative_counts[j] + 1
        upper_bound = cumulative_counts[j + 1]
        r[idx] = np.random.permutation(np.arange(lower_bound, upper_bound + 1))

    # Make sure the output is of integer type
    r = r.astype(int)
    
    return r



def generate_maximal_information_auxiliary_variable(x):
    """
    Generate a maximal information auxiliary variable for x.

    Parameters
    ----------
    x : array-like (numeric, categorical, string, or boolean)

    Returns
    -------
    m : np.ndarray
        Maximal information auxiliary variable aligned to the ranks of x
    """
    x = pd.Series(x)  # ensure pandas Series for type handling
    n = len(x)

    # Create sorted uniform random values
    m = np.sort(np.random.uniform(size=n))

    if pd.api.types.is_numeric_dtype(x):
        # Compute standard rank with ties broken at random
        r = ranking_with_random_tie_breaking(x)
        m = m[r - 1]
        
    if not pd.api.types.is_numeric_dtype(x):
        # Convert to categorical and apply the numeric rank encoding 
        # to the categorical variable
        x = pd.Categorical(x)
        r = numeric_rank_encoding(x)
        m = m[r - 1]

    return m



def generate_prediction_using_tabpfn(X_trn, X_tst, y_trn):
    
    # Determine target data type
    target_type = y_trn.dtype

    # If target type is numeric, fit TabPFN regression
    if pd.api.types.is_numeric_dtype(target_type):
        # Initialize the regressor
        regressor = TabPFNRegressor()
        regressor.fit(X_trn, y_trn)
        # Predict on the test set
        predictions = regressor.predict(X_tst)

    # Otherwise, fit TabPFN classification
    if not pd.api.types.is_numeric_dtype(target_type):
        # Initialize the classifier
        clf = TabPFNClassifier()
        clf.fit(X_trn, y_trn)
        # Predict on the test set
        predictions = clf.predict(X_tst)
    
    return predictions



def icl_with_miav_tabpfn(X_trn: pd.DataFrame, X_tst: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """
    This function uses, for each column j:
      1) A maximal-information auxiliary variable (MIAV) computed from y_trn = X_trn[j]
         and y_tst = X_tst[j].
      2) A TabPFN-based predictor trained on the MIAV feature (1 column "m") with y_trn
         as targets, to generate synthetic predictions for the test rows.

    Parameters
    ----------
    X_trn : pd.DataFrame
        Training data (rows = samples, columns = variables).
    X_tst : pd.DataFrame
        Test data (same columns as X_trn).

    Returns
    -------
    Z_tst : pd.DataFrame
        Synthetic data with the same shape and columns as X_tst.
        Column j contains the prediction produced for that variable.
    """
    # Initialize synthetic output preallocated with the expected data types
    Z_tst = pd.DataFrame(
        {c: pd.Series(index=X_tst.index, dtype=object) for c in X_tst.columns},
        index=X_tst.index,
    )

    # Iterate columns (variables)
    it = tqdm(X_tst.columns, desc="Synthesizing columns") if show_progress else X_tst.columns

    for col in it:

        # Compute MIAVs
        m_trn_vec = generate_maximal_information_auxiliary_variable(X_trn[col]) 
        m_tst_vec = generate_maximal_information_auxiliary_variable(X_tst[col])

        # Wrap as 1-column DataFrames named "m" (mirrors R's data.frame(matrix(..., nc=1)))
        m_trn_df = pd.DataFrame({"m": np.asarray(m_trn_vec).reshape(-1)}, index=X_trn.index)
        m_tst_df = pd.DataFrame({"m": np.asarray(m_tst_vec).reshape(-1)}, index=X_tst.index)

        # Predict using TabPFN on MIAV with y_trn as targets
        pred = generate_prediction_using_tabpfn(X_trn=m_trn_df, X_tst=m_tst_df, y_trn=X_trn[col])

        # Fill synthetic column
        Z_tst[col] = np.asarray(pred).reshape(-1)

    return Z_tst



def miav_tabpfn_generator(X: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """
    Steps:
      1) Split X into two halves (X1 = first half, X2 = second half).
      2) Train on X2, predict on X1 -> Z1 via ICLwithMiavTabPFN.
      3) Train on X1, predict on X2 -> Z2 via ICLwithMiavTabPFN.
      4) Row-bind predictions: Z = rbind(Z1, Z2)  (i.e., stack them).
      5) Round synthetic columns that are integer-valued in X.

    Parameters
    ----------
    X : pd.DataFrame
        Input data (rows = samples, columns = variables).

    Returns
    -------
    pd.DataFrame
        Synthetic DataFrame with the same shape and columns as X.
    """
    X = pd.DataFrame(X)
    
    n = len(X)

    # Match R's round(n/2): R rounds to even; np.rint does that, too.
    split = int(np.rint(n / 2.0))

    # X1 = first half; X2 = second half 
    X1 = X.iloc[:split, :]
    X2 = X.iloc[split:, :]

    #print("train with X2, query with X1")
    Z1 = icl_with_miav_tabpfn(X_trn=X2, X_tst=X1, show_progress = show_progress)

    #print("train with X1, query with X2")
    Z2 = icl_with_miav_tabpfn(X_trn=X1, X_tst=X2, show_progress = show_progress)

    # rbind(Z1, Z2) → vertical concat; this preserves the original row order here
    Z = pd.concat([Z1, Z2], axis=0)

    # Round integer-valued columns (detected from X) in the synthetic data Z
    Z = round_integer_variables(df_ori=X, df_syn=Z)

    return Z



def icl_with_joint_factorization_tabpfn(X_trn: pd.DataFrame, X_tst: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """
    For j = 1:
      - Build a single random feature X0 (uniform[0,1]) for train/test.
      - Predict y_trn = X_trn.iloc[:, 0] using TabPFN with X0 as the only feature.

    For j = 2..p:
      - Use the first (j-1) ORIGINAL columns of X as features to predict column j.

    Parameters
    ----------
    X_trn : pd.DataFrame
        Training data, shape (n_trn, p).
    X_tst : pd.DataFrame
        Test data, shape (n_tst, p). Must have the same columns as X_trn.

    Returns
    -------
    Z_tst : pd.DataFrame
        Synthetic data with the same shape and columns as X_tst.
    """
    n_trn, p = X_trn.shape
    n_tst = len(X_tst)

    # Initialize synthetic output preallocated with the expected data types
    Z_tst = pd.DataFrame(
        {c: pd.Series(index=X_tst.index, dtype=object) for c in X_tst.columns},
        index=X_tst.index,
    )


    # ----- j = 1 (first column uses a single random feature X0) -----
    X0_trn = pd.DataFrame({"X0": np.random.uniform(size=n_trn)}, index=X_trn.index)
    X0_tst = pd.DataFrame({"X0": np.random.uniform(size=n_tst)}, index=X_tst.index)

    x_trn_j = X_trn.iloc[:, 0]
    z_col = generate_prediction_using_tabpfn(X_trn=X0_trn, X_tst=X0_tst, y_trn=x_trn_j)
    Z_tst.iloc[:, 0] = np.asarray(z_col).reshape(-1)

    # ----- j = 2..p (use first j-1 ORIGINAL columns as features) -----

    it = tqdm(range(2, p + 1), desc="Synthesizing columns") if show_progress else range(2, p + 1)

    for j in it:
        feats_trn = X_trn.iloc[:, : (j - 1)].copy()
        feats_tst = X_tst.iloc[:, : (j - 1)].copy()

        x_trn_j = X_trn.iloc[:, j - 1]  # column j (0-based)
        z_col = generate_prediction_using_tabpfn(X_trn=feats_trn, X_tst=feats_tst, y_trn=x_trn_j)
        Z_tst.iloc[:, j - 1] = np.asarray(z_col).reshape(-1)

    return Z_tst



def joint_factorization_tabpfn_generator(X: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """
    Steps:
      1) Split X rows in half: X1 (first half), X2 (second half).
      2) Train with X2, query with X1 -> Z1 via icl_with_joint_factorization_tabpfn.
      3) Train with X1, query with X2 -> Z2 via icl_with_joint_factorization_tabpfn.
      4) Row-bind Z1 and Z2 in original order.
      5) Round integer-valued columns (detected from X) in Z.

    Parameters
    ----------
    X : pd.DataFrame
        Input data (rows = samples, columns = variables).
    generate_prediction_tabpfn : callable
        Python equivalent of GeneratePredictionUsingTabPFN(X_trn, X_tst, y_trn).
        This is forwarded to icl_with_joint_factorization_tabpfn.
    round_integer_variables_fn : callable
        Function like round_integer_variables(df_ori, df_syn) -> df_syn_rounded.

    Returns
    -------
    pd.DataFrame
        Synthetic DataFrame with same shape/columns as X.
    """
    n = len(X)

    # Match R's round(n/2) behavior (banker's rounding)
    split = int(np.rint(n / 2.0))

    X1 = X.iloc[:split, :]
    X2 = X.iloc[split:, :]

    #print("train with X2, query with X1")
    Z1 = icl_with_joint_factorization_tabpfn(X_trn=X2, X_tst=X1, show_progress = show_progress)

    #print("train with X1, query with X2")
    Z2 = icl_with_joint_factorization_tabpfn(X_trn=X1, X_tst=X2, show_progress = show_progress)

    # rbind(Z1, Z2) -> vertical concat; preserves the original row split order
    Z = pd.concat([Z1, Z2], axis=0)

    # Round integer-valued variables (detected from X) in the synthetic data
    Z = round_integer_variables(df_ori=X, df_syn=Z)

    return Z



def icl_with_full_conditionals_tabpfn(X_trn: pd.DataFrame, X_tst: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """
    For each column j:
      - Features = all columns EXCEPT j (train/test)
      - Target   = column j of X_trn
      - Predict  = TabPFN(X_trn_minus_j -> y_trn) on X_tst_minus_j

    Parameters
    ----------
    X_trn : pd.DataFrame
        Training data (rows = samples, cols = variables).
    X_tst : pd.DataFrame
        Test data (same columns as X_trn, same order).

    Returns
    -------
    pd.DataFrame
        Synthetic data with same index/columns as X_tst.
    """
    # Preallocate DataFrame with object dtype columns
    Z_tst = pd.DataFrame(
        {c: pd.Series(index=X_tst.index, dtype=object) for c in X_tst.columns},
        index=X_tst.index,
    )

    # Progress bar over columns
    it = tqdm(X_tst.columns, desc="Synthesizing columns") if show_progress else X_tst.columns

    for col in it:
        # All features except the target column
        X_trn_minus = X_trn.drop(columns=[col])
        X_tst_minus = X_tst.drop(columns=[col])

        # Predict column `col` using TabPFN
        z_col = generate_prediction_using_tabpfn(
            X_trn=X_trn_minus, X_tst=X_tst_minus, y_trn=X_trn[col]
        )

        # Assign predictions directly into the preallocated column
        Z_tst[col] = pd.Series(np.asarray(z_col).reshape(-1), index=X_tst.index)

    return Z_tst



def full_conditionals_tabpfn_generator(X: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """
    Steps:
      1) Split X rows in half: X1 (first half), X2 (second half).
      2) Train with X2, query with X1 -> Z1 via icl_with_joint_factorization_tabpfn.
      3) Train with X1, query with X2 -> Z2 via icl_with_joint_factorization_tabpfn.
      4) Row-bind Z1 and Z2 in original order.
      5) Round integer-valued columns (detected from X) in Z.

    Parameters
    ----------
    X : pd.DataFrame
        Input data (rows = samples, columns = variables).

    Returns
    -------
    pd.DataFrame
        Synthetic DataFrame with same shape/columns as X.
    """
    n = len(X)

    # Match R's round(n/2) behavior (banker's rounding)
    split = int(np.rint(n / 2.0))

    X1 = X.iloc[:split, :]
    X2 = X.iloc[split:, :]

    #print("train with X2, query with X1")
    Z1 = icl_with_full_conditionals_tabpfn(X_trn=X2, X_tst=X1, show_progress = show_progress)

    #print("train with X1, query with X2")
    Z2 = icl_with_full_conditionals_tabpfn(X_trn=X1, X_tst=X2, show_progress = show_progress)

    # rbind(Z1, Z2) -> vertical concat; preserves the original row split order
    Z = pd.concat([Z1, Z2], axis=0)

    # Round integer-valued variables (detected from X) in the synthetic data
    Z = round_integer_variables(df_ori=X, df_syn=Z)

    return Z




def _categorize_variable(x: np.ndarray, n_levels: int) -> np.ndarray:
    """
    Discretize numeric vector x into ~n_levels quantile bins.
    Returns string labels "0","1",... for each bin (R returns character).
    """
    x = np.asarray(x, dtype=float)
    mask = ~np.isnan(x)
    if n_levels <= 1 or mask.sum() == 0:
        # single bucket (or all-NaN): everyone same class
        out = np.full(x.shape, "0", dtype=object)
        out[~mask] = np.nan
        return out

    # quantile breaks; unique to avoid duplicates
    probs = np.linspace(0.0, 1.0, num=n_levels + 1)
    breaks = np.unique(np.nanquantile(x, probs))
    if breaks.size < 2:
        # constant vector (or degenerate quantiles)
        out = np.full(x.shape, "0", dtype=object)
        out[~mask] = np.nan
        return out

    # pandas.cut gives categorical bins; include lowest to match R include.lowest=TRUE
    # labels: 0..(len(breaks)-2) as strings
    labels = [str(i) for i in range(len(breaks) - 1)]
    bins = pd.cut(x[mask], bins=breaks, labels=labels, include_lowest=True)
    out = np.full(x.shape, np.nan, dtype=object)
    out[mask] = bins.astype(object)
    return out



def icl_with_miav_tabpfn_noisy(
    X_trn: pd.DataFrame,
    X_tst: pd.DataFrame,
    percent: float,
    random_state: Optional[int] = None,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    For each column j:
        - compute MIAV on train/test columns,
        - add Gaussian noise with sd = percent * sd(MIAV),
        - predict test column via TabPFN using the 1-D noisy MIAV as feature.

    Parameters
    ----------
    X_trn, X_tst : pd.DataFrame
        Train/test tables (same schema).
    percent : float
        Noise scale factor; noise_sd = percent * sd(miav).
        Use e.g. 0.05 to mimic R's rnorm(..., 0, 0.05*sd(x)).
    random_state : int | None
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Z_tst with same columns/index as X_tst.
    """
    rng = np.random.default_rng(random_state)

    # Preallocate with object dtype to avoid dtype-compat warnings across mixed types
    Z_tst = pd.DataFrame({c: pd.Series(index=X_tst.index, dtype=object) for c in X_tst.columns})

    it = tqdm(X_tst.columns, desc="Synthesizing columns") if show_progress else X_tst.columns

    for col in it:

        # --- Train MIAV + noise ---
        m_trn = generate_maximal_information_auxiliary_variable(X_trn[col])
        m_trn = np.asarray(m_trn, dtype=float)
        if percent > 0:
            sd_trn = np.nanstd(m_trn, ddof=1)  # R's sd uses ddof=1
            if np.isfinite(sd_trn) and sd_trn > 0:
                m_trn = m_trn + rng.normal(0.0, percent * sd_trn, size=m_trn.shape)
        m_trn_df = pd.DataFrame({"m": m_trn}, index=X_trn.index)

        # --- Test MIAV + noise ---
        m_tst = generate_maximal_information_auxiliary_variable(X_tst[col])
        m_tst = np.asarray(m_tst, dtype=float)
        if percent > 0:
            sd_tst = np.nanstd(m_tst, ddof=1)
            if np.isfinite(sd_tst) and sd_tst > 0:
                m_tst = m_tst + rng.normal(0.0, percent * sd_tst, size=m_tst.shape)
        m_tst_df = pd.DataFrame({"m": m_tst}, index=X_tst.index)

        # --- TabPFN prediction (target = original train column) ---
        y_trn = X_trn[col]
        z_col = generate_prediction_using_tabpfn(X_trn=m_trn_df, X_tst=m_tst_df, y_trn=y_trn)

        Z_tst[col] = np.asarray(z_col).reshape(-1)

    return Z_tst



def noisy_miav_tabpfn_generator(X: pd.DataFrame, percent: float = 0.0, show_progress: bool = False) -> pd.DataFrame:
    """
    Split X in half by row order. Train on one half, query on the other (both ways).
    Concatenate predictions and round integer-valued columns based on X.
    """
    n = len(X)
    split = int(np.round(n / 2.0))
    X1 = X.iloc[:split, :].copy()
    X2 = X.iloc[split:, :].copy()

    #print("train with X2, query with X1")
    Z1 = icl_with_miav_tabpfn_noisy(X_trn=X2, X_tst=X1, percent=percent, show_progress = show_progress)

    #print("train with X1, query with X2")
    Z2 = icl_with_miav_tabpfn_noisy(X_trn=X1, X_tst=X2, percent=percent, show_progress = show_progress)

    # Stack back to original order (first rows of X1, then X2)
    Z = pd.concat([Z1, Z2], axis=0, ignore_index=True)

    # Round integer-valued variables
    Z = round_integer_variables(df_ori=X, df_syn=Z)

    return Z




