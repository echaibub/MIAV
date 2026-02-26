import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm, beta
from typing import Sequence, Optional, Union

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm


def synth_smotenc(
    dat: pd.DataFrame,
    k: int,
    round_int_vars: bool = True
) -> pd.DataFrame:
    """
    Implements a simple SMOTE-like synthesizer:
      - k-NN is computed in the space of numeric columns only.
      - Numeric columns are synthesized by interpolation with one randomly
        chosen neighbor among the k neighbors.
      - Categorical / boolean / string columns are synthesized by majority vote
        across {sample} U {its k numeric-nearest neighbors}.

    Parameters
    ----------
    dat : pd.DataFrame
        Input data.
    k : int
        Number of nearest neighbors.
    round_int_vars : bool, default True
        If True, round back columns that are integer-valued in `dat`.
    atol : float, default 1e-8
        Tolerance to decide if an original numeric column is integer-like.

    Returns
    -------
    pd.DataFrame
        Synthetic data with the same shape/columns/index as `dat`.
    """

    # ---- Identify numeric vs. categorical-like columns (like GetVariableTypes) ----
    num_cols = dat.select_dtypes(include="number").columns.tolist()
    cat_cols = dat.select_dtypes(include=["category", "object", "string", "bool", "boolean"]).columns.tolist()

    n = len(dat)
    if len(num_cols) == 0 and len(cat_cols) == 0:
        # Nothing to do
        return dat.copy()

    # ---- kNN on numeric columns (if any) ----
    if len(num_cols) > 0:
        X_num = dat[num_cols].to_numpy(dtype=float)
        # sklearn kneighbors on the same data includes self as the first neighbor;
        # we ask for k+1 and then drop the self neighbor to match R's FNN::get.knn behavior.
        nbrs = NearestNeighbors(n_neighbors=min(k + 1, max(1, n)), algorithm="auto")
        nbrs.fit(X_num)
        distances, indices = nbrs.kneighbors(X_num, return_distance=True)
        # Drop self (first column) if present
        if indices.shape[1] > 1:
            nn_index = indices[:, 1:]  # shape: (n, k) when n > 1 and k >= 1
        else:
            # Degenerate case: only one neighbor (itself)
            nn_index = np.zeros((n, 0), dtype=int)
    else:
        nn_index = np.zeros((n, 0), dtype=int)

    # ---- Helpers ----
    def generate_synth_num_data(dat_df: pd.DataFrame, nn_idx: np.ndarray, num_variables: list[str]) -> pd.DataFrame:
        """Interpolate numeric columns toward a random neighbor among k."""
        if len(num_variables) == 0:
            return dat_df.copy()

        out = dat_df.copy()
        Xn = out[num_variables].to_numpy(dtype=float)
        for i in range(n):
            if nn_idx.shape[1] == 0:  # no neighbors
                continue
            # pick one neighbor index uniformly at random
            j_idx = np.random.randint(nn_idx.shape[1])
            nb = nn_idx[i, j_idx]
            lam = np.random.rand()
            # x + lambda * (x_nn - x)
            Xn[i, :] = Xn[i, :] + lam * (Xn[nb, :] - Xn[i, :])
        out[num_variables] = Xn
        return out

    def majority_vote(values: list[str]) -> str:
        """Majority vote with a deterministic tie-break (lexicographic)."""
        s = pd.Series(values, dtype="string")
        counts = s.value_counts(dropna=False)
        # tie-break by label lexicographic to mimic R's sort(table(...), decreasing=TRUE)
        top_count = counts.max()
        candidates = sorted([idx for idx, c in counts.items() if c == top_count], key=lambda x: str(x))
        return candidates[0]

    def generate_synth_cat_data(dat_df: pd.DataFrame, nn_idx: np.ndarray, cat_variables: list[str]) -> pd.DataFrame:
        """Majority vote across sample + its k neighbors for each categorical column."""
        if len(cat_variables) == 0:
            return dat_df.copy()

        out = dat_df.copy()
        for i in range(n):
            # neighbor indices (could be empty if no numeric columns)
            neigh = nn_idx[i, :].tolist() if nn_idx.shape[1] else []
            # for each categorical column, vote among sample + neighbors
            for col in cat_variables:
                vals = [dat_df.iloc[i][col]]
                if neigh:
                    vals.extend(dat_df.iloc[neigh][col].astype("string").tolist())
                out.at[out.index[i], col] = majority_vote(vals)
        return out

    # ---- Generate numeric-only and categorical-only synthetic datasets ----
    synth_num_dat = generate_synth_num_data(dat, nn_index, num_cols)
    synth_cat_dat = generate_synth_cat_data(dat, nn_index, cat_cols)

    # ---- Combine: numeric from synth_num_dat + categorical from synth_cat_dat ----
    synthetic_dat = synth_num_dat.copy()
    if len(cat_cols) > 0:
        synthetic_dat[cat_cols] = synth_cat_dat[cat_cols]

    # ---- Optionally round back integer-valued numeric columns (like RoundIntegerVariables) ----
    if round_int_vars and len(num_cols) > 0:
        synthetic_dat = round_integer_variables(df_ori=dat, df_syn=synthetic_dat)

    # Preserve original column order/dtypes 
    synthetic_dat = synthetic_dat.astype(
        {
            c: dat[c].dtype
            for c in synthetic_dat.columns
            if isinstance(dat[c].dtype, pd.CategoricalDtype)
        },
        copy=False,
    )
    
    return synthetic_dat




def simulate_correlated_beta_data(
    n: int,
    rho: float,
    beta_pars_list: Sequence[Sequence[float]],
    *,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> pd.DataFrame:
    """
    Simulate an n x p DataFrame with Beta marginals coupled via a Gaussian copula.
    - Correlation matrix is Toeplitz with entries rho**|i-j|.
    - Beta parameters are given as [(a1, b1), (a2, b2), ..., (ap, bp)].

    This mirrors the R:
      SimulateCorrelatedBetaData(n, rho, beta_pars_list)
    """
    p = len(beta_pars_list)
    if p == 0:
        return pd.DataFrame(index=np.arange(n))

    # Toeplitz correlation Sigma[i,j] = rho^|i-j|  (valid for |rho| < 1)
    idx = np.arange(p)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])

    # Random generator
    rng = np.random.default_rng(random_state)

    # Sample correlated normals: Z ~ N(0, I), X = Z @ L^T where Sigma = L L^T
    L = np.linalg.cholesky(Sigma)              # lower-triangular
    Z = rng.normal(size=(n, p))                # shape: (n, p)
    X = Z @ L.T                                # correlated normals, (n, p)

    # Map to uniforms via Gaussian CDF (copula)
    U = norm.cdf(X)
    # Guard against numerical 0/1 at extreme tails before PPF
    eps = np.finfo(float).eps
    U = np.clip(U, eps, 1 - eps)

    # Map uniforms to Beta via inverse CDF (PPF)
    out = np.empty_like(U)
    for j, (a, bpar) in enumerate(beta_pars_list):
        out[:, j] = beta.ppf(U[:, j], a, bpar)

    columns = [f"X{j+1}" for j in range(p)]
    return pd.DataFrame(out, columns=columns)




def marginal_density_plots(
    var_idx,
    dat_real: pd.DataFrame,
    dat_synt: pd.DataFrame,
    leg_pos: str = "topright",
    *,
    gridsize: int = 200,
    bw_method: str | float = "scott",   # "scott", "silverman", or a numeric factor
    ax: plt.Axes | None = None,
):
    """
    Plot marginal densities for one variable from real vs synthetic data.

    Parameters
    ----------
    var_idx : int | str
        Column to plot. If int, interpreted as **1-based** (R-style).
        If str, treated as column name.
    dat_real, dat_synt : pd.DataFrame
        Real and synthetic data (same columns).
    leg_pos : {"topright","topleft","bottomright","bottomleft","best"}
        Legend position (R-like names).
    gridsize : int
        Number of points in the evaluation grid.
    bw_method : {"scott","silverman"} or float
        Bandwidth rule for gaussian_kde (float scales Scott’s factor).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates a new figure/axes.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Resolve column
    if isinstance(var_idx, int):
        # R-style 1-based indexing
        col = dat_real.columns[var_idx - 1]
    else:
        col = var_idx

    x_real = pd.to_numeric(dat_real[col], errors="coerce").dropna().to_numpy()
    x_synt = pd.to_numeric(dat_synt[col], errors="coerce").dropna().to_numpy()

    if x_real.size == 0 or x_synt.size == 0:
        raise ValueError(f"Column '{col}' has no numeric data to plot.")

    # Handle zero-variance edge cases by adding tiny jitter (R's density would warn)
    def _safe_kde(x):
        if np.nanstd(x) == 0:
            x = x + np.random.normal(scale=1e-9, size=x.shape)
        return gaussian_kde(x, bw_method=bw_method)

    kde_real = _safe_kde(x_real)
    kde_synt = _safe_kde(x_synt)

    xmin = np.nanmin([x_real.min(), x_synt.min()])
    xmax = np.nanmax([x_real.max(), x_synt.max()])
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        # fallback range
        xmin, xmax = xmin - 1.0, xmax + 1.0

    xs = np.linspace(xmin, xmax, gridsize)
    yr = kde_real(xs)
    ys = kde_synt(xs)

    # y-limits like R: from 0 to the max of both densities
    ax.plot(xs, yr, color="blue", lw=2, label="real")
    ax.plot(xs, ys, color="red",  lw=1.5, label="synth")
    ax.set_ylim(0, max(yr.max(), ys.max()) * 1.05)

    ax.set_xlabel("")
    ax.set_ylabel("density")
    ax.set_title("")

    # Map R legend positions to matplotlib
    loc_map = {
        "topright": "upper right",
        "topleft": "upper left",
        "bottomright": "lower right",
        "bottomleft": "lower left",
        "best": "best",
    }
    ax.legend(loc=loc_map.get(leg_pos, "upper right"))

    return ax



def enforce_dtypes(dat, 
                   num_variables, 
                   cat_variables):
    """
    Enforce "float64" type for numeric variables and "object" type for the
    categorical variables
    Parameters:
        dat (pd.DataFrame): Input data matrix (numeric, categorical, or mixed).
        num_variables (list): Indices of numeric variables.
        cat_variables (list): Indices of categorical variables.

    Returns:
    pd.DataFrame: with transformed data types
    """
    if num_variables is not None and cat_variables is None:
        dat_N = pd.DataFrame(dat.iloc[:, num_variables], dtype = "float64")
        dat = dat_N

    elif num_variables is None and cat_variables is not None:
        dat_C = pd.DataFrame(dat.iloc[:, cat_variables], dtype = "str")
        dat = dat_C

    elif num_variables is not None and cat_variables is not None:
        dat_N = pd.DataFrame(dat.iloc[:, num_variables], dtype = "float64")
        dat_C = pd.DataFrame(dat.iloc[:, cat_variables], dtype = "str")
        dat = pd.concat([dat_N, dat_C], axis=1)
        # Reorder columns to match the order in the original data
        reordered_indices = num_variables + cat_variables
        dat = dat.iloc[:, np.argsort(reordered_indices)]

    else:
        raise ValueError("At least one of num_variables or cat_variables must be specified.")
    
    return dat 


#####################################################################
## functions for running experiments
#####################################################################

def build_all_splits(X, split_seeds, ds_name, task_id):
    """
    Create all data-splits for each dataframe X and return:
      - splits_by_task: {task_id: {"dataset_name": str,
                                   "splits": [{"split": j, "orig": df, "hold": df}, ...]}}
      - splits_long: one big DataFrame where *dataset columns are prefixed*
                     so no cross-dataset name collisions occur.
        Metadata columns: __dataset__, __task_id__, __split__, __role__
    """
    long_chunks = []
    
    # --- generate splits for this dataset ---
    ds_splits = []
    for j, seed in enumerate(split_seeds, start=1):
        X_orig, X_hold = train_test_split(X, test_size=0.5, random_state=seed)

        # remove rows with NA values 
        X_orig = X_orig.dropna().reset_index(drop=True)
        X_hold = X_hold.dropna().reset_index(drop=True)

        # store per-dataset copies in the Python structure
        ds_splits.append({"split": j, "orig": X_orig, "hold": X_hold})

        # add copies to the long-form table
        for role, df in (("orig", X_orig), ("hold", X_hold)):
            chunk = df.copy()
            # metadata columns (placed in front)
            chunk.insert(0, "__role__", role)
            chunk.insert(0, "__split__", j)
            chunk.insert(0, "__task_id__", task_id)
            chunk.insert(0, "__dataset__", ds_name)
            long_chunks.append(chunk)

    splits_long = pd.concat(long_chunks, axis=0, ignore_index=True)
    return splits_long




def build_all_synthetics_jf(
    X: pd.DataFrame,
    split_seeds,
    generator_kwargs=None,
    *,
    task_id: int = 0,
    ds_name: str = "",
):
    """
    For each split seed:
      - take a 50/50 split (use the 'orig' half),
      - drop rows with NA,
      - synthesize with miav_tabpfn_generator,
      - add metadata columns (no column prefixing),
    then stack everything into one long DataFrame.

    Returns
    -------
    syn_long : pd.DataFrame
        Rows from all splits, with metadata columns first:
        [__dataset__, __task_id__, __split__, __role__] + original data columns
    failures : list[dict]
        Any split-level errors: {'task_id', 'split', 'role', 'error'}
    """
    if generator_kwargs is None:
        generator_kwargs = {}

    all_chunks = []
    failures = []

    for j, seed in tqdm(enumerate(split_seeds, start=1), desc='Data split', total=len(split_seeds)):
        try:
            X_orig, _ = train_test_split(X, test_size=0.5, random_state=seed)

            # remove rows with NA values
            X_orig = X_orig.dropna().reset_index(drop=True)
            if X_orig.empty:
                continue

            # generate synthetic copy of the original half
            X_syn = joint_factorization_tabpfn_generator(X_orig, **generator_kwargs)

            # add metadata 
            chunk = X_syn.copy()
            chunk.insert(0, "__role__", "syn")
            chunk.insert(0, "__split__", np.int32(j))
            chunk.insert(0, "__task_id__", np.int32(task_id))
            chunk.insert(0, "__dataset__", str(ds_name))

            all_chunks.append(chunk)

        except Exception as e:
            failures.append({
                "task_id": task_id,
                "split": j,
                "role": "syn",
                "error": repr(e),
            })

    if all_chunks:
        syn_long = pd.concat(all_chunks, axis=0, ignore_index=True)
    else:
        syn_long = pd.DataFrame(columns=["__dataset__", "__task_id__", "__split__", "__role__"])

    return syn_long, failures




def build_all_synthetics_fc(
    X: pd.DataFrame,
    split_seeds,
    generator_kwargs=None,
    *,
    task_id: int = 0,
    ds_name: str = "",
):
    """
    For each split seed:
      - take a 50/50 split (use the 'orig' half),
      - drop rows with NA,
      - synthesize with miav_tabpfn_generator,
      - add metadata columns (no column prefixing),
    then stack everything into one long DataFrame.

    Returns
    -------
    syn_long : pd.DataFrame
        Rows from all splits, with metadata columns first:
        [__dataset__, __task_id__, __split__, __role__] + original data columns
    failures : list[dict]
        Any split-level errors: {'task_id', 'split', 'role', 'error'}
    """
    if generator_kwargs is None:
        generator_kwargs = {}

    all_chunks = []
    failures = []

    for j, seed in tqdm(enumerate(split_seeds, start=1), desc='Data split', total=len(split_seeds)):
        try:
            X_orig, _ = train_test_split(X, test_size=0.5, random_state=seed)

            # remove rows with NA values
            X_orig = X_orig.dropna().reset_index(drop=True)
            if X_orig.empty:
                continue

            # generate synthetic copy of the original half
            X_syn = full_conditionals_tabpfn_generator(X_orig, **generator_kwargs)

            # add metadata 
            chunk = X_syn.copy()
            chunk.insert(0, "__role__", "syn")
            chunk.insert(0, "__split__", np.int32(j))
            chunk.insert(0, "__task_id__", np.int32(task_id))
            chunk.insert(0, "__dataset__", str(ds_name))

            all_chunks.append(chunk)

        except Exception as e:
            failures.append({
                "task_id": task_id,
                "split": j,
                "role": "syn",
                "error": repr(e),
            })

    if all_chunks:
        syn_long = pd.concat(all_chunks, axis=0, ignore_index=True)
    else:
        syn_long = pd.DataFrame(columns=["__dataset__", "__task_id__", "__split__", "__role__"])

    return syn_long, failures





def build_all_synthetics_miav(
    X: pd.DataFrame,
    split_seeds,
    generator_kwargs=None,
    *,
    task_id: int = 0,
    ds_name: str = "",
):
    """
    For each split seed:
      - take a 50/50 split (use the 'orig' half),
      - drop rows with NA,
      - synthesize with miav_tabpfn_generator,
      - add metadata columns (no column prefixing),
    then stack everything into one long DataFrame.

    Returns
    -------
    syn_long : pd.DataFrame
        Rows from all splits, with metadata columns first:
        [__dataset__, __task_id__, __split__, __role__] + original data columns
    failures : list[dict]
        Any split-level errors: {'task_id', 'split', 'role', 'error'}
    """
    if generator_kwargs is None:
        generator_kwargs = {}

    all_chunks = []
    failures = []

    for j, seed in tqdm(enumerate(split_seeds, start=1), desc='Data split', total=len(split_seeds)):
        try:
            X_orig, _ = train_test_split(X, test_size=0.5, random_state=seed)

            # remove rows with NA values
            X_orig = X_orig.dropna().reset_index(drop=True)
            if X_orig.empty:
                continue

            # generate synthetic copy of the original half
            X_syn = miav_tabpfn_generator(X_orig, **generator_kwargs)

            # add metadata 
            chunk = X_syn.copy()
            chunk.insert(0, "__role__", "syn")
            chunk.insert(0, "__split__", np.int32(j))
            chunk.insert(0, "__task_id__", np.int32(task_id))
            chunk.insert(0, "__dataset__", str(ds_name))

            all_chunks.append(chunk)

        except Exception as e:
            failures.append({
                "task_id": task_id,
                "split": j,
                "role": "syn",
                "error": repr(e),
            })

    if all_chunks:
        syn_long = pd.concat(all_chunks, axis=0, ignore_index=True)
    else:
        syn_long = pd.DataFrame(columns=["__dataset__", "__task_id__", "__split__", "__role__"])

    return syn_long, failures



def build_all_synthetics_miav_noisy(
    X: pd.DataFrame,
    split_seeds,
    generator_kwargs=None,
    *,
    task_id: int = 0,
    ds_name: str = "",
    percent: float = 0.0,
    show_progress: bool = False
):
    """
    For each split seed:
      - take a 50/50 split (use the 'orig' half),
      - drop rows with NA,
      - synthesize with miav_tabpfn_generator,
      - add metadata columns (no column prefixing),
    then stack everything into one long DataFrame.

    Returns
    -------
    syn_long : pd.DataFrame
        Rows from all splits, with metadata columns first:
        [__dataset__, __task_id__, __split__, __role__] + original data columns
    failures : list[dict]
        Any split-level errors: {'task_id', 'split', 'role', 'error'}
    """
    if generator_kwargs is None:
        generator_kwargs = {}

    all_chunks = []
    failures = []

    for j, seed in tqdm(enumerate(split_seeds, start=1), desc='Data split', total=len(split_seeds)):
        try:
            X_orig, _ = train_test_split(X, test_size=0.5, random_state=seed)

            # remove rows with NA values
            X_orig = X_orig.dropna().reset_index(drop=True)
            if X_orig.empty:
                continue

            # generate synthetic copy of the original half
            X_syn = noisy_miav_tabpfn_generator(X_orig, percent = percent, show_progress = show_progress, **generator_kwargs)

            # add metadata
            chunk = X_syn.copy()
            chunk.insert(0, "__role__", "syn")
            chunk.insert(0, "__split__", np.int32(j))
            chunk.insert(0, "__task_id__", np.int32(task_id))
            chunk.insert(0, "__dataset__", str(ds_name))

            all_chunks.append(chunk)

        except Exception as e:
            failures.append({
                "task_id": task_id,
                "split": j,
                "role": "syn",
                "error": repr(e),
            })

    if all_chunks:
        syn_long = pd.concat(all_chunks, axis=0, ignore_index=True)
    else:
        syn_long = pd.DataFrame(columns=["__dataset__", "__task_id__", "__split__", "__role__"])

    return syn_long, failures



#################################################################
## These functions generate the long feather file for multiple
## datasets in the AuttoML-CC18 benchmark suite
#################################################################

def build_all_splits_tasks(tasks_to_keep, split_seeds):
    """
    Create all data-splits for each OpenML task and return:
      - splits_by_task: {task_id: {"dataset_name": str,
                                   "splits": [{"split": j, "orig": df, "hold": df}, ...]}}
      - splits_long: one big DataFrame where *dataset columns are prefixed*
                     so no cross-dataset name collisions occur.
        Metadata columns: __dataset__, __task_id__, __split__, __role__
    """
    splits_by_task = {}
    long_chunks = []

    for i, tsk in tqdm(enumerate(tasks_to_keep, start=1),
                       desc="Dataset", total=len(tasks_to_keep)):
        # --- fetch dataset ---
        task = openml.tasks.get_task(tsk)
        dataset = task.get_dataset()
        ds_name = dataset.name

        # ALL columns together (features + target)
        X, y, categorical_mask, attr_names = dataset.get_data(
            target=None, dataset_format="dataframe"
        )

        # Per-dataset column prefix (prevents collisions across datasets)
        pref = _safe_prefix(ds_name, tsk)

        # --- generate splits for this dataset ---
        ds_splits = []
        for j, seed in enumerate(split_seeds, start=1):
            X_orig, X_hold = train_test_split(X, test_size=0.5, random_state=seed)

            # remove rows with NA values 
            X_orig = X_orig.dropna().reset_index(drop=True)
            X_hold = X_hold.dropna().reset_index(drop=True)

            # store per-dataset (unprefixed) copies in the Python structure
            ds_splits.append({"split": j, "orig": X_orig, "hold": X_hold})

            # add *prefixed* copies to the long-form table
            for role, df in (("orig", X_orig), ("hold", X_hold)):
                df_pref = df.copy()
                df_pref.columns = [pref + c for c in df_pref.columns]  # e.g., 31__adult__age
                chunk = df_pref
                # metadata columns (placed in front)
                chunk.insert(0, "__role__", role)
                chunk.insert(0, "__split__", j)
                chunk.insert(0, "__task_id__", tsk)
                chunk.insert(0, "__dataset__", ds_name)
                long_chunks.append(chunk)

        splits_by_task[tsk] = {"dataset_name": ds_name, "splits": ds_splits}

    splits_long = pd.concat(long_chunks, axis=0, ignore_index=True)
    return splits_by_task, splits_long



def summarize_openml_tasks(tasks_to_keep: list[int]) -> pd.DataFrame:
    """
    Build a summary table for a list of OpenML task IDs.

    Returns a DataFrame with columns:
      - task_id
      - dataset_name
      - n_rows
      - n_cols
      - n_categorical               (# of non-numeric columns)
      - max_categorical_levels      (max # of levels among those non-numeric columns)
    """
    rows = []

    # Iterate over tasks with a progress bar
    for tsk in tqdm(tasks_to_keep, desc="Summarizing tasks"):
        try:
            # Load the task and its dataset
            task = openml.tasks.get_task(tsk)
            dataset = task.get_dataset()

            # Get ALL columns (features + target) together as a single DataFrame
            X, y, categorical_mask, attr_names = dataset.get_data(
                target=None, dataset_format="dataframe"
            )

            # Basic shape
            n_rows, n_cols = X.shape

            # Identify non-numeric columns (treat as categorical-like)
            non_num_cols = X.select_dtypes(exclude="number").columns
            n_categorical = len(non_num_cols)

            # Max number of levels among the categorical-like columns
            if n_categorical > 0:
                # nunique(dropna=True) ignores NaNs when counting levels
                max_classes = int(X[non_num_cols].nunique(dropna=True).max())
            else:
                max_classes = 0  # no categorical columns

            # Append one row for this dataset/task
            rows.append(
                {
                    "task_id": tsk,
                    "dataset_name": dataset.name,   # OpenML dataset name
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "n_categorical": n_categorical,
                    "max_categorical_levels": max_classes,
                }
            )

        except Exception as e:
            # If anything goes wrong, skip and log a short message
            tqdm.write(f"[skip task {tsk}] {e!r}")

    # Build the final table
    df_summary = pd.DataFrame(
        rows,
        columns=[
            "dataset_name",
            "n_rows",
            "n_cols",
            "n_categorical",
            "max_categorical_levels",
            "task_id",
        ],
    )

    df_summary = df_summary.sort_values(["task_id"], ignore_index=True)
    return df_summary


# This is needed in case some of the datasets have columns with identical names
def _safe_prefix(ds_name: str, task_id: int) -> str:
    """Column-safe prefix: <task_id>__<dataset_name>__ (non-word chars -> '_')."""
    safe = re.sub(r"\W+", "_", str(ds_name)).strip("_")
    return f"{task_id}__{safe}__"



def build_all_synthetics_jf_tasks(
    tasks_to_keep,
    split_seeds,
    generator_kwargs=None,
    *,
    show_progress: bool = True,
):
    """
    Synthesize from X_orig only for each (task_id, split).
    Returns:
      syn_long  : big DataFrame with metadata cols [__dataset__, __task_id__, __split__, __role__='syn']
                  and per-dataset *prefixed* data columns
      failures  : list of {task_id, split, role, error}
    """
    if generator_kwargs is None:
        generator_kwargs = {}

    all_chunks: list[pd.DataFrame] = []
    failures: list[dict] = []

    total_steps = len(tasks_to_keep) * len(split_seeds)
    pbar = tqdm(total=total_steps, desc="Synthesizing (datasets × splits)", unit="split") if show_progress else None

    for i, tsk in enumerate(tasks_to_keep, start=1):
        # fetch dataset once
        task = openml.tasks.get_task(tsk)
        dataset = task.get_dataset()
        ds_name = dataset.name
        pref = _safe_prefix(ds_name, tsk)

        # load all columns (features + target)
        X, y, categorical_mask, attr_names = dataset.get_data(
            target=None, dataset_format="dataframe"
        )

        ds_chunks = []
        ds_fail = 0

        for j, seed in enumerate(split_seeds, start=1):
            try:
                X_orig, _ = train_test_split(X, test_size=0.5, random_state=seed)

                # remove rows with NA values
                X_orig = X_orig.dropna().reset_index(drop=True)
                
                X_syn = joint_factorization_tabpfn_generator(X_orig)

                chunk = X_syn.copy()
                chunk.columns = [pref + c for c in chunk.columns]
                chunk.insert(0, "__role__", "syn")
                chunk.insert(0, "__split__", np.int32(j))
                chunk.insert(0, "__task_id__", np.int32(tsk))
                chunk.insert(0, "__dataset__", str(ds_name))

                ds_chunks.append(chunk)
                all_chunks.append(chunk)

            except Exception as e:
                ds_fail += 1
                failures.append({"task_id": tsk, "split": j, "role": "syn", "error": repr(e)})
                if pbar:
                    pbar.write(f"[ERROR] task_id={tsk} split={j}: {type(e).__name__}: {e}")
            finally:
                # advance the single progress bar by 1 split no matter what
                if pbar:
                    pbar.set_postfix(task_id=tsk, split=j, ds=ds_name[:24], refresh=False)
                    pbar.update(1)

    if pbar:
        pbar.close()

    syn_long = pd.concat(all_chunks, axis=0, ignore_index=True) if all_chunks else \
               pd.DataFrame(columns=["__dataset__", "__task_id__", "__split__", "__role__"])
    return syn_long, failures




def build_all_synthetics_fc_tasks(
    tasks_to_keep,
    split_seeds,
    generator_kwargs=None,
    *,
    show_progress: bool = True,
):
    """
    Synthesize from X_orig only for each (task_id, split).
    Returns:
      syn_long  : big DataFrame with metadata cols [__dataset__, __task_id__, __split__, __role__='syn']
                  and per-dataset *prefixed* data columns
      failures  : list of {task_id, split, role, error}
    """
    if generator_kwargs is None:
        generator_kwargs = {}

    all_chunks: list[pd.DataFrame] = []
    failures: list[dict] = []

    total_steps = len(tasks_to_keep) * len(split_seeds)
    pbar = tqdm(total=total_steps, desc="Synthesizing (datasets × splits)", unit="split") if show_progress else None

    for i, tsk in enumerate(tasks_to_keep, start=1):
        # fetch dataset once
        task = openml.tasks.get_task(tsk)
        dataset = task.get_dataset()
        ds_name = dataset.name
        pref = _safe_prefix(ds_name, tsk)

        # load all columns (features + target)
        X, y, categorical_mask, attr_names = dataset.get_data(
            target=None, dataset_format="dataframe"
        )

        ds_chunks = []
        ds_fail = 0

        for j, seed in enumerate(split_seeds, start=1):
            try:
                X_orig, _ = train_test_split(X, test_size=0.5, random_state=seed)

                # remove rows with NA values
                X_orig = X_orig.dropna().reset_index(drop=True)
                
                X_syn = full_conditionals_tabpfn_generator(X_orig)

                chunk = X_syn.copy()
                chunk.columns = [pref + c for c in chunk.columns]
                chunk.insert(0, "__role__", "syn")
                chunk.insert(0, "__split__", np.int32(j))
                chunk.insert(0, "__task_id__", np.int32(tsk))
                chunk.insert(0, "__dataset__", str(ds_name))

                ds_chunks.append(chunk)
                all_chunks.append(chunk)

            except Exception as e:
                ds_fail += 1
                failures.append({"task_id": tsk, "split": j, "role": "syn", "error": repr(e)})
                if pbar:
                    pbar.write(f"[ERROR] task_id={tsk} split={j}: {type(e).__name__}: {e}")
            finally:
                # advance the single progress bar by 1 split no matter what
                if pbar:
                    pbar.set_postfix(task_id=tsk, split=j, ds=ds_name[:24], refresh=False)
                    pbar.update(1)

    if pbar:
        pbar.close()

    syn_long = pd.concat(all_chunks, axis=0, ignore_index=True) if all_chunks else \
               pd.DataFrame(columns=["__dataset__", "__task_id__", "__split__", "__role__"])
    return syn_long, failures




def build_all_synthetics_miav_tasks(
    tasks_to_keep,
    split_seeds,
    generator_kwargs=None,
    *,
    show_progress: bool = True,
):
    """
    Synthesize from X_orig only for each (task_id, split).
    Returns:
      syn_long  : big DataFrame with metadata cols [__dataset__, __task_id__, __split__, __role__='syn']
                  and per-dataset *prefixed* data columns
      failures  : list of {task_id, split, role, error}
    """
    if generator_kwargs is None:
        generator_kwargs = {}

    all_chunks: list[pd.DataFrame] = []
    failures: list[dict] = []

    total_steps = len(tasks_to_keep) * len(split_seeds)
    pbar = tqdm(total=total_steps, desc="Synthesizing (datasets × splits)", unit="split") if show_progress else None

    for i, tsk in enumerate(tasks_to_keep, start=1):
        # fetch dataset once
        task = openml.tasks.get_task(tsk)
        dataset = task.get_dataset()
        ds_name = dataset.name
        pref = _safe_prefix(ds_name, tsk)

        # load all columns (features + target)
        X, y, categorical_mask, attr_names = dataset.get_data(
            target=None, dataset_format="dataframe"
        )

        ds_chunks = []
        ds_fail = 0

        for j, seed in enumerate(split_seeds, start=1):
            try:
                X_orig, _ = train_test_split(X, test_size=0.5, random_state=seed)

                # remove rows with NA values
                X_orig = X_orig.dropna().reset_index(drop=True)
                
                X_syn = miav_tabpfn_generator(X_orig)

                chunk = X_syn.copy()
                chunk.columns = [pref + c for c in chunk.columns]
                chunk.insert(0, "__role__", "syn")
                chunk.insert(0, "__split__", np.int32(j))
                chunk.insert(0, "__task_id__", np.int32(tsk))
                chunk.insert(0, "__dataset__", str(ds_name))

                ds_chunks.append(chunk)
                all_chunks.append(chunk)

            except Exception as e:
                ds_fail += 1
                failures.append({"task_id": tsk, "split": j, "role": "syn", "error": repr(e)})
                if pbar:
                    pbar.write(f"[ERROR] task_id={tsk} split={j}: {type(e).__name__}: {e}")
            finally:
                # advance the single progress bar by 1 split no matter what
                if pbar:
                    pbar.set_postfix(task_id=tsk, split=j, ds=ds_name[:24], refresh=False)
                    pbar.update(1)

    if pbar:
        pbar.close()

    syn_long = pd.concat(all_chunks, axis=0, ignore_index=True) if all_chunks else \
               pd.DataFrame(columns=["__dataset__", "__task_id__", "__split__", "__role__"])
    return syn_long, failures



def build_all_synthetics_noisy_miav_tasks(
    tasks_to_keep,
    split_seeds,
    generator_kwargs=None,
    *,
    show_progress: bool = True,
    percent: float=0.0
):
    """
    Synthesize from X_orig only for each (task_id, split).
    Returns:
      syn_long  : big DataFrame with metadata cols [__dataset__, __task_id__, __split__, __role__='syn']
                  and per-dataset *prefixed* data columns
      failures  : list of {task_id, split, role, error}
    """
    if generator_kwargs is None:
        generator_kwargs = {}

    all_chunks: list[pd.DataFrame] = []
    failures: list[dict] = []

    total_steps = len(tasks_to_keep) * len(split_seeds)
    pbar = tqdm(total=total_steps, desc="Synthesizing (datasets × splits)", unit="split") if show_progress else None

    for i, tsk in enumerate(tasks_to_keep, start=1):
        # fetch dataset once
        task = openml.tasks.get_task(tsk)
        dataset = task.get_dataset()
        ds_name = dataset.name
        pref = _safe_prefix(ds_name, tsk)

        # load all columns (features + target)
        X, y, categorical_mask, attr_names = dataset.get_data(
            target=None, dataset_format="dataframe"
        )

        ds_chunks = []
        ds_fail = 0

        for j, seed in enumerate(split_seeds, start=1):
            try:
                X_orig, _ = train_test_split(X, test_size=0.5, random_state=seed)

                # remove rows with NA values
                X_orig = X_orig.dropna().reset_index(drop=True)
                
                X_syn = noisy_miav_tabpfn_generator(X = X_orig, percent = percent)

                chunk = X_syn.copy()
                chunk.columns = [pref + c for c in chunk.columns]
                chunk.insert(0, "__role__", "syn")
                chunk.insert(0, "__split__", np.int32(j))
                chunk.insert(0, "__task_id__", np.int32(tsk))
                chunk.insert(0, "__dataset__", str(ds_name))

                ds_chunks.append(chunk)
                all_chunks.append(chunk)

            except Exception as e:
                ds_fail += 1
                failures.append({"task_id": tsk, "split": j, "role": "syn", "error": repr(e)})
                if pbar:
                    pbar.write(f"[ERROR] task_id={tsk} split={j}: {type(e).__name__}: {e}")
            finally:
                # advance the single progress bar by 1 split no matter what
                if pbar:
                    pbar.set_postfix(task_id=tsk, split=j, ds=ds_name[:24], refresh=False)
                    pbar.update(1)

    if pbar:
        pbar.close()

    syn_long = pd.concat(all_chunks, axis=0, ignore_index=True) if all_chunks else \
               pd.DataFrame(columns=["__dataset__", "__task_id__", "__split__", "__role__"])
    return syn_long, failures



#####################################################################
## Analogous functions for TabICL (which currently only handle 
## datasets containing only categorical variables)
#####################################################################

from tabicl import TabICLClassifier

def generate_prediction_using_tabicl(X_trn, X_tst, y_trn):
    
    # Determine target data type
    target_type = y_trn.dtype

    # If target type is numeric, fit TabPFN regression
    if pd.api.types.is_numeric_dtype(target_type):
        # return a not implemented message
        raise NotImplementedError("Regression is currently not implemented.")

    # Otherwise, fit TabPFN classification
    if not pd.api.types.is_numeric_dtype(target_type):
        # Initialize the classifier
        clf = TabICLClassifier()
        clf.fit(X_trn, y_trn)
        # Predict on the test set
        predictions = clf.predict(X_tst)
    
    return predictions



def icl_with_miav_tabicl(X_trn: pd.DataFrame, X_tst: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """
    This function uses, for each column j:
      1) A maximal-information auxiliary variable (MIAV) computed from y_trn = X_trn[j]
         and y_tst = X_tst[j].
      2) A TabICL-based predictor trained on the MIAV feature (1 column "m") with y_trn
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

        # Predict using TabICL on MIAV with y_trn as targets
        pred = generate_prediction_using_tabicl(X_trn=m_trn_df, X_tst=m_tst_df, y_trn=X_trn[col])

        # Fill synthetic column
        Z_tst[col] = np.asarray(pred).reshape(-1)

    return Z_tst



def miav_tabicl_generator(X: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """
    Steps (matching the R version):
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

    # X1 = first half; X2 = second half (like idx1 <- seq(round(n/2)); X[-idx1,])
    X1 = X.iloc[:split, :]
    X2 = X.iloc[split:, :]

    #print("train with X2, query with X1")
    Z1 = icl_with_miav_tabicl(X_trn=X2, X_tst=X1, show_progress = show_progress)

    #print("train with X1, query with X2")
    Z2 = icl_with_miav_tabicl(X_trn=X1, X_tst=X2, show_progress = show_progress)

    # rbind(Z1, Z2) → vertical concat; this preserves the original row order here
    Z = pd.concat([Z1, Z2], axis=0)

    # Round integer-valued columns (detected from X) in the synthetic data Z
    Z = round_integer_variables(df_ori=X, df_syn=Z)

    return Z




def icl_with_joint_factorization_tabicl(X_trn: pd.DataFrame, X_tst: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
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
    z_col = generate_prediction_using_tabicl(X_trn=X0_trn, X_tst=X0_tst, y_trn=x_trn_j)
    Z_tst.iloc[:, 0] = np.asarray(z_col).reshape(-1)

    # ----- j = 2..p (use first j-1 ORIGINAL columns as features) -----

    it = tqdm(range(2, p + 1), desc="Synthesizing columns") if show_progress else range(2, p + 1)

    for j in it:
        feats_trn = X_trn.iloc[:, : (j - 1)].copy()
        feats_tst = X_tst.iloc[:, : (j - 1)].copy()

        x_trn_j = X_trn.iloc[:, j - 1]  # column j (0-based)
        z_col = generate_prediction_using_tabicl(X_trn=feats_trn, X_tst=feats_tst, y_trn=x_trn_j)
        Z_tst.iloc[:, j - 1] = np.asarray(z_col).reshape(-1)

    return Z_tst



def joint_factorization_tabicl_generator(X: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
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
    Z1 = icl_with_joint_factorization_tabicl(X_trn=X2, X_tst=X1, show_progress = show_progress)

    #print("train with X1, query with X2")
    Z2 = icl_with_joint_factorization_tabicl(X_trn=X1, X_tst=X2, show_progress = show_progress)

    # rbind(Z1, Z2) -> vertical concat; preserves the original row split order
    Z = pd.concat([Z1, Z2], axis=0)

    # Round integer-valued variables (detected from X) in the synthetic data
    Z = round_integer_variables(df_ori=X, df_syn=Z)

    return Z



def icl_with_full_conditionals_tabicl(X_trn: pd.DataFrame, X_tst: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
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
        z_col = generate_prediction_using_tabicl(
            X_trn=X_trn_minus, X_tst=X_tst_minus, y_trn=X_trn[col]
        )

        # Assign predictions directly into the preallocated column
        Z_tst[col] = pd.Series(np.asarray(z_col).reshape(-1), index=X_tst.index)

    return Z_tst



def full_conditionals_tabicl_generator(X: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
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
    Z1 = icl_with_full_conditionals_tabicl(X_trn=X2, X_tst=X1, show_progress = show_progress)

    #print("train with X1, query with X2")
    Z2 = icl_with_full_conditionals_tabicl(X_trn=X1, X_tst=X2, show_progress = show_progress)

    # rbind(Z1, Z2) -> vertical concat; preserves the original row split order
    Z = pd.concat([Z1, Z2], axis=0)

    # Round integer-valued variables (detected from X) in the synthetic data
    Z = round_integer_variables(df_ori=X, df_syn=Z)

    return Z



def grab_dataset(df: pd.DataFrame, task_id, split_idx, role) -> pd.DataFrame:
    """
    Extract one dataset (task_id, split_idx, role) from a long table with
    metadata columns on the left and safe-prefixed feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Long table containing many datasets. Must include columns:
        ["__dataset__", "__task_id__", "__split__", "__role__"].
    task_id : int | str
    split_idx : int | str
    role : str
        e.g., "orig", "hold", "syn", etc.

    Returns
    -------
    pd.DataFrame
        Sub-dataframe with only the feature columns (metadata removed),
        and columns that are all-NA dropped.
    """
    meta_cols = ["__dataset__", "__task_id__", "__split__", "__role__"]

    # Filter rows for the requested dataset/split/role
    idx = (
        (df["__task_id__"] == task_id) &
        (df["__split__"]   == split_idx) &
        (df["__role__"]    == role)
    )
    df_sub = df.loc[idx].copy()

    if df_sub.empty:
        # Return an empty frame with no feature columns
        return pd.DataFrame(index=df_sub.index)

    # Drop metadata columns (by name, robust to column order)
    df_sub = df_sub.drop(columns=[c for c in meta_cols if c in df_sub.columns])

    # Drop columns that are entirely NA (equivalent to the R apply/sum==nrow)
    df_sub = df_sub.dropna(axis=1, how="all")

    return df_sub



def build_all_splits_tasks_cat(tasks_to_keep, split_seeds):
    """
    Create all data-splits for each OpenML task and return:
      - splits_by_task: {task_id: {"dataset_name": str,
                                   "splits": [{"split": j, "orig": df, "hold": df}, ...]}}
      - splits_long: one big DataFrame where *dataset columns are prefixed*
                     so no cross-dataset name collisions occur.
        Metadata columns: __dataset__, __task_id__, __split__, __role__
    """
    splits_by_task = {}
    long_chunks = []

    for i, tsk in tqdm(enumerate(tasks_to_keep, start=1),
                       desc="Dataset", total=len(tasks_to_keep)):
        # --- fetch dataset ---
        task = openml.tasks.get_task(tsk)
        dataset = task.get_dataset()
        ds_name = dataset.name

        # ALL columns together (features + target)
        X, y, categorical_mask, attr_names = dataset.get_data(
            target=None, dataset_format="dataframe"
        )

        # Per-dataset column prefix (prevents collisions across datasets)
        pref = _safe_prefix(ds_name, tsk)

        # --- generate splits for this dataset ---
        ds_splits = []
        for j, seed in enumerate(split_seeds, start=1):
            X_orig, X_hold = train_test_split(X, test_size=0.5, random_state=seed)

            # remove rows with NA values 
            X_orig = X_orig.dropna().reset_index(drop=True)
            X_hold = X_hold.dropna().reset_index(drop=True)

            # remove numerical columns
            X_orig = X_orig.select_dtypes(exclude="number")
            X_hold = X_hold.select_dtypes(exclude="number")

            # store per-dataset (unprefixed) copies in the Python structure
            ds_splits.append({"split": j, "orig": X_orig, "hold": X_hold})

            # add *prefixed* copies to the long-form table
            for role, df in (("orig", X_orig), ("hold", X_hold)):
                df_pref = df.copy()
                df_pref.columns = [pref + c for c in df_pref.columns]  # e.g., 31__adult__age
                chunk = df.copy()
                # metadata columns (placed in front)
                chunk.insert(0, "__role__", role)
                chunk.insert(0, "__split__", j)
                chunk.insert(0, "__task_id__", tsk)
                chunk.insert(0, "__dataset__", ds_name)
                long_chunks.append(chunk)

        splits_by_task[tsk] = {"dataset_name": ds_name, "splits": ds_splits}

    splits_long = pd.concat(long_chunks, axis=0, ignore_index=True)
    return splits_by_task, splits_long




def build_all_synthetics_jf_tasks_cat(
    tasks_to_keep,
    split_seeds,
    pfn_method,
    generator_kwargs=None,
    *,
    show_progress: bool = True,
):
    """
    Synthesize from X_orig only for each (task_id, split).
    Returns:
      syn_long  : big DataFrame with metadata cols [__dataset__, __task_id__, __split__, __role__='syn']
                  and per-dataset *prefixed* data columns
      failures  : list of {task_id, split, role, error}
    """
    if generator_kwargs is None:
        generator_kwargs = {}

    all_chunks: list[pd.DataFrame] = []
    failures: list[dict] = []

    total_steps = len(tasks_to_keep) * len(split_seeds)
    pbar = tqdm(total=total_steps, desc="Synthesizing (datasets × splits)", unit="split") if show_progress else None

    for i, tsk in enumerate(tasks_to_keep, start=1):
        # fetch dataset once
        task = openml.tasks.get_task(tsk)
        dataset = task.get_dataset()
        ds_name = dataset.name
        pref = _safe_prefix(ds_name, tsk)

        # load all columns (features + target)
        X, y, categorical_mask, attr_names = dataset.get_data(
            target=None, dataset_format="dataframe"
        )

        ds_chunks = []
        ds_fail = 0

        for j, seed in enumerate(split_seeds, start=1):
            try:
                X_orig, _ = train_test_split(X, test_size=0.5, random_state=seed)

                # remove rows with NA values
                X_orig = X_orig.dropna().reset_index(drop=True)

                # remove numerical columns
                X_orig = X_orig.select_dtypes(exclude="number")
                
                # select pfn method
                if pfn_method == "tabpfn":
                    X_syn = joint_factorization_tabpfn_generator(X_orig)

                if pfn_method == "tabicl":
                    X_syn = joint_factorization_tabicl_generator(X_orig)

                chunk = X_syn.copy()
                chunk.columns = [pref + c for c in chunk.columns]
                chunk.insert(0, "__role__", "syn")
                chunk.insert(0, "__split__", np.int32(j))
                chunk.insert(0, "__task_id__", np.int32(tsk))
                chunk.insert(0, "__dataset__", str(ds_name))

                ds_chunks.append(chunk)
                all_chunks.append(chunk)

            except Exception as e:
                ds_fail += 1
                failures.append({"task_id": tsk, "split": j, "role": "syn", "error": repr(e)})
                if pbar:
                    pbar.write(f"[ERROR] task_id={tsk} split={j}: {type(e).__name__}: {e}")
            finally:
                # advance the single progress bar by 1 split no matter what
                if pbar:
                    pbar.set_postfix(task_id=tsk, split=j, ds=ds_name[:24], refresh=False)
                    pbar.update(1)

    if pbar:
        pbar.close()

    syn_long = pd.concat(all_chunks, axis=0, ignore_index=True) if all_chunks else \
               pd.DataFrame(columns=["__dataset__", "__task_id__", "__split__", "__role__"])
    return syn_long, failures




def build_all_synthetics_fc_tasks_cat(
    tasks_to_keep,
    split_seeds,
    pfn_method,
    generator_kwargs=None,
    *,
    show_progress: bool = True,
):
    """
    Synthesize from X_orig only for each (task_id, split).
    Returns:
      syn_long  : big DataFrame with metadata cols [__dataset__, __task_id__, __split__, __role__='syn']
                  and per-dataset *prefixed* data columns
      failures  : list of {task_id, split, role, error}
    """
    if generator_kwargs is None:
        generator_kwargs = {}

    all_chunks: list[pd.DataFrame] = []
    failures: list[dict] = []

    total_steps = len(tasks_to_keep) * len(split_seeds)
    pbar = tqdm(total=total_steps, desc="Synthesizing (datasets × splits)", unit="split") if show_progress else None

    for i, tsk in enumerate(tasks_to_keep, start=1):
        # fetch dataset once
        task = openml.tasks.get_task(tsk)
        dataset = task.get_dataset()
        ds_name = dataset.name
        pref = _safe_prefix(ds_name, tsk)

        # load all columns (features + target)
        X, y, categorical_mask, attr_names = dataset.get_data(
            target=None, dataset_format="dataframe"
        )

        ds_chunks = []
        ds_fail = 0

        for j, seed in enumerate(split_seeds, start=1):
            try:
                X_orig, _ = train_test_split(X, test_size=0.5, random_state=seed)

                # remove rows with NA values
                X_orig = X_orig.dropna().reset_index(drop=True)

                # remove numerical columns
                X_orig = X_orig.select_dtypes(exclude="number")

                # select pfn method
                if pfn_method == "tabpfn":
                    X_syn = full_conditionals_tabpfn_generator(X_orig)

                if pfn_method == "tabicl":
                    X_syn = full_conditionals_tabicl_generator(X_orig)

                chunk = X_syn.copy()
                chunk.columns = [pref + c for c in chunk.columns]
                chunk.insert(0, "__role__", "syn")
                chunk.insert(0, "__split__", np.int32(j))
                chunk.insert(0, "__task_id__", np.int32(tsk))
                chunk.insert(0, "__dataset__", str(ds_name))

                ds_chunks.append(chunk)
                all_chunks.append(chunk)

            except Exception as e:
                ds_fail += 1
                failures.append({"task_id": tsk, "split": j, "role": "syn", "error": repr(e)})
                if pbar:
                    pbar.write(f"[ERROR] task_id={tsk} split={j}: {type(e).__name__}: {e}")
            finally:
                # advance the single progress bar by 1 split no matter what
                if pbar:
                    pbar.set_postfix(task_id=tsk, split=j, ds=ds_name[:24], refresh=False)
                    pbar.update(1)

    if pbar:
        pbar.close()

    syn_long = pd.concat(all_chunks, axis=0, ignore_index=True) if all_chunks else \
               pd.DataFrame(columns=["__dataset__", "__task_id__", "__split__", "__role__"])
    return syn_long, failures




def build_all_synthetics_miav_tasks_cat(
    tasks_to_keep,
    split_seeds,
    pfn_method,
    generator_kwargs=None,
    *,
    show_progress: bool = True,
):
    """
    Synthesize from X_orig only for each (task_id, split).
    Returns:
      syn_long  : big DataFrame with metadata cols [__dataset__, __task_id__, __split__, __role__='syn']
                  and per-dataset *prefixed* data columns
      failures  : list of {task_id, split, role, error}
    """
    if generator_kwargs is None:
        generator_kwargs = {}

    all_chunks: list[pd.DataFrame] = []
    failures: list[dict] = []

    total_steps = len(tasks_to_keep) * len(split_seeds)
    pbar = tqdm(total=total_steps, desc="Synthesizing (datasets × splits)", unit="split") if show_progress else None

    for i, tsk in enumerate(tasks_to_keep, start=1):
        # fetch dataset once
        task = openml.tasks.get_task(tsk)
        dataset = task.get_dataset()
        ds_name = dataset.name
        pref = _safe_prefix(ds_name, tsk)

        # load all columns (features + target)
        X, y, categorical_mask, attr_names = dataset.get_data(
            target=None, dataset_format="dataframe"
        )

        ds_chunks = []
        ds_fail = 0

        for j, seed in enumerate(split_seeds, start=1):
            try:
                X_orig, _ = train_test_split(X, test_size=0.5, random_state=seed)

                # remove rows with NA values
                X_orig = X_orig.dropna().reset_index(drop=True)

                # remove numerical columns
                X_orig = X_orig.select_dtypes(exclude="number")

                # select pfn method
                if pfn_method == "tabpfn":
                    X_syn = miav_tabpfn_generator(X_orig)

                if pfn_method == "tabicl":
                    X_syn = miav_tabicl_generator(X_orig)

                chunk = X_syn.copy()
                chunk.columns = [pref + c for c in chunk.columns]
                chunk.insert(0, "__role__", "syn")
                chunk.insert(0, "__split__", np.int32(j))
                chunk.insert(0, "__task_id__", np.int32(tsk))
                chunk.insert(0, "__dataset__", str(ds_name))

                ds_chunks.append(chunk)
                all_chunks.append(chunk)

            except Exception as e:
                ds_fail += 1
                failures.append({"task_id": tsk, "split": j, "role": "syn", "error": repr(e)})
                if pbar:
                    pbar.write(f"[ERROR] task_id={tsk} split={j}: {type(e).__name__}: {e}")
            finally:
                # advance the single progress bar by 1 split no matter what
                if pbar:
                    pbar.set_postfix(task_id=tsk, split=j, ds=ds_name[:24], refresh=False)
                    pbar.update(1)

    if pbar:
        pbar.close()

    syn_long = pd.concat(all_chunks, axis=0, ignore_index=True) if all_chunks else \
               pd.DataFrame(columns=["__dataset__", "__task_id__", "__split__", "__role__"])
    return syn_long, failures



