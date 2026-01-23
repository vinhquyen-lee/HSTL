import os
import re
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math


def _is_missing_scalar(x):
    # True only for real scalar NaN / None. Lists/arrays are not "missing".
    return x is None or (isinstance(x, float) and math.isnan(x))


def _as_list(x):
    if _is_missing_scalar(x):
        return None
    # if already list-like, normalize to python list
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x  # list or other sequence


def _display_model_name(model_name, use_vn: bool = False):
    """
    Map internal folder/model names to Vietnamese display names for plots.

    Only affects labels when use_vn is True.
    """
    if not use_vn:
        return model_name

    mapping = {
        "kaggle": "Gốc",
        "p3d": "P3D",
        "med-circleloss": "Circle Loss",
        "p3d-circleloss": "P3D + Circle Loss",
    }
    return mapping.get(model_name, model_name)


def parse_log_file(log_path):
    """Parse a single log file and extract Rank-1 results."""
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    results = {}

    # Extract iteration from filename
    filename = os.path.basename(log_path)
    match = re.search(r"_(\d+)\.log$", filename)
    if match:
        results["iterations"] = int(match.group(1))
    else:
        return None

    # Parse Include identical-view cases
    include_nm = None
    include_bg = None
    include_cl = None

    for i, line in enumerate(lines):
        if "===Rank-1 (Include identical-view cases)===" in line:
            if i + 1 < len(lines):
                match = re.search(
                    r"NM:\s+([\d.]+),\s+BG:\s+([\d.]+),\s+CL:\s+([\d.]+)", lines[i + 1]
                )
                if match:
                    include_nm = float(match.group(1))
                    include_bg = float(match.group(2))
                    include_cl = float(match.group(3))
                    break

    # Parse Exclude identical-view cases
    exclude_nm = None
    exclude_bg = None
    exclude_cl = None

    for i, line in enumerate(lines):
        if "===Rank-1 (Exclude identical-view cases)===" in line:
            if i + 1 < len(lines):
                match = re.search(
                    r"NM:\s+([\d.]+),\s+BG:\s+([\d.]+),\s+CL:\s+([\d.]+)", lines[i + 1]
                )
                if match:
                    exclude_nm = float(match.group(1))
                    exclude_bg = float(match.group(2))
                    exclude_cl = float(match.group(3))
                    break

    # Parse angle-wise lists
    nm_list = None
    bg_list = None
    cl_list = None

    for i, line in enumerate(lines):
        if "===Rank-1 of each angle (Exclude identical-view cases)===" in line:
            for j in range(i + 1, min(i + 5, len(lines))):
                if "NM:" in lines[j] and "[" in lines[j]:
                    nm_str = re.search(r"NM:\s+\[(.*?)\]", lines[j])
                    if nm_str:
                        nm_list = [float(x) for x in nm_str.group(1).split()]
                if "BG:" in lines[j] and "[" in lines[j]:
                    bg_str = re.search(r"BG:\s+\[(.*?)\]", lines[j])
                    if bg_str:
                        bg_list = [float(x) for x in bg_str.group(1).split()]
                if "CL:" in lines[j] and "[" in lines[j]:
                    cl_str = re.search(r"CL:\s+\[(.*?)\]", lines[j])
                    if cl_str:
                        cl_list = [float(x) for x in cl_str.group(1).split()]
            break

    # # Calculate means
    # if include_nm is not None and include_bg is not None and include_cl is not None:
    #     included_mean = (include_nm + include_bg + include_cl) / 3.0
    # else:
    #     included_mean = None

    # if exclude_nm is not None and exclude_bg is not None and exclude_cl is not None:
    #     excluded_mean = (exclude_nm + exclude_bg + exclude_cl) / 3.0
    # else:
    #     excluded_mean = None

    results["NM"] = nm_list
    results["BG"] = bg_list
    results["CL"] = cl_list
    results["NM_Incl"] = include_nm
    results["BG_Incl"] = include_bg
    results["CL_Incl"] = include_cl
    results["NM_Excl"] = exclude_nm
    results["BG_Excl"] = exclude_bg
    results["CL_Excl"] = exclude_cl

    # results['Included_mean'] = included_mean
    # results['Excluded_mean'] = excluded_mean

    return results


def add_std_columns(df):
    """Calculate standard deviation for NM, BG, CL lists and add as new columns."""
    df = df.copy()

    # Calculate std for each list column
    df["NM_std"] = df["NM"].apply(
        lambda x: np.round(np.std(x), 1) if x is not None and len(x) > 0 else np.nan
    )
    df["BG_std"] = df["BG"].apply(
        lambda x: np.round(np.std(x), 1) if x is not None and len(x) > 0 else np.nan
    )
    df["CL_std"] = df["CL"].apply(
        lambda x: np.round(np.std(x), 1) if x is not None and len(x) > 0 else np.nan
    )

    return df


def plot_column_over_iterations(
    dataframes_dict,
    column_name,
    title=None,
    figsize=(12, 6),
    save_path=None,
    use_vn_labels: bool = False,
):
    """
    Plot a single column value over iterations for multiple dataframes.

    Parameters:
    -----------
    dataframes_dict : dict
        Dictionary where keys are model names and values are dataframes
    column_name : str
        Name of the column to plot (e.g., 'NM_std', 'BG_std', 'CL_std', 'Included_mean', 'Excluded_mean')
    title : str, optional
        Plot title. If None, uses column_name
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save the plot. If None, plot is displayed but not saved
    """
    plt.figure(figsize=figsize)

    min_iterations = []

    for model_name, df in dataframes_dict.items():
        if column_name not in df.columns:
            print(
                f"Warning: Column '{column_name}' not found in {model_name}, skipping..."
            )
            continue

        # Filter out None values
        plot_df = df[["iterations", column_name]].copy()
        plot_df = plot_df.dropna(subset=[column_name])

        if len(plot_df) == 0:
            print(f"Warning: No valid data for {model_name}, skipping...")
            continue

        # Get minimum iteration for this dataframe
        min_iter = plot_df["iterations"].min()
        min_iterations.append(min_iter)

        # # Plot line
        # plt.plot(
        #     plot_df["iterations"],
        #     plot_df[column_name],
        #     marker="o",
        #     markersize=4,
        #     linewidth=1.5,
        #     label=model_name,
        # )

    # Set x-axis range: start from the maximum of minimums (excluding 0)
    if min_iterations:
        print(min_iterations)
        # Filter out 0 from minimum iterations
        min_iterations_filtered = [m for m in min_iterations if m > 0]
        print(min_iterations_filtered)
        if min_iterations_filtered:
            x_min = max(min_iterations_filtered)
        else:
            x_min = 0
        print(x_min)
        # x_max = 100000
        # plt.xlim(x_min, x_max)
    else:
        # plt.xlim(0, 100000)
        x_min = 0

    # Second pass: plot data filtered by x_min
    for model_name, df in dataframes_dict.items():
        if column_name not in df.columns:
            print(
                f"Warning: Column '{column_name}' not found in {model_name}, skipping..."
            )
            continue

        # Filter out None values and iterations below x_min
        plot_df = df[["iterations", column_name]].copy()
        plot_df = plot_df.dropna(subset=[column_name])
        plot_df = plot_df[plot_df["iterations"] >= x_min]

        if len(plot_df) == 0:
            print(
                f"Warning: No valid data for {model_name} after filtering, skipping..."
            )
            continue

        # Plot line
        display_name = _display_model_name(model_name, use_vn_labels)
        plt.plot(
            plot_df["iterations"],
            plot_df[column_name],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=display_name,
        )

    # Set x-axis range
    x_max = 100000
    plt.xlim(x_min, x_max)

    # plt.xlabel("Bước lặp", fontsize=12)
    # plt.ylabel(column_name, fontsize=12)
    # plt.title(title if title else f"{column_name} over Iterations", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=10)
    plt.xlim(0, 100000)

    # Format x-axis to show in thousands
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"{int(x / 1000)}K" if x > 0 else "0")
    )

    plt.tight_layout()

    if save_path:
        save_path = os.path.join(save_path, f"{column_name}.jpg")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()


def extract_final_iteration_data(dataframes_dict, target_iteration=100000):
    """
    Extract NM, BG, CL values at target iteration from all dataframes.

    Parameters:
    -----------
    dataframes_dict : dict
        Dictionary where keys are model names and values are dataframes
    target_iteration : int
        Target iteration to extract (default: 100000)

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: model_name, view, NM, BG, CL
    """
    all_data = []

    # Generate view angles (11 views from 0 to 180, step ~16.36)
    # CASIA-B standard: 0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180
    views = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]

    for model_name, df in dataframes_dict.items():
        # Find row closest to target_iteration
        df_sorted = df.sort_values("iterations")

        # Get the row with iteration <= target_iteration, closest to target
        candidates = df_sorted[df_sorted["iterations"] <= target_iteration]

        if len(candidates) == 0:
            # If no data <= target, get the minimum iteration
            candidates = df_sorted.head(1)
        else:
            # Get the row with maximum iteration (closest to target)
            candidates = candidates.tail(1)

        if len(candidates) == 0:
            print(f"Warning: No data found for {model_name}")
            continue

        row = candidates.iloc[0]
        actual_iteration = row["iterations"]

        # Extract lists
        nm_list = _as_list(row["NM"])
        bg_list = _as_list(row["BG"])
        cl_list = _as_list(row["CL"])

        if nm_list is None or len(nm_list) != 11:
            print(
                f"Warning: Invalid NM data for {model_name} at iteration {actual_iteration}"
            )
            continue

        # Create rows for each view
        for i, view in enumerate(views):
            if i < len(nm_list):
                all_data.append(
                    {
                        "model_name": model_name,
                        "iteration": actual_iteration,
                        "view": view,
                        "NM": nm_list[i],
                        "BG": bg_list[i] if bg_list and i < len(bg_list) else None,
                        "CL": cl_list[i] if cl_list and i < len(cl_list) else None,
                    }
                )

    result_df = pd.DataFrame(all_data)
    return result_df


def extract_final_iteration_stats_for_model(
    dataframes_dict, model_name, target_iteration=100000
):
    """
    Extract per-view NM/BG/CL at target iteration for a single model and
    return compact stats table:

    Rows: one for each metric type (NM, BG, CL)
    Columns: ['metric', 11 view angles, 'mean', 'std']
    """
    if model_name not in dataframes_dict:
        print(f"Warning: model '{model_name}' not found in dataframes_dict")
        return pd.DataFrame()

    df = dataframes_dict[model_name]
    if df is None or df.empty:
        print(f"Warning: empty dataframe for model '{model_name}'")
        return pd.DataFrame()

    df_sorted = df.sort_values("iterations")

    # Get the row with iteration <= target_iteration, closest to target
    candidates = df_sorted[df_sorted["iterations"] <= target_iteration]
    if len(candidates) == 0:
        # If no data <= target, get the minimum iteration
        candidates = df_sorted.head(1)
    else:
        candidates = candidates.tail(1)

    if len(candidates) == 0:
        print(f"Warning: No data found for model '{model_name}'")
        return pd.DataFrame()

    row = candidates.iloc[0]

    nm_list = _as_list(row["NM"])
    bg_list = _as_list(row.get("BG"))
    cl_list = _as_list(row.get("CL"))

    if nm_list is None:
        print(
            f"Warning: Invalid NM data for {model_name} at iteration {row['iterations']}"
        )
        return pd.DataFrame()

    # CASIA-B standard view angles (11 views)
    views = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
    view_cols = [f"view_{v}" for v in views]

    def _row_from_list(metric_name, values_list, incl_val, excl_val, std_val):
        # Ensure we have a list of length 11 (pad/truncate if necessary)
        vals = list(values_list) if values_list is not None else [None] * 11
        if len(vals) < 11:
            vals = vals + [None] * (11 - len(vals))
        elif len(vals) > 11:
            vals = vals[:11]

        data = {"metric": metric_name}
        for v, col_name in zip(vals, view_cols):
            data[col_name] = v
        # Use existing stats from dataframe; do not recompute
        data["Incl"] = incl_val
        data["Excl"] = excl_val
        data["Std"] = std_val
        return data

    rows = []
    rows.append(
        _row_from_list(
            "NM",
            nm_list,
            row.get("NM_Incl"),
            row.get("NM_Excl"),
            row.get("NM_std"),
        )
    )
    rows.append(
        _row_from_list(
            "BG",
            bg_list,
            row.get("BG_Incl"),
            row.get("BG_Excl"),
            row.get("BG_std"),
        )
    )
    rows.append(
        _row_from_list(
            "CL",
            cl_list,
            row.get("CL_Incl"),
            row.get("CL_Excl"),
            row.get("CL_std"),
        )
    )

    return pd.DataFrame(rows, columns=["metric"] + view_cols + ["Incl", "Excl", "Std"])


def plot_final_iteration_scatter(
    final_df, save_path=None, figsize=(14, 8), use_vn_labels: bool = False
):
    """
    Plot scatter plot of NM, BG, CL values over views for all models at final iteration.

    Parameters:
    -----------
    final_df : pd.DataFrame
        DataFrame from extract_final_iteration_data
    save_path : str or Path, optional
        Path to save the plot
    figsize : tuple, optional
        Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)

    metrics = ["NM", "BG", "CL"]
    colors = plt.cm.tab10(range(len(final_df["model_name"].unique())))
    model_names = sorted(final_df["model_name"].unique())
    color_map = {model: colors[i] for i, model in enumerate(model_names)}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for model_name in model_names:
            display_name = _display_model_name(model_name, use_vn_labels)
            model_data = final_df[final_df["model_name"] == model_name]
            metric_data = model_data[metric].dropna()
            views_data = model_data.loc[metric_data.index, "view"]

            if len(metric_data) > 0:
                ax.scatter(
                    views_data,
                    metric_data,
                    label=display_name,
                    color=color_map[model_name],
                    alpha=0.7,
                    s=60,
                    edgecolors="black",
                    linewidths=0.5,
                )

        ax.set_xlabel("Góc", fontsize=11)  # View Angle (degrees)
        ax.set_ylabel(f"Độ chính xác (%)", fontsize=11)  # {metric}
        # ax.set_title(f"{metric} at Final Iteration", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 36, 72, 108, 144, 180])
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()

    if save_path:
        save_path = os.path.join(save_path, f"final_iteration_scatter_{metric}.jpg")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_final_iteration_scatter_per_metric(
    final_df, save_dir=None, figsize=(8, 6), use_vn_labels: bool = False
):
    """
    Make separate scatter plots of NM, BG, CL over views for all models at final iteration.

    final_df columns: ['model_name', 'iteration', 'view', 'NM', 'BG, 'CL']
    """
    metrics = ["NM", "BG", "CL"]
    model_names = sorted(final_df["model_name"].unique())
    colors = plt.cm.tab10(range(len(model_names)))
    color_map = {model: colors[i] for i, model in enumerate(model_names)}

    for metric in metrics:
        plt.figure(figsize=figsize)

        for model_name in model_names:
            display_name = _display_model_name(model_name, use_vn_labels)
            model_data = final_df[final_df["model_name"] == model_name]
            metric_data = model_data[metric].dropna()
            views_data = model_data.loc[metric_data.index, "view"]

            if len(metric_data) == 0:
                continue

            plt.scatter(
                views_data,
                metric_data,
                label=display_name,
                color=color_map[model_name],
                alpha=0.7,
                s=60,
                edgecolors="black",
                linewidths=0.5,
            )

        # plt.xlabel('View Angle (degrees)', fontsize=11)
        plt.ylabel(f"Độ chính xác (%)", fontsize=11)
        # plt.title(f'{metric} at Final Iteration', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks([0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180])
        plt.legend(loc="lower left", fontsize=9)
        plt.tight_layout()

        if save_dir is not None:
            out_path = os.path.join(save_dir, f"final_iteration_scatter_{metric}.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {out_path}")

        plt.show()


def extract_final_wide_df(dataframes_dict, metric, target_iteration=100000):
    """
    Build a wide dataframe at (closest <= target_iteration) for one metric in {'NM','BG','CL'}.

    Output columns:
      - model_name
      - view_000, view_018, ..., view_180  (11 columns)
      - excluded_mean  (from *_Excl columns; no recompute)
      - excluded_std   (from *_std columns; no recompute)

    Assumptions:
      - Each df has: iterations, NM/BG/CL (list of 11), NM_Excl/BG_Excl/CL_Excl (scalar)
      - And after your std step: NM_std/BG_std/CL_std (scalar std of the 11-view list)
    """
    metric = metric.upper()
    if metric not in {"NM", "BG", "CL"}:
        raise ValueError("metric must be one of: 'NM', 'BG', 'CL'")

    views = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
    view_cols = [f"{v}" for v in views]

    excl_col = f"{metric}_Excl"
    std_col = f"{metric}_std"

    rows = []
    for model_name, df in dataframes_dict.items():
        if "iterations" not in df.columns or metric not in df.columns:
            continue

        # pick closest iteration <= target_iteration; if none, take earliest
        df_sorted = df.sort_values("iterations")
        candidates = df_sorted[df_sorted["iterations"] <= target_iteration]
        row = (candidates.tail(1) if len(candidates) else df_sorted.head(1)).iloc[0]

        vals = row[metric]
        if isinstance(vals, np.ndarray):
            vals = vals.tolist()
        if not isinstance(vals, (list, tuple)) or len(vals) != 11:
            continue

        out = {"model_name": model_name}
        for c, v in zip(view_cols, vals):
            out[c] = float(v)

        out["excluded_mean"] = row.get(excl_col, None)
        out["excluded_std"] = row.get(std_col, None)
        rows.append(out)

    return pd.DataFrame(rows)


def process_log_folder(logs_dir):
    """Process all log files in a folder and return a dataframe."""
    logs_path = Path(logs_dir)

    if not logs_path.exists():
        print(f"Directory not found: {logs_dir}")
        return None

    all_results = []

    # Get all .log files
    log_files = sorted(logs_path.glob("*.log"))

    for log_file in log_files:
        result = parse_log_file(log_file)
        if result:
            all_results.append(result)

    if not all_results:
        return None

    # Create dataframe
    df = pd.DataFrame(all_results)

    # Sort by iterations
    df = df.sort_values("iterations").reset_index(drop=True)

    return df


def process_all_folders(base_logs_dir):
    """Process all subfolders and return a dictionary of dataframes."""
    base_path = Path(base_logs_dir)

    if not base_path.exists():
        print(f"Base directory not found: {base_logs_dir}")
        return {}

    dataframes = {}

    # Process each subfolder
    for folder in sorted(base_path.iterdir()):
        if folder.is_dir():
            print(f"Processing folder: {folder.name}")
            df = process_log_folder(folder)
            if df is not None and not df.empty:
                dataframes[folder.name] = df
                print(f"  Found {len(df)} log files")
            else:
                print(f"  No valid log files found")

    return dataframes


if __name__ == "__main__":
    # Set the base logs directory
    base_dir = "logs/test/batch_test_results/logs"

    # Process all folders
    dataframes = process_all_folders(base_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for folder_name, df in dataframes.items():
        print(f"\n{folder_name}:")
        print(f"  Total iterations: {len(df)}")
        print(f"  Iteration range: {df['iterations'].min()} - {df['iterations'].max()}")
        print(f"\n  First few rows:")
        print(df[["iterations", "Included_mean", "Excluded_mean"]].head())

    # Save dataframes to CSV files
    output_dir = Path("logs/test/batch_test_results/parsed_dataframes")
    output_dir.mkdir(parents=True, exist_ok=True)

    for folder_name, df in dataframes.items():
        csv_path = output_dir / f"{folder_name}_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    # Save all dataframes to a single Excel file with multiple sheets
    excel_path = output_dir / "all_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for folder_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=folder_name, index=False)
    print(f"\nSaved all results to: {excel_path}")
