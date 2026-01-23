import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def remove_metadata(content):
    """Remove metadata lines (lines with ==, --, and LOG FILE lines)"""
    lines = content.split("\n")
    clean_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("=="):
            continue
        if stripped.startswith("--"):
            continue
        if "LOG FILE:" in stripped:
            continue
        clean_lines.append(line)

    return clean_lines


def add_cum_cost(df):
    """
    Add cumulative cost column 'cum_cost' along the training timeline.
    - Sorts by iterations.
    - Treats missing cost as 0 so gaps or boundary rows do not break the sum.
    """
    df = df.copy()
    df = df.sort_values("iterations")
    cost_filled = df["cost"].fillna(0)
    df["cum_cost"] = cost_filled.cumsum().round(2)
    return df


def validate_timestamp_order(df, stage_name="Data"):
    """
    Validate that timestamps are in strictly ascending order.
    Raises ValueError if not.
    """
    if df is None or df.empty:
        return

    # Check if we need to convert to datetime temporarily for validation
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        try:
            timestamps = pd.to_datetime(df["timestamp"])
        except Exception as e:
            print(
                f"Warning: Could not parse timestamps for validation in {stage_name}: {e}"
            )
            return
    else:
        timestamps = df["timestamp"]

    if not timestamps.is_monotonic_increasing:
        # Find first violation for debugging
        diff = timestamps.diff()
        # diff is Timedelta, negative means violation
        # We need to handle NaT for the first element
        mask = diff < pd.Timedelta(0)

        if mask.any():
            first_idx = mask.idxmax()

            # Get context
            prev_time = timestamps[first_idx - 1]
            curr_time = timestamps[first_idx]

            msg = (
                f"Timestamp validation failed at '{stage_name}'! "
                f"Found disordered timestamps.\n"
                f"Violation at index {first_idx}: {curr_time} (Current) < {prev_time} (Previous)"
            )
            raise ValueError(msg)

    print(f"Timestamp validation passed for '{stage_name}'.")


def parse_log_line(line):
    """Parse a single log line and extract all metrics"""
    pattern = (
        r"\[([^\]]+)\]\s+\[INFO\]:\s+Iteration\s+(\d+),\s+Cost\s+([\d.]+)s,\s+(.+)"
    )
    match = re.match(pattern, line.strip())

    if not match:
        return None

    timestamp = match.group(1)
    iteration = int(match.group(2))
    cost = float(match.group(3))
    metrics_str = match.group(4)

    metrics = {"timestamp": timestamp, "iterations": iteration, "cost": cost}

    for metric in metrics_str.split(","):
        metric = metric.strip()
        key_value = metric.split("=")
        if len(key_value) == 2:
            key = key_value[0].strip()
            value = float(key_value[1].strip())
            metrics[key] = value

    return metrics


def parse_logs_to_dataframe(clean_lines):
    """Parse all log lines into a dataframe with dynamic columns"""
    parsed_data = []

    for line in clean_lines:
        result = parse_log_line(line)
        if result:
            parsed_data.append(result)

    if not parsed_data:
        return None

    df = pd.DataFrame(parsed_data)

    return df


def expand_to_default_full_range(df, max_iter=100_000):
    """
    Ensure the dataframe contains boundary rows at iterations 0 and max_iter.
    Does NOT create a dense range.
    """
    df = df.copy()

    existing_iters = set(df["iterations"])

    rows_to_add = []

    if 0 not in existing_iters:
        rows_to_add.append({"iterations": 0})
        print("add Iterations: 0")

    if max_iter not in existing_iters:
        rows_to_add.append({"iterations": max_iter})
        print(f"add Iterations: {max_iter}")

    if rows_to_add:
        df = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)

    df = df.sort_values("iterations").reset_index(drop=True)

    return df


def normalize_column(series):
    """Normalize a series to 0-1 range"""
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_val) / (max_val - min_val)


def create_plots_over_time(df, output_dir):
    """
    Plot each numeric metric vs cum_cost (cumulative training time).
    Requires 'cum_cost' column (use add_cum_cost first).
    """
    os.makedirs(output_dir, exist_ok=True)

    if "cum_cost" not in df.columns:
        raise ValueError("cum_cost column not found. Call add_cum_cost(df) first.")

    plot_cols = [
        col
        for col in df.columns
        if col
        not in [
            # "iterations",
            "timestamp",
            "datetime",
            "time_from_prev",
            "cumulative_cost",
            "time_from_prev_estimated",
            "time_from_prev_filled",
            "cum_cost",
            "cost",
        ]
        and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    for col in plot_cols:
        # Filter out rows where either x or y is NaN
        mask = df["cum_cost"].notna() & df[col].notna()
        x_data = df.loc[mask, "cum_cost"]
        y_data = df.loc[mask, col]
        
        if len(x_data) == 0:
            print(f"Warning: No valid data for {col}, skipping plot.")
            continue
                
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_data, y_data, marker="o", linestyle="-", markersize=3)
        ax.set_xlabel("Cumulative Cost (s)")
        ax.set_ylabel(col)
        ax.set_title(f"{col} vs Cumulative Cost")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_vs_cum_cost.png"), dpi=150)
        plt.show()
        plt.close()

    print(f"Time-based plots saved to {output_dir}")


def create_plots(df, output_dir):
    """Create plots for all columns vs iterations"""
    os.makedirs(output_dir, exist_ok=True)

    plot_cols = [
        col
        for col in df.columns
        if col
        not in [
            "iterations",
            "timestamp",
            "datetime",
            "time_from_prev",
            "cumulative_cost",
            "time_from_prev_estimated",
            "time_from_prev_filled",
        ]
        and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    for col in plot_cols:
        # Filter out rows where either x or y is NaN
        mask = df["iterations"].notna() & df[col].notna()
        x_data = df.loc[mask, "iterations"]
        y_data = df.loc[mask, col]
        
        if len(x_data) == 0:
            print(f"Warning: No valid data for {col}, skipping plot.")
            continue

        if col == "accuracy" or col == "softmax_accuracy":
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_data, y_data, marker="o", linestyle="-", markersize=3)
            ax.set_xlabel("Iterations")
            ax.set_ylabel(col)
            ax.set_title(f"{col} vs Iterations")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{col}_raw.png"), dpi=150)
            plt.show()
            plt.close()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            axes[0].plot(
                x_data, y_data, marker="o", linestyle="-", markersize=3
            )
            axes[0].set_xlabel("Iterations")
            axes[0].set_ylabel(col)
            axes[0].set_title(f"{col} vs Iterations (Raw)")
            axes[0].grid(True, alpha=0.3)

            normalized = normalize_column(df[col])
            axes[1].plot(
                df["iterations"],
                normalized,
                marker="o",
                linestyle="-",
                markersize=3,
                color="orange",
            )
            axes[1].set_xlabel("Iterations")
            axes[1].set_ylabel(f"{col} (normalized)")
            axes[1].set_title(f"{col} vs Iterations (Normalized 0-1)")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{col}_comparison.png"), dpi=150)
            plt.show()
            plt.close()

    print(f"Plots saved to {output_dir}")


def process_train_log(input_path):
    """Main function to process a train_log.txt file"""
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Processing {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    print("Stage 1: Removing metadata...")
    clean_lines = remove_metadata(content)

    print("Stage 1: Parsing to dataframe...")
    df = parse_logs_to_dataframe(clean_lines)

    if df is None or df.empty:
        print("No valid data found in file")
        return

    print(f"Found {len(df)} records with columns: {list(df.columns)}")

    # Pre-save raw parsed data if needed, but we'll proceed to cleaning first
    # output_path = input_path.parent / f"{input_path.stem}_raw.csv"
    # df.to_csv(output_path, index=False)

    print("\nStage 1.5: Removing overlapped sessions (Issue 1)...")
    df_clean = remove_overlapped_data(df)
    print(f"Reduced from {len(df)} to {len(df_clean)} records")

    output_path = input_path.parent / f"{input_path.stem}_preprocessed_stage1.csv"
    df_clean.to_csv(output_path, index=False, float_format="%.15g")
    print(f"Stage 1 output saved to {output_path}")

    print("\nStage 2: Preprocessing timestamps...")
    df = preprocess_timestamps(df_clean)

    print("Stage 2: Removing similar records...")
    df = remove_similar_records(df)

    output_path2 = input_path.parent / f"{input_path.stem}_preprocessed_stage2.csv"
    df.to_csv(output_path2, index=False, float_format="%.15g")
    print(f"Stage 2 output saved to {output_path2}")

    print("\nStage 3: Creating plots...")
    imgs_dir = input_path.parent / "imgs"
    create_plots(df, str(imgs_dir))

    print("\nProcessing complete!")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


# -----------------------------


def analyze_processing_time(df):
    """
    Detailed analytics of the relationship between iterations and processing time.
    Generates plots for Processing Speed (Observed vs Estimated), Distribution, and Cumulative Time.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Work on a copy to avoid affecting original df
    df_copy = df.copy()

    # Ensure cumulative_cost exists
    if "cumulative_cost" not in df_copy.columns:
        if "cost" in df_copy.columns:
            df_copy["cumulative_cost"] = df_copy["cost"].cumsum()
        else:
            print("Warning: 'cost' column missing, cannot calculate cumulative cost.")
            df_copy["cumulative_cost"] = 0

    # Identify observed vs estimated
    # Observed: time_from_prev is not NaN and > 0
    # Estimated: time_from_prev is NaN (but time_from_prev_filled has value)

    df_copy["speed_observed"] = df_copy["time_from_prev"] / df_copy["iterations"].diff()
    df_copy["speed_filled"] = (
        df_copy["time_from_prev_filled"] / df_copy["iterations"].diff()
    )

    # Valid masks
    observed_mask = df_copy["speed_observed"].notna() & (df_copy["speed_observed"] > 0)
    estimated_mask = df_copy["time_from_prev"].isna() & df_copy["speed_filled"].notna()

    valid_speeds = df_copy.loc[observed_mask, "speed_observed"]

    if valid_speeds.empty:
        print("No valid speed data to analyze.")
        return

    median_speed = valid_speeds.median()
    mean_speed = valid_speeds.mean()

    print(f"\n--- Processing Speed Analysis ---")
    print(f"Median Speed: {median_speed:.6f} s/iter")
    print(f"Mean Speed:   {mean_speed:.6f} s/iter")
    print(f"Std Dev:      {valid_speeds.std():.6f} s/iter")
    print(f"Observed Points: {observed_mask.sum()}")
    print(f"Estimated Points: {estimated_mask.sum()}")

    # Create a figure with 3 subplots
    plt.figure(figsize=(20, 6))

    # Plot 1: Scatter of Speed over Time (Iterations)
    plt.subplot(1, 3, 1)

    # Plot observed
    plt.scatter(
        df_copy.loc[observed_mask, "iterations"],
        valid_speeds,
        alpha=0.6,
        s=15,
        label="Observed Speed",
        color="blue",
    )

    # Plot estimated (if any)
    if estimated_mask.any():
        plt.scatter(
            df_copy.loc[estimated_mask, "iterations"],
            df_copy.loc[estimated_mask, "speed_filled"],
            alpha=0.6,
            s=15,
            marker="x",
            label="Estimated (Median)",
            color="red",
        )

    # Lines for Mean and Median
    plt.axhline(
        y=median_speed,
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Median ({median_speed:.2f}s)",
    )
    plt.axhline(
        y=mean_speed,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Mean ({mean_speed:.2f}s)",
    )

    plt.title("Processing Speed vs Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Seconds per Iteration")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Auto-scale Y
    q99 = valid_speeds.quantile(0.99)
    if not np.isnan(q99) and q99 > 0:
        plt.ylim(0, q99 * 1.5)

    # Plot 2: Distribution (Histogram + KDE)
    plt.subplot(1, 3, 2)
    sns.histplot(valid_speeds, kde=True, bins=50, color="blue", label="Observed Data")
    plt.axvline(
        x=median_speed, color="green", linestyle="-", linewidth=2, label=f"Median"
    )
    plt.axvline(
        x=mean_speed, color="orange", linestyle="--", linewidth=2, label=f"Mean"
    )

    plt.title("Distribution of Observed Speeds")
    plt.xlabel("Seconds per Iteration")
    if not np.isnan(q99) and q99 > 0:
        plt.xlim(0, q99 * 1.5)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Cumulative Time (Estimated Total Training Time)
    plt.subplot(1, 3, 3)
    plt.plot(
        df_copy["iterations"],
        df_copy["cumulative_cost"] / 3600.0,
        color="purple",
        linewidth=2,
    )

    plt.title("Cumulative Training Time")
    plt.xlabel("Iteration")
    plt.ylabel("Total Time (Hours)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def remove_overlapped_data(df):
    """
    Remove data from overlapped sessions.
    Strategy:
    1. Sort by timestamp to establish temporal order.
    2. For duplicated iterations, keep the LAST one (assuming latest run is the valid one).
    3. Sort by iterations (to reconstruct the training timeline).
    4. Filter out any records with timestamps earlier than the cumulative maximum timestamp
       (this removes 'future' records from abandoned sessions that were superseded by later runs).
    """
    df_clean = df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_clean["timestamp"]):
        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])

    # Sort by timestamp globally first
    df_clean = df_clean.sort_values("timestamp")

    # Keep the latest record for each iteration
    df_clean = df_clean.drop_duplicates(subset=["iterations"], keep="last")

    # Sort by iteration to check for timeline consistency
    df_clean = df_clean.sort_values("iterations")

    # Remove records that effectively go "back in time" relative to the progressed timeline
    # This handles cases where a previous run went further (e.g. to iter 30000)
    # but the restart happened at 20000 and only reached 22000 so far.
    # The old 22001-30000 are invalid "orphaned futures".

    # We enforce that as iterations increase, time must generally not decrease below the valid watermark.
    # cummax() gives the "latest valid time" seen so far.

    valid_mask = df_clean["timestamp"] >= df_clean["timestamp"].cummax()
    df_clean = df_clean[valid_mask]

    return df_clean.reset_index(drop=True)


def expand_to_full_range(df, max_iter=100000):
    """
    Expand the dataframe to cover iterations from 0 to max_iter based on the detected step size.
    """
    # Detect step size (mode of differences)
    # We filter for positive differences to avoid noise from overlaps if any remain
    diffs = df["iterations"].sort_values().diff()
    valid_diffs = diffs[diffs > 0]

    if valid_diffs.empty:
        step_size = 100  # Default fallback
    else:
        step_size = int(valid_diffs.mode().iloc[0])

    print(f"Detected step size: {step_size}")

    # Create full range
    full_iterations = range(0, max_iter + 1, step_size)
    df_full = pd.DataFrame({"iterations": full_iterations})

    # Merge with existing data
    # We use 'left' join on the full range to ensure we have all steps
    df_merged = pd.merge(df_full, df, on="iterations", how="left")

    return df_merged


def preprocess_timestamps(df):
    """
    Stage 2: Preprocess timestamps.
    1. Expand to full iteration range (0 to 100k).
    2. Calculate time differences (time_from_prev) for valid consecutive segments.
    """
    # Expand first
    df = expand_to_full_range(df)

    df["datetime"] = pd.to_datetime(df["timestamp"])

    # Calculate time from previous (row-wise diff)
    # Note: This will only be valid if BOTH current and prev rows have valid timestamps.
    # Gaps in the data will result in NaNs here, which will be filled by estimation later.
    df["time_from_prev"] = df["datetime"].diff().dt.total_seconds()

    # Cumulative cost is meaningless with gaps, so we might want to recalculate it later
    # or just leave it as is for valid rows.
    # The user asked to "leave other columns empty", so we respect that.
    # Estimate using statistics instead of regression
    if "time_from_prev" not in df.columns:
        df["time_from_prev"] = np.nan

    df = estimate_time_statistically(df)

    # Calculate cumulative cost (training time)
    # cumsum() will skip NaNs by default in pandas but we want to know total accumulated cost from valid records
    df["cumulative_cost"] = df["cost"].cumsum()

    return df


def estimate_time_statistically(df):
    """
    Estimate missing time values using statistical distribution of time per iteration.
    Instead of regression, we calculate the median time per iteration from valid segments
    and use it to predict missing gaps.
    """
    df_copy = df.copy()

    # We need to look at valid segments to determine the distribution
    # A valid segment is where we have both current and previous timestamps relative to the step size
    # But here 'time_from_prev' is already calculated as simple diff of datetimes.
    # Rows with NaT timestamps resulted in NaN time_from_prev.
    # Rows following a gap also resulted in NaN time_from_prev (because prev was NaT).

    # Identify strictly valid rows: those that have a valid time_from_prev
    # These represent segments where we had contiguous logs.
    valid_mask = df_copy["time_from_prev"].notna() & (df_copy["time_from_prev"] > 0)

    # Calculate step sizes for these valid rows to normalize speed
    # We can't rely on fixed step size everywhere, so we calculate dynamic speed
    # Note: 'iterations' is monotonic, so diff is valid
    iter_diffs = df_copy["iterations"].diff()

    # We only care about the speed from the valid time segments
    valid_speeds = (
        df_copy.loc[valid_mask, "time_from_prev"] / iter_diffs.loc[valid_mask]
    )

    if valid_speeds.empty:
        print("Warning: No valid segments found to estimate speed. Filling with 0.")
        df_copy["time_from_prev_filled"] = df_copy["time_from_prev"].fillna(0)
        return df_copy

    # Statistical Analysis
    median_speed = valid_speeds.median()
    mean_speed = valid_speeds.mean()
    std_speed = valid_speeds.std()

    print(f"Time per Iteration Statistics (seconds/iter):")
    print(f"  Median: {median_speed:.6f}")
    print(f"  Mean:   {mean_speed:.6f}")
    print(f"  StdDev: {std_speed:.6f}")

    # Fill missing values
    # For a missing row, we estimate its time_from_prev as: step_size * median_speed
    # We need the step size for each row (diff from previous iteration)
    # Since iterations are full range now, step size should be constant (e.g. 100),
    # but using diff() is safer.

    # We fill step sizes for the first row if NaN (though it's 0 usually)
    current_iter_diffs = df_copy["iterations"].diff().fillna(0)

    estimated_times = current_iter_diffs * median_speed

    # Create the filled column
    # If time_from_prev is valid, use it. If NaN (missing or gap), use estimated.
    df_copy["time_from_prev_estimated"] = estimated_times
    df_copy["time_from_prev_filled"] = df_copy["time_from_prev"].fillna(estimated_times)

    return df_copy


def remove_similar_records(df, threshold=0.001):
    """Remove duplicate or similar records based on small differences in values"""
    df_copy = df.copy()

    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    numeric_cols = [
        col for col in numeric_cols if col not in ["iterations", "datetime"]
    ]

    if len(numeric_cols) == 0:
        return df_copy

    df_copy["diff_sum"] = 0
    for col in numeric_cols:
        if col in df_copy.columns:
            df_copy[f"{col}_diff"] = df_copy[col].diff().abs()
            df_copy["diff_sum"] += df_copy[f"{col}_diff"].fillna(1.0)

    keep_mask = (df_copy["diff_sum"] > threshold) | (df_copy.index == 0)

    for col in numeric_cols:
        if f"{col}_diff" in df_copy.columns:
            df_copy.drop(f"{col}_diff", axis=1, inplace=True)
    df_copy.drop("diff_sum", axis=1, inplace=True)

    return df_copy[keep_mask]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess_train_logs.py <path_to_train_log.txt>")
        sys.exit(1)

    input_file = sys.argv[1]
    process_train_log(input_file)
