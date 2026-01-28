# graph.py
import sys
import argparse
import json
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# If your Windows environment has GUI issues, try forcing a GUI backend:
# matplotlib.use("TkAgg")  # uncomment if needed

def plot_stacked_bar_by_charge_range(
    data,
    bins=None,
    labels=None,
    colors=("tomato", "skyblue"),
    title="Number of People per Charge Range (Smokers vs Non-Smokers)",
    xlabel="Charge Range",
    ylabel="Number of People"
):
    # Convert to DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data, columns=["smoker", "charges"])
    else:
        df = data.copy()

    # Default bins
    if bins is None:
        bins = list(range(0, 100001, 10000))  # 0–100k in 10k steps

    # Default labels
    if labels is None:
        labels = [f"${bins[i]//1000}k–{bins[i+1]//1000}k" for i in range(len(bins)-1)]

    # Validate labels length
    if len(labels) != len(bins) - 1:
        raise ValueError("labels length must be len(bins) - 1")

    # Bin the charges
    df['charge_bin'] = pd.cut(df['charges'], bins=bins, labels=labels, include_lowest=True)

    # Group counts
    grouped = df.groupby(['charge_bin', 'smoker']).size().unstack(fill_value=0)

    # Ensure both smoker categories exist
    for cat in ["yes", "no"]:
        if cat not in grouped.columns:
            grouped[cat] = 0

    smoker_counts = grouped['yes']
    non_smoker_counts = grouped['no']
    total_counts = smoker_counts + non_smoker_counts

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(grouped.index, smoker_counts, label='Smoker', color=colors[0])
    ax.bar(grouped.index, non_smoker_counts, bottom=smoker_counts, label='Non-Smoker', color=colors[1])

    # Totals above bars
    for idx, total in enumerate(total_counts):
        ax.text(idx, total + 0.5, str(int(total)), ha='center', va='bottom', fontsize=9)

    # Styling
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig, ax


def load_input(input_path: str | None):
    """
    Accepts CSV or JSON input; or generates a small demo dataset if None.
    - CSV columns: smoker,charges
    - JSON: list of objects with keys smoker, charges
    """
    if input_path is None:
        # Demo data so you always see something
        print("[info] No input provided; using demo data.")
        demo = [
            # smoker, charges
            ("yes", 12000), ("no", 8000), ("no", 9500), ("yes", 24000),
            ("no", 30500), ("yes", 33000), ("no", 47000), ("no", 52000),
            ("yes", 68000), ("no", 72000), ("yes", 15000), ("no", 22000),
            ("yes", 9800), ("no", 8800), ("yes", 41000), ("no", 39000),
        ]
        return pd.DataFrame(demo, columns=["smoker", "charges"])

    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".csv"]:
        print(f"[info] Loading CSV: {input_path}")
        df = pd.read_csv(input_path)
        # Normalize expected columns
        rename_map = {}
        for c in df.columns:
            c_norm = c.strip().lower()
            if c_norm == "smoker":
                rename_map[c] = "smoker"
            elif c_norm == "charges":
                rename_map[c] = "charges"
        df = df.rename(columns=rename_map)
        if not {"smoker", "charges"} <= set(df.columns):
            raise ValueError("CSV must contain columns 'smoker' and 'charges'")
        return df

    if ext in [".json"]:
        print(f"[info] Loading JSON: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Accept list[dict] or list[tuple]
        if isinstance(data, list):
            if len(data) == 0:
                return pd.DataFrame(columns=["smoker", "charges"])
            if isinstance(data[0], dict):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame(data, columns=["smoker", "charges"])
        else:
            raise ValueError("JSON must be a list of objects or tuples")
        if not {"smoker", "charges"} <= set(df.columns):
            raise ValueError("JSON must contain keys 'smoker' and 'charges'")
        return df

    raise ValueError(f"Unsupported input type for: {input_path}")


def parse_bins(bin_str: str | None, step: int | None):
    """
    Either parse explicit bins like: "0,10000,20000,50000"
    or build range with step up to 100k: --step 10000
    """
    if bin_str:
        parts = [int(x.strip()) for x in bin_str.split(",") if x.strip()]
        if len(parts) < 2:
            raise ValueError("Provide at least two bin edges when using --bins")
        return parts
    if step:
        if step <= 0:
            raise ValueError("--step must be positive")
        # default 0..100k
        return list(range(0, 100001, step))
    return None


def main(argv):
    parser = argparse.ArgumentParser(description="Stacked bar of smokers vs non-smokers by charge range")
    sub = parser.add_subparsers(dest="cmd", required=False)

    run_p = sub.add_parser("run", help="Generate the plot")
    run_p.add_argument("--input", type=str, help="CSV/JSON with columns smoker,charges")
    run_p.add_argument("--bins", type=str, help='Comma-separated bin edges, e.g. "0,10000,20000,50000"')
    run_p.add_argument("--step", type=int, help="Build default 0..100k bins with this step (e.g. 10000)")
    run_p.add_argument("--labels", type=str, help='Comma-separated labels matching bins-1 (optional)')
    run_p.add_argument("--title", type=str, default="Number of People per Charge Range (Smokers vs Non-Smokers)")
    run_p.add_argument("--xlabel", type=str, default="Charge Range")
    run_p.add_argument("--ylabel", type=str, default="Number of People")
    run_p.add_argument("--show", action="store_true", help="Show interactive window")
    run_p.add_argument("--save", type=str, help="Save figure to this path (e.g., chart.png)")
    run_p.add_argument("--backend", type=str, help="Matplotlib backend override, e.g. TkAgg, Agg")

    args = parser.parse_args(argv)

    if args.cmd != "run":
        parser.print_help()
        print("\nHint: try 'python graph.py run --show' or 'python graph.py run --save chart.png'")
        return 0

    if args.backend:
        matplotlib.use(args.backend, force=True)
        print(f"[info] Using matplotlib backend: {matplotlib.get_backend()}")

    # Load data
    df = load_input(args.input)

    # Bins & labels
    bins = parse_bins(args.bins, args.step)
    labels = None
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",") if s.strip()]

    # Plot
    fig, ax = plot_stacked_bar_by_charge_range(
        data=df,
        bins=bins,
        labels=labels,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel
    )

    # Save/Show outcomes
    did_anything = False
    if args.save:
        out = os.path.abspath(args.save)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[ok] Saved figure to: {out}")
        did_anything = True

    if args.show:
        print("[info] Opening interactive window...")
        plt.show()
        did_anything = True

    if not did_anything:
        # Default: save a PNG next to the script so there is *some* output
        out = os.path.abspath("chart.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[ok] No --show/--save given. Wrote default: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
