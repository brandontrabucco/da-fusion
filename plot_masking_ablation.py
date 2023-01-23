import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from itertools import product

import os
import glob
import argparse
import math


def pretty(text):
    """Convert a string into a consistent format for
    presentation in a matplotlib pyplot:
    this version looks like: One Two Three Four
    """

    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    text = text.strip()
    prev_c = None
    out_str = []
    for c in text:
        if prev_c is not None and \
                prev_c.islower() and c.isupper():
            out_str.append(" ")
            prev_c = " "
        if prev_c is None or prev_c == " ":
            c = c.upper()
        out_str.append(c)
        prev_c = c
    return "".join(out_str)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Few-Shot Baseline")

    parser.add_argument("--logdirs", nargs="+", type=str, 
                        default=["./pascal-baselines", "./coco-baselines"])
    
    parser.add_argument("--datasets", nargs="+", type=str, 
                        default=["Pascal", "COCO"])
    
    parser.add_argument("--method-dirs", nargs="+", type=str, 
                        default=["textual-inversion-0.5", 
                                 "textual-inversion-mask-0-0.5", 
                                 "textual-inversion-mask-0.5-0"])
    
    parser.add_argument("--baseline-dirs", nargs="+", type=str, 
                        default=["real-guidance-0.5-cap", 
                                 "real-guidance-mask-0-0.5", 
                                 "real-guidance-mask-0.5-0"])
    
    parser.add_argument("--method-names", nargs="+", type=str, 
                        default=["Original", 
                                 "Masked Foreground", 
                                 "Masked Background"])
    
    parser.add_argument("--name", type=str, default="masking-results")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--num-trials", type=int, default=8)
    
    parser.add_argument("--no-legend", action="store_true")

    args = parser.parse_args()

    combined_dataframe = []

    for logdir, dataset in zip(args.logdirs, args.datasets):

        for bname in os.listdir(logdir):

            bpath = os.path.join(logdir, bname)

            if not os.path.isdir(bpath):
                continue

            files = list(glob.glob(os.path.join(bpath, "*.csv")))

            if len(files) == 0:
                continue

            data = pd.concat([pd.read_csv(x, index_col=0) 
                              for x in files], ignore_index=True)

            data = data[(data["metric"] == "Accuracy") & 
                        (data[ "split"] == "Validation")]

            def select_by_epoch(df):
                selected_row = df.loc[df["value"].idxmax()]
                return data[(data["epoch"] == selected_row["epoch"]) & 
                            (data[ "examples_per_class"] == 
                            selected_row["examples_per_class"])]

            best = data.groupby(["examples_per_class", "epoch"])
            best = best["value"].mean().to_frame('value').reset_index()
            best = best.groupby("examples_per_class").apply(
                select_by_epoch
            )

            best["method"] = bname
            best["dataset"] = dataset
            combined_dataframe.append(best)

    matplotlib.rc('font', family='Times New Roman', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')
    plt.rcParams['text.usetex'] = False

    combined_dataframe = pd.concat(
        combined_dataframe, ignore_index=True)

    combined_dataframe = pd.concat([
        combined_dataframe[combined_dataframe['method'] == n] 
        for n in (args.method_dirs + args.baseline_dirs)])
    
    color_palette = sns.color_palette(n_colors=len(args.method_dirs))

    legend_rows = int(math.ceil(len(args.method_names) / (2 * len(args.datasets))))
    columns = int(math.ceil(2 * len(args.datasets) / args.rows))

    fig, axs = plt.subplots(
        args.rows, columns,
        figsize=(6 * columns, 4 * args.rows + ((
            2.0 if legend_rows == 1 else
            2.5 if legend_rows == 2 else 3
        ) if not args.no_legend else 1.0)))

    baseline_performance = defaultdict(list)

    for dataset in args.datasets:
        for seed in range(args.num_trials):
            for bi, method in enumerate(args.baseline_dirs):

                results = combined_dataframe[
                    (combined_dataframe["dataset"] == dataset) & 
                    (combined_dataframe["method"] == method) & 
                    (combined_dataframe["seed"] == seed)
                ]

                for examples in [1, 2, 4, 8, 16]:

                    value = results[results["examples_per_class"] 
                                    == examples]["value"].to_numpy()

                    if value.size > 0: baseline_performance[
                        (dataset, bi, examples)].append(value[0])

                cumulative_value = 0.0
                invalid = False

                for examples_a, examples_b in zip([1, 2, 4, 8], [2, 4, 8, 16]):

                    value_a = results[results["examples_per_class"] == examples_a]["value"].to_numpy()
                    value_b = results[results["examples_per_class"] == examples_b]["value"].to_numpy()

                    if value_a.size > 0 and value_b.size > 0:
                        cumulative_value += ((value_a + value_b) / 2) * (examples_b - examples_a)

                    else: invalid = True

                if not invalid: baseline_performance[
                    (dataset, bi)].append(cumulative_value[0])

    performance_df = []
    performance_auc_df = []

    for dataset in args.datasets:
        for seed in range(args.num_trials):
            for bi, method in enumerate(args.method_dirs):

                results = combined_dataframe[
                    (combined_dataframe["dataset"] == dataset) & 
                    (combined_dataframe["method"] == method) & 
                    (combined_dataframe["seed"] == seed)
                ]

                for examples in [1, 2, 4, 8, 16]:

                    if (dataset, bi, examples) not in baseline_performance: continue

                    value = results[results["examples_per_class"] 
                                    == examples]["value"].to_numpy()

                    baseline_value = np.mean(
                        baseline_performance[(dataset, bi, examples)])

                    if value.size > 0: performance_df.append(dict(
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        examples_per_class=examples,
                        value=value[0] - baseline_value,
                    ))

                if (dataset, bi) not in baseline_performance: continue

                valid = 0
                cumulative_value = -np.mean(baseline_performance[(dataset, bi)])

                for examples_a, examples_b in zip([1, 2, 4, 8], [2, 4, 8, 16]):

                    value_a = results[results["examples_per_class"] == examples_a]["value"].to_numpy()
                    value_b = results[results["examples_per_class"] == examples_b]["value"].to_numpy()

                    if value_a.size > 0 and value_b.size > 0:
                        cumulative_value += ((value_a + value_b) / 2) * (examples_b - examples_a)
                        valid += 1

                if valid == 4:

                    performance_auc_df.append(dict(
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        value=cumulative_value[0],
                    ))

    performance_df = pd.DataFrame.from_records(performance_df)
    performance_auc_df = pd.DataFrame.from_records(performance_auc_df)

    for dataset in args.datasets:

        df = performance_df.loc[performance_df["dataset"] == dataset]
        if df.size == 0: continue

        acc_max = df["value"].to_numpy().max()
        acc_min = df["value"].to_numpy().min()

        performance_df.loc[
            performance_df["dataset"] == dataset, 
            "normalized_value"
        ] = (df["value"] - acc_min) / (acc_max - acc_min)

        df = performance_auc_df.loc[performance_auc_df["dataset"] == dataset]
        if df.size == 0: continue

        acc_max = df["value"].to_numpy().max()
        acc_min = df["value"].to_numpy().min()

        performance_auc_df.loc[
            performance_auc_df["dataset"] == dataset, 
            "normalized_value"
        ] = (df["value"] - acc_min) / (acc_max - acc_min)

    for i, (style, dataset) in enumerate(
            product(["line", "bar"], args.datasets)):

        results = performance_df[
            performance_df["dataset"] == dataset] \
            if dataset != "Overall" else performance_df

        results_auc = performance_auc_df[
            performance_auc_df["dataset"] == dataset] \
            if dataset != "Overall" else performance_auc_df

        if style == "line":

            axis = sns.lineplot(
                y="normalized_value" if dataset == "Overall" else "value", 
                x="examples_per_class", hue="method", 
                data=results, errorbar=('ci', 68),
                linewidth=4, palette=color_palette,
                ax=(axs[i // columns, i % columns] 
                    if args.rows > 1 and len(args.datasets) > 1 
                    else axs[i] if len(args.datasets) > 1 else axs))

            if i == 0: handles, labels = axis.get_legend_handles_labels()
            axis.legend([],[], frameon=False)

            axis.set(xlabel=None)
            axis.set(ylabel=None)

            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)

            axis.xaxis.set_ticks_position('bottom')
            axis.yaxis.set_ticks_position('left')

            axis.yaxis.set_tick_params(labelsize=16)
            axis.xaxis.set_tick_params(labelsize=16)

            if i // columns == args.rows - 1:
                axis.set_xlabel("Examples Per Class", fontsize=24,
                                fontweight='bold', labelpad=12)

            axis.set_ylabel("Normalized Score" if dataset == "Overall" 
                            else "Gained Accuracy (Val)", fontsize=24,
                            fontweight='bold', labelpad=12)

        elif style == "bar":

            axis = sns.barplot(
                y="normalized_value" if dataset == "Overall" else "value", 
                x="method", data=results_auc, errorbar=('ci', 68),
                linewidth=4, palette=color_palette,
                ax=(axs[i // columns, i % columns] 
                    if args.rows > 1 and len(args.datasets) > 1 
                    else axs[i] if len(args.datasets) > 1 else axs))

            if i == 0: handles, labels = axis.get_legend_handles_labels()
            axis.legend([],[], frameon=False)

            axis.set(xlabel=None)
            axis.set(ylabel=None)

            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)

            axis.xaxis.set_ticks_position('bottom')
            axis.yaxis.set_ticks_position('left')

            axis.yaxis.set_tick_params(labelsize=16)
            axis.xaxis.set_ticklabels([])

            acc_max = results_auc["normalized_value" if dataset == "Overall" else "value"].to_numpy().max()
            acc_min = results_auc["normalized_value" if dataset == "Overall" else "value"].to_numpy().min()
            axis.set_ylim(max(0, acc_min), acc_max)

            axis.set_ylabel("Normalized Score" if dataset == "Overall" 
                            else "Gained AUC (Val)", fontsize=24,
                            fontweight='bold', labelpad=12)

        axis.set_title(dataset, fontsize=24, fontweight='bold', pad=12)

        axis.grid(color='grey', linestyle='dotted', linewidth=2)

    if not args.no_legend:

        legend = fig.legend(handles, [x for x in args.method_names],
                            loc="lower center", prop={'size': 24, 'weight': 'bold'}, 
                            ncol=min(len(args.method_names), 2 * len(args.datasets)))

        for i, legend_object in enumerate(legend.legendHandles):
            legend_object.set_linewidth(4.0)
            legend_object.set_color(color_palette[i])

    plt.tight_layout(pad=1.0)
    fig.subplots_adjust(hspace=0.3)

    if not args.no_legend:

        fig.subplots_adjust(bottom=(
            0.25 if legend_rows == 1 else
            0.35 if legend_rows == 2 else 0.4
        ) / args.rows + 0.05)

    plt.savefig(f"{args.name}.pdf")
    plt.savefig(f"{args.name}.png")