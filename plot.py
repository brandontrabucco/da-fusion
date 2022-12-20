import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns

import os
import glob
import argparse


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

    parser.add_argument("--logdirs", nargs="+", type=str, default=[
        "./spurge-baselines", "./coco-baselines", "./imagenet-baselines", "./pascal-baselines"])
    
    parser.add_argument("--datasets", nargs="+", type=str, 
                        default=["Spurge", "COCO", "ImageNet", "Pascal"])
    
    parser.add_argument("--method-names", nargs="+", type=str, 
                        default=["baseline", "real-guidance-0.5", "textual-inversion-0.5"])

    args = parser.parse_args()

    combined_dataframe = []

    for dataset, logdir in zip(
            args.datasets, args.logdirs):

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

    combined_dataframe = pd.concat([combined_dataframe[
        combined_dataframe['method'] == n] for n in args.method_names])
    
    color_palette = sns.color_palette(n_colors=len(args.method_names))

    fig, axs = plt.subplots(1, len(args.datasets),
                            figsize=(6 * len(args.datasets), 6))

    for i, dataset in enumerate(args.datasets):

        results = combined_dataframe[combined_dataframe["dataset"] == dataset]

        axis = sns.lineplot(x="examples_per_class", y="value", hue="method", 
                            data=results, errorbar=('ci', 68),
                            linewidth=4, palette=color_palette,
                            ax=axs[i] if len(args.datasets) > 1 else axs)

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

        axis.set_xlabel("Examples Per Class", fontsize=24,
                        fontweight='bold', labelpad=12)

        axis.set_ylabel("Accuracy (Val)", fontsize=24,
                        fontweight='bold', labelpad=12)

        axis.set_title(f"Dataset = {dataset}",
                       fontsize=24, fontweight='bold', pad=12)

        axis.grid(color='grey', linestyle='dotted', linewidth=2)

    legend = fig.legend(handles, [pretty(x) for x in args.method_names],
                        loc="lower center", ncol=len(args.method_names),
                        prop={'size': 24, 'weight': 'bold'})

    for i, legend_object in enumerate(legend.legendHandles):
        legend_object.set_linewidth(4.0)
        legend_object.set_color(color_palette[i])

    plt.tight_layout(pad=1.0)
    fig.subplots_adjust(bottom=0.35)

    plt.savefig("visualization.pdf")
    plt.savefig("visualization.png")