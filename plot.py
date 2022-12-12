import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os
import glob
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Few-Shot Baseline")
    parser.add_argument("--logdir", type=str, default="./baselines")
    args = parser.parse_args()

    combined_dataframe = []

    for bname in os.listdir(args.logdir):

        bpath = os.path.join(args.logdir, bname)

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
        combined_dataframe.append(best)

    combined_dataframe = pd.concat(
        combined_dataframe, ignore_index=True)

    plt.figure(figsize=(8, 4))
    fig = sns.lineplot(x="examples_per_class", 
                       y="value", hue="method", 
                       data=combined_dataframe)

    sns.move_legend(fig, "center left", bbox_to_anchor=(1.05, 0.45))

    plt.ylabel("Accuracy (Val)")
    plt.xlabel("Examples Per Class")

    plt.tight_layout()
    plt.savefig(os.path.join(
        args.logdir, f"barplot.png"))