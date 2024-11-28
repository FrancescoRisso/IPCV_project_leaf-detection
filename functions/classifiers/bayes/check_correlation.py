import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt

sns.set_style("darkgrid")


def BAYES_check_correlation() -> None:
    """
    Loads the csv version of the training data, and shows its correlation
    matrix
    """
    data = pd.read_csv("./dataset/alldata.csv")
    corr = data.iloc[:, :].corr(method="pearson")
    cmap = sns.diverging_palette(250, 354, 80, 60, center="dark", as_cmap=True)
    sns.heatmap(corr, vmax=1, vmin=-1, cmap=cmap, square=True, linewidths=0.2)
    plt.show()


def BAYES_check_ABS_correlation() -> None:
    """
    Loads the csv version of the training data, and shows its correlation
    matrix
    """
    data = pd.read_csv("./dataset/alldata.csv")
    corr = data.iloc[:, :].corr(method="pearson")
    cmap = sns.light_palette(color="black", as_cmap=True)
    sns.heatmap(abs(corr), vmax=1, vmin=0, cmap=cmap, square=True, linewidths=0.2)
    plt.show()
