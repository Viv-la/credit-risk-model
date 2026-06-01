import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(df, column):
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], bins=50, kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()