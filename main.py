import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print



def main():
    DataFrame = pd.read_csv("credit_risk_dataset.csv")

    print("=" *80)
    print("[bold red] DataFrame Head [/bold red] \n ", DataFrame.head(20))
    print("=" *80)
    print("[bold red] DataFrame Describe [/bold red] \n ", DataFrame.describe())
    print("=" *80)
    #print("[bold red] DataFrame shape [/bold red] \n ", DataFrame.shape)
    #print("=" *80)

    # ! This Data set have 32581 row and 12 columns

    Target_Column = "loan_status"


    print("=" *80)
    print("[bold red] Target Column Value Counts [/bold red] ")
    print("=" *80)
    print(DataFrame[Target_Column].value_counts()) # * Target column is not balanced, we need to balance it before training the model, e.g. using SMOTE
    print("=" *80)

    sns.heatmap(DataFrame.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    Dataframe_Filtered = DataFrame[DataFrame["person_income"] < 150000]

    plt.figure(figsize=(10, 6))
    sns.histplot(Dataframe_Filtered["loan_amnt"], bins=30, kde=True)
    plt.show()






if __name__ == "__main__":
    main()