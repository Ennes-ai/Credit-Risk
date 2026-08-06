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


    # print("=" *80)
    # print("[bold red] Target Column Value Counts [/bold red] ")
    # print("=" *80)
    # print(DataFrame[Target_Column].value_counts()) # * Target column is not balanced, we need to balance it before training the model, e.g. using SMOTE
    # print("=" *80)

    # sns.heatmap(DataFrame.corr(numeric_only=True), annot=True, cmap="coolwarm")
    # plt.title("Correlation Heatmap")
    # plt.show()

    Dataframe_Filtered = DataFrame[DataFrame["person_income"] < 150000]

    #plt.figure(figsize=(10, 6))
    #sns.histplot(Dataframe_Filtered["loan_amnt"], bins=30, kde=True)
    #plt.show()


     # * filling the missing values in the person_emp_Length column with the median value of that column
     # * Because the filling the missing values with the median method is less sensitive to outliers than the mean
    emp_median = DataFrame["person_emp_length"].median()
    DataFrame["person_emp_length"] = DataFrame["person_emp_length"].fillna(emp_median)

    DataFrame["loan_int_rate"] = DataFrame.groupby("loan_grade")["loan_int_rate"].transform(lambda x: x.fillna(x.median()))

    # print("=" *80)
    # print("[bold red] DataFrame Null Values[/bold red]")
    # print("=" *80)
    # print(DataFrame.isnull().sum())
    print("=" *80)
    print("[bold red] DataFrame Info [/bold red] ")
    print("=" *80)
    print(DataFrame.info())
    print("=" *80)
    print("[bold red] DataFrame Type The Columns [/bold red] ") # ! Clear all the columns data types is correct, we don't need to change any data type
    print("=" *80)
    print(DataFrame.dtypes)
    print("=" *80)

    DataFrame["cb_person_default_on_file"] = DataFrame["cb_person_default_on_file"].map({"Y": 1, "N": 0})
    DataFrame["loan_grade"] = DataFrame["loan_grade"].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7})

    DataFrame = pd.get_dummies(data = DataFrame, columns = ["person_home_ownership" , "loan_intent"] , drop_first = True , dtype = int)

    print("=" *80)
    print("[bold red] DataFrame Head After Encoding [/bold red] ")
    print("=" *80)
    print(DataFrame.head())
    print("=" *80)
    print("[bold red] DataFrame dtypes After Encoding[/bold red] ", DataFrame.dtypes)







if __name__ == "__main__":
    main()