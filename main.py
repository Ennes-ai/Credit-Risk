import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print
from IQR import Calculate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc


def create_model(DataFrame : pd.DataFrame , Target_Column : str = None):
    X = DataFrame.drop(columns=[Target_Column])
    y = DataFrame[Target_Column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 , stratify=y)

    RFC = RandomForestClassifier(n_estimators=100, random_state=42 , n_jobs=-1)
    RFC.fit(X_train, y_train)

    y_pred = RFC.predict(X_test)
    y_proba = RFC.predict_proba(X_test)[:, 1]

    c_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    sns.heatmap(c_matrix, annot=True, fmt="d", cmap="Blues")
    plt.show()



    print("=" *80)
    print("[bold red] Model Accuracy [/bold red]  ")
    print("=" *80)
    print(accuracy_score(y_test, y_pred))
    print("=" *80)

    print("=" *80)
    print("[bold red] Classification Report [/bold red]  ")
    print("=" *80)
    print(classification_report(y_test, y_pred))
    print("=" *80)

    print("=" *80)
    print("[bold red] Confusion Matrix [/bold red]  ")
    print("=" *80)
    print(c_matrix)
    print("=" *80)

    print("=" *80)
    print("[bold red] ROC AUC Score [/bold red]  ")
    print("=" *80)
    print(roc_auc)
    print("=" *80)

    print("=" *80)
    print("[bold red] ROC Curve [/bold red]  ")
    print("=" *80)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    print("=" *80)

    """
    ROC eğrisi, mümkün olan her eşikte (pratikte, seçilen aralıklarda) gerçek pozitif oranı 
    (TPR) ve yanlış pozitif oranı (FPR) hesaplanarak ve ardından TPR'nin FPR'ye göre grafiği çizilerek oluşturulur
    """

    train_auc = roc_auc_score(y_train, RFC.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, RFC.predict_proba(X_test)[:, 1])

    print(f"Eğitim Seti ROC-AUC : {train_auc:.4f}")
    print(f"Test Seti ROC-AUC   : {test_auc:.4f}")


    feature_importances = pd.Series(
        RFC.feature_importances_ , index = X.columns
    ).sort_values(ascending = False)

    plt.figure(figsize = (10,6))
    sns.barplot(x = feature_importances , y = feature_importances.index , palette = "viridis")
    plt.title("Feature İmportanes")
    plt.xlabel("feature imp.")
    plt.ylabel("index")
    plt.show()




def visualize_data(DataFrame: pd.DataFrame , Target_Column: str = None):
    for column in DataFrame.columns:
        if DataFrame[column].dtype in ["int64", "float64"]:
            plt.figure(figsize=(10, 6))
            sns.histplot(DataFrame[column], bins=30, kde=True)
            plt.title(f"Distribution of {column}")
            plt.show()
            
"""
person_income, person_age, person_emp_length 

gibi grafiklerin hepsi Sağa Çarpık dağılımdır.
Sağa Çarpık Dağılımın Altın Kuralı:Kuyruk sağdaki büyük sayılara doğru uzadığı için ortalamayı (mean) yukarı çeker.
Bu grafiklerde her zaman: Ortalama (Mean) > Medyan (Median) çıkar.
"""

def detect_outliers():
    
    # ! just take the continuous columns to find the outliers
    continuous_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate"
    ]

    print("=" * 80)
    print("[bold red] Outliers in DataFrame [/bold red]")
    print("=" * 80)

  
    for column in continuous_cols:
        outliers = calculate.get_outliers(column)
        outlier_count = len(outliers)
        
       
        percentage = (outlier_count / len(DataFrame)) * 100
        
        print(f"📌 [bold cyan]{column:20}[/bold cyan] : [bold red]{outlier_count:5d}[/bold red] outliers count (%{percentage:.2f})")

    print("=" * 80)

def main():
    global DataFrame, calculate
    DataFrame = pd.read_csv("credit_risk_dataset.csv")
    calculate = Calculate(DataFrame = DataFrame)

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


    


     # * filling the missing values in the person_emp_Length column with the median value of that column
     # * Because the filling the missing values with the median method is less sensitive to outliers than the mean
    emp_median = DataFrame["person_emp_length"].median()
    DataFrame["person_emp_length"] = DataFrame["person_emp_length"].fillna(emp_median)

    DataFrame["loan_int_rate"] = DataFrame.groupby("loan_grade")["loan_int_rate"].transform(lambda x: x.fillna(x.median()))

    detect_outliers()
    # ! claer the outliers in the continuous columns
    DataFrame = DataFrame[DataFrame["person_age"] < 100]
    DataFrame = DataFrame[DataFrame["person_emp_length"] < 60]

    income_upper_limit = DataFrame["person_income"].quantile(0.99)
    DataFrame["person_income"] = np.where(DataFrame["person_income"] > income_upper_limit, income_upper_limit, DataFrame["person_income"])


    loan_cap = DataFrame["loan_amnt"].quantile(0.99)
    DataFrame["loan_amnt"] = np.where(DataFrame["loan_amnt"] > loan_cap, loan_cap, DataFrame["loan_amnt"])


    print("=" * 80)
    print("[bold green] Temizlik Sonrası Maksimum Değerler [/bold green]")
    print("Maksimum Yaş:          ", DataFrame["person_age"].max())
    print("Maksimum Gelir:        ", DataFrame["person_income"].max())
    print("Maksimum Kredi Tutarı: ", DataFrame["loan_amnt"].max())
    print("=" * 80)

    sns.histplot(DataFrame["loan_amnt"], bins=30, kde=True)
    plt.title("after the outlier removal")
    plt.show()


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


    #visualize_data(DataFrame= DataFrame, Target_Column=Target_Column)

    sns.scatterplot(data=DataFrame, x="person_age", y="cb_person_cred_hist_length", hue="loan_status")
    plt.xlabel("person_age")
    plt.ylabel("cb_person_cred_hist_length")
    plt.title("cb_person_cred_hist_length vs person_age")
    plt.show()
    print("=" *80)
    print("[bold red] DataFrame Null Values[/bold red]")
    print("=" *80)
    print(DataFrame.isnull().sum().sum() if DataFrame.isnull().sum().sum() > 0 else "[bold green] No Null Values in the DataFrame[/bold green]")
    print("=" *80)


    # ! Feature Engineering

   # 1. Kredi Geçmişi / Yaş Oranı 
    DataFrame["cred_hist_to_age_ratio"] = DataFrame["cb_person_cred_hist_length"] / DataFrame["person_age"]
    DataFrame.drop(columns=["cb_person_cred_hist_length"], inplace=True)

    # 2. Toplam Faiz Maliyeti 
    DataFrame["total_interest_cost"] = DataFrame["loan_amnt"] * (DataFrame["loan_int_rate"] / 100)

    # 3. Çalışma Yılı Başına Düşen Kredi Tutarı 
    DataFrame["loan_to_emp_ratio"] = DataFrame["loan_amnt"] / (DataFrame["person_emp_length"] + 1)

    # 4. İş Tecrübesi / Yaş Oranı 
    DataFrame["emp_to_age_ratio"] = DataFrame["person_emp_length"] / DataFrame["person_age"]


    print("=" *80)
    print("[bold red] DataFrame group by loan_status and describe the cred_hist_to_age_ratio column[/bold red] ")
    print("=" *80)
    print("=" * 80)
    print(DataFrame.groupby("loan_status")["cred_hist_to_age_ratio"].describe())
    print("=" * 80)
    print(DataFrame["cred_hist_to_age_ratio"].describe())
    print("=" *80)

    # ! Encoding the categorical columns to numerical values

    DataFrame["cb_person_default_on_file"] = DataFrame["cb_person_default_on_file"].map({"Y": 1, "N": 0})
    DataFrame["loan_grade"] = DataFrame["loan_grade"].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7})

    DataFrame = pd.get_dummies(data = DataFrame, columns = ["person_home_ownership" , "loan_intent"] , drop_first = True , dtype = int)

    print("=" *80)
    print("[bold red] DataFrame Head After Encoding [/bold red] ")
    print("=" *80)
    print(DataFrame.head())
    #print("=" *80)
    #print("[bold red] DataFrame dtypes After Encoding[/bold red] ", DataFrame.dtypes) # ! Encoding is over all column is to be correct data type is int64

    create_model(DataFrame = DataFrame,
                 Target_Column = Target_Column)


if __name__ == "__main__":
    main()