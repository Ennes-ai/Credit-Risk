import pandas as pd
import numpy as np
from rich import print


class Calculate:
    def __init__(self, DataFrame):
        self.DataFrame = DataFrame

    @staticmethod
    def Mean(Data: list | pd.Series) -> float:
        """This function calculates the mean of a list of numbers or pandas
        Series.

        :param Data: list of numbers or pandas series
        :return: mean float value
        """
        
        if isinstance(Data, pd.Series):
            Data = Data.dropna().tolist()
        elif isinstance(Data, list):
            Data = [x for x in Data if x is not None]
        else:
            raise TypeError("Data must be a list or pandas series")

        
        r = len(Data)

        if r == 0:
            raise ValueError("Doesn't calculate a mean of an empty list.")

        
        return sum(Data) / r

    @staticmethod
    def median(Data: list | pd.Series) -> float:
        """This function calculates the median of a list of numbers or pandas
        Series.

        :param Data: list of numbers or pandas series
        :return: median float value
        """
       
        if isinstance(Data, pd.Series):
            
            Data = Data.dropna().sort_values().tolist()
        elif isinstance(Data, list):
           
            Data = sorted(Data)
        else:
            raise TypeError("Data must be a list or pandas series")

        
        r = len(Data)

        if r == 0:
            raise ValueError("Doesn't calculate a mean of an empty list.")

        
        if r % 2 == 0:
            return (Data[r // 2] + Data[r // 2 - 1]) / 2.0
        else:
            return float(Data[r // 2])
        
    def get_igr(self, column_name : str)-> dict:
        """This function calculates the interquartile range (IQR) of a specified column in the DataFrame.

        :param column_name: The name of the column for which to calculate the IQR
        :return: The IQR value
        """
        if column_name not in self.DataFrame.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        try:
            Q1 = self.DataFrame[column_name].quantile(0.25) # ! take the first quartile (25th percentile)
            Q3 = self.DataFrame[column_name].quantile(0.75) # ! take the third quartile (75th percentile)
            IQR = Q3 - Q1 

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            return {
                "Q1" : Q1,
                "Q3" : Q3,
                "IQR" : IQR,
                "lower_bound" : lower_bound,
                "upper_bound" : upper_bound
            }
        except Exception as e:
            raise ValueError(f"Error calculating IQR for column '{column_name}': {e}")


    def get_outliers(self, column_name: str)-> pd.DataFrame:
        """This function identifies outliers in a specified column of the DataFrame based on the IQR method.

        :param column_name: The name of the column for which to identify outliers
        :return: A DataFrame containing the outliers
        """
        if column_name not in self.DataFrame.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        try:
            iqr_values = self.get_igr(column_name)
            lower_bound = iqr_values["lower_bound"]
            upper_bound = iqr_values["upper_bound"]

            outliers = self.DataFrame[(self.DataFrame[column_name] < lower_bound) | (self.DataFrame[column_name] > upper_bound)]
            return outliers
        except Exception as e:
            raise ValueError(f"Error  '{column_name}': {e}")