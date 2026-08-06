import pandas as pd
import numpy as np
from rich import print


class Calculate:
    def __init__(self, DataFrame):
        self.DataFrame = DataFrame


    @staticmethod
    def median(l : list,r : int) -> float:
        """
        This function calculates the median of a list of numbers.
        :param l: list of numbers
        :param r: length of the list
        :return: median of the list
        """
        if r % 2 == 0:
            return (l[r // 2] + l[r // 2 - 1]) / 2
        else:
            return l[r // 2]
        
    @staticmethod
    def IQR(l : list) -> tuple:
        """
        This function calculates the interquartile range (IQR) of a list of numbers.
        :param l: list of numbers
        :return: IQR of the list , Q1 and Q3
        """
        l.sort()
        r = len(l)
        q1 = Calculate.median(l[:r // 2], r // 2)
        q3 = Calculate.median(l[(r + 1) // 2:], r // 2)
        iqr = q3 - q1
        return q1, q3, iqr