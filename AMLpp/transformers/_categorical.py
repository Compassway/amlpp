from ._base import BaseTransform

from sklearn.preprocessing import OrdinalEncoder

from typing import List

import pandas as pd
import numpy as np

__all__ = ["echo", "surround", "reverse"]

class CategoricalEncoder(BaseTransform):

    """ Класс кодирования категориальных данных, с заполнение пропусков на некоторое значение определенное сратегией

    Parameters
    ----------
    columns : List[str]
        Названия столбцов, которые будут подвегнуты обработке

    straegy : str
        Строка указывающая на используемую стратегию заполнения пропусков
    
    fill_value : float or str
        Заполнитель, которым будут заполняться пропущенные значения,
        при использование стратегии const
        
    """    

    def __init__(self, columns:List[str]):
        super().__init__({'columns':columns})
        self.encoder = {column: OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = np.nan) for column in columns}

    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        for column in self.encoder.copy():
            if column in X.columns:
                X_fit = pd.DataFrame(X[column].loc[~X[column].isnull()])
                if len(X_fit) > 0:
                    self.encoder[column].fit(X_fit)
                else:
                    self.encoder[column] = False
        return self

    def transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series = None):
        for column in self.encoder:
            if column in X.columns:
                if self.encoder[column]:
                    X[column] = self.encoder[column].transform(pd.DataFrame(X[column].fillna('NAN')))
                else:
                    del X[column]
        return X