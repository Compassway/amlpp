from sklearn.preprocessing import OrdinalEncoder

from typing import List
import pandas as pd
import numpy as np

from ._base_transform import BaseTransform

##############################################################################
class CategoricalEncoder(BaseTransform):
    """ Categorical encoder 
    Parameters
    ----------
    columns: List [str] 
        Columns that encode 
    """ 
    def __init__(self, columns:List[str]):
        super().__init__({'columns':columns})
        self.order_columns = None
        self.encoder = {column: OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = np.nan) for column in columns}

    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        for column in self.encoder.copy():
            if column in X.columns:
                X_fit = pd.DataFrame(X[column].loc[~X[column].isnull()])
                if len(X_fit) > 0:
                    self.encoder[column].fit(X_fit)
                else:
                    self.encoder[column] = False
        self.order_columns = X.columns
        return self

    def transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series = None):
        for column in self.encoder:
            if column in X.columns:
                if self.encoder[column]:
                    X[column] = self.encoder[column].transform(pd.DataFrame(X[column].fillna('NAN')))
                else:
                    del X[column]
        X = X[self.order_columns]
        return X