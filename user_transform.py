import pandas as pd
from typing import List

class UserTransform:
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame, y:pd.Series or List[float or int]):
        return self

    def transform(self, X:pd.DataFrame, y = None) -> pd.DataFrame:
        X = X[['closed_credits_count', 'ubki_week_queries','loan_amount','loan_days','ubki_email_deltatime','ubki_phone_deltatime','ubki_maxnowexp','ubki_expyear']]
        X = X.fillna(0)
        return X

    def target_transform(self, dataset:pd.DataFrame) -> pd.Series or List[float or int]:
        y = dataset['user_id']
        return y

# def target_transform(dataset:pd.DataFrame):
#     y = dataset
#     return y