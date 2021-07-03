from typing import List
import pandas as pd

class Conveyor:

    def __init__(self, *blocks, **params):
        print("__init__")
        self.blocks = blocks
        pass

    def train(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        for i in self.blocks:
            i.fit(X, Y)

    def export(self):
        pass