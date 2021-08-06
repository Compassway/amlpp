import tqdm

##############################################################################
class ProgressBar():
    def __init__(self, n):
        self.pbar = tqdm.tqdm(total = n)
        self.postfix = {}
        
    def set_postfix(self, name:str, postfix:str):
        self.postfix[name] = postfix
        return self
    def update(self):
        self.pbar.set_postfix(self.postfix, refresh=True)
        self.pbar.update(1)

        if(self.pbar.n == self.pbar.total):
            self.pbar.close()

class ProgressFitModel(ProgressBar):
    def __init__(self, n):
        super().__init__(n)
        self.postfix['model'] = ""
        self.postfix['best_value'] = 0.

    def update(self, new_value:float, best_model:str):
        if new_value > self.postfix['best_value']:
            self.postfix['best_value'] = new_value
            self.postfix['best_model'] = best_model
        super().update()