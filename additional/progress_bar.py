import tqdm

##############################################################################
class ProgressBar():
    def __init__(self, n):
        self.pbar = tqdm.tqdm(total = n)
        self.postfix = {}
        
    def set_postfix(self, name:str, postfix:str):
        self.postfix[name] = postfix
        self.pbar.set_postfix(self.postfix)
        return self

    def update(self):
        self.pbar.set_postfix(self.postfix)
        self.pbar.update(1)

        if(self.pbar.n == self.pbar.total):
            self.pbar.close()
    
    def close(self):
        self.pbar.close()

class ProgressFitModel(ProgressBar):
    def __init__(self, n:int, best_value:float):
        super().__init__(n)
        self.postfix['model'] = ""
        self.postfix['best_value'] = best_value

    def update(self, new_value:float, best_model:str):
        if new_value > self.postfix['best_value']:
            self.postfix['best_value'] = new_value
            self.postfix['best_model'] = best_model
        super().update()
