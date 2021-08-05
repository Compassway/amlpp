class ProgressBar():
    def __init__(self, n):
        self.pbar = tqdm.tqdm(total = n)
        self.postfix = ""
    def set_postfix(self, postfix:str):
        self.postfix = postfix
        
    def up(self):
        self.pbar.set_postfix({'model':self.postfix}, refresh=True)
        self.pbar.update(1)