class BaseTransform():
    def __init__(self, params:dict):
        self.attr = params
        
    def __repr__(self):
        repr = self.__class__.__name__ + "("
        for param in self.attr:
            repr += "{}={}, ".format(param, str(self.attr[param]))
        repr = repr[:-2] + ")"
        return repr