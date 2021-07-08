from itertools import takewhile
from sklearn.model_selection import train_test_split

class Split():
    """This class will drop the target column
    and will create X and y data"""
    def __init__(self, data):
        self.data = data

    
    def X_and_y(self, target):
        """Will drop the target  column and return X, and y"""
        try:
            self.X = self.data.drop(target, axis=1)
            self.y = self.data[target]
            return self.X, self.y
        except Exception as ex:
            print('split module error' + str(ex))
    