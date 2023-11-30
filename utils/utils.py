import numpy as np

class Normalize:
    def __init__(self, range_lower_bound:int, range_upper_bound:int):
        self.min = range_lower_bound
        self.max = range_upper_bound

    @staticmethod
    def standardize_data(data : np.ndarray):
        return (data - np.mean(data))/np.std(data)

    def transform(self, data: np.ndarray):
        scale_data = Normalize.standardize_data(data)
        return self.min + ((scale_data - np.min(scale_data))/(np.max(scale_data)- np.min(scale_data))) * (self.max - self.min)

    

    


        
