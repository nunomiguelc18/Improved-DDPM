import torch
class Normalize:
    def __init__(self, range_lower_bound:int, range_upper_bound:int):
        self.min = range_lower_bound
        self.max = range_upper_bound

    @staticmethod
    def standardize_data(data : torch.Tensor) -> torch.Tensor:
        return (data - torch.mean(data))/torch.std(data)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        scale_data = Normalize.standardize_data(data.to(torch.float))
        return self.min + ((scale_data - torch.min(scale_data))/(torch.max(scale_data)- torch.min(scale_data))) * (self.max - self.min)

    

    


        
