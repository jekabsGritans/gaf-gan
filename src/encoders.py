from abc import ABC, abstractmethod
import torch
import numpy as np

class Encoder(ABC):

    @abstractmethod
    def encode(self, x) -> torch.Tensor:
        """Encode batch."""

    @abstractmethod
    def decode(self, x) -> np.ndarray:
        """Decode a single item."""

    
class SimpleRasterizeEncoder(Encoder):
    """Broadcast array as rows to fill matrix."""
    def encode(self, x) -> torch.Tensor:
        """Encode batch."""
        x = x.view(-1,1,x.size(1))
        mat = x.repeat(1,x.size(2),1)
        return mat.view(-1,1,mat.size(1), mat.size(2))
    
    def decode(self, x) -> np.ndarray:
        """Decode a single item.""" 
        return x[0,0].detach().cpu().numpy().squeeze()

    def decode_noisy(self, x) -> np.ndarray:
        """Decode a single item."""
        return x.mean(dim=1).detach().cpu().numpy().squeeze()


class GasfEncoder(Encoder):
    """Gramian Angular Summation Field."""

    def encode(self, x) -> torch.Tensor:
        """Encode batch."""
        if not (x.max()<=1 and x.min()>=0):
            raise ValueError("x must be in [0,1]")
        x = torch.arccos(x)
        x = x.view(-1,1,x.size(1))+x.view(-1,x.size(1),1)
        x = torch.cos(x)
        return x.view(-1,1,x.size(1), x.size(2))

    def decode(self, x) -> np.ndarray:
        """Decode a single item."""
        mat = x[0]
        vals = torch.diagonal(mat, dim1=0, dim2=1)
        log_prices = torch.cos((torch.arccos(vals)/2)).squeeze().detach().cpu().numpy()
        return log_prices

    def decode_noisy(self, x) -> np.ndarray:
        """Decode a single item."""
        a = x.shape[0]
        S = x.sum()/(2*a)
        vals = torch.zeros(a)
        for n in range(x.shape[0]):
            vals[n] = (x[n,:].sum()+x[:,n].sum()-2*S)/(2*a)
        return vals

class NegGasfEncoder(Encoder):
    """GASF with the ability to encode (-1,1)."""

    def encode(self, x) -> torch.Tensor:
        """Encode batch."""
        if not x.abs().max()<1:
            raise ValueError("x must be in (-1,1)")
        x = torch.arccos(x)
        x = x.view(-1,1,x.size(1))+x.view(-1,x.size(1),1)
        x = torch.cos(x)
        return x.view(-1,1,x.size(1), x.size(2))
    
    def decode(self, x) -> np.ndarray:
        """Decode a single item."""
        mat = torch.arccos(x[0][0])
        diag = torch.diagonal(mat, dim1=0, dim2=1)
        negative_condition = (2*mat[0,:]-mat[0,0]-diag).abs()>1e-6 # B[0,i] - B[0,0]/2 != B[i,i]/2
        vals = torch.cos(diag/2) * (1-2*negative_condition.float()) # reverse sign if negative condition is true
        return vals
