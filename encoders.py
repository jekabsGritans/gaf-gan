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
        return x.mean(dim=1).detach().cpu().numpy().squeeze()


def _gasf(x):
    """Gramian Angular Summation Field"""
    x = torch.arccos(x)
    x = x.view(-1,1,x.size(1))+x.view(-1,x.size(1),1)
    x = torch.cos(x)
    return x.view(-1,1,x.size(1), x.size(2))

def _gadf(x):
    """Gramian Angular Difference Field"""
    x = torch.asin(x)
    x = x.view(-1,1,x.size(1))-x.view(-1,x.size(1),1)
    x = torch.sin(x)
    return x.view(-1,1,x.size(1), x.size(2))


class GasfEncoder(Encoder):
    """Gramian Angular Summation Field."""

    def encode(self, x) -> torch.Tensor:
        """Encode batch."""
        return _gasf(x)
    
    def decode(self, x) -> np.ndarray:
        """Decode a single item."""
        a = x.shape[0]
        S = x.sum()/(2*a)
        vals = torch.zeros(a)
        for n in range(x.shape[0]):
            vals[n] = (x[n,:].sum()+x[:,n].sum()-2*S)/(2*a)
        return vals


class RGGafEncoder(Encoder):
    """GASF and GADF stacked (for [-1;1] injection)."""

    def encode(self, x) -> torch.Tensor:
        """Encode batch."""
        c1 = _gasf(x)
        c2 = _gadf(x)
        return torch.stack((c1,c2),dim=1).view(-1,2,x.size(1),x.size(1))
    
    def decode(self, x) -> np.ndarray:
        """Decode a single item."""
        raise NotImplementedError("Not implemented.")