# Authors: Samuel Cahyawijaya
# License: BSD 3 Clause
"""  
PyTorch MatrixFact Semi Non-negative Matrix Factorization.

    TorchSNMF(NMF) : Class for semi non-negative matrix factorization with PyTorch
    
[1] Ding, C., Li, T. and Jordan, M.. Convex and Semi-Nonnegative Matrix Factorizations.
IEEE Trans. on Pattern Analysis and Machine Intelligence 32(1), 45-55. 
"""
import torch
from .base import TorchMatrixFactBase

__all__ = ["TorchSNMF"]


class TorchSNMF(TorchMatrixFactBase):
    def _update_w(self):
        W1 = torch.mm(self.data, self.H.T)
        W2 = torch.mm(self.H, self.H.T)    
        self.W = torch.mm(W1, torch.linalg.inv(W2))
        
    def _update_h(self):
        def separate_positive(m):
            return (m.abs() + m)/2.0 
        
        def separate_negative(m):
            return (m.abs() - m)/2.0
            
        XW = torch.mm(self.data[:,:].T, self.W)                

        WW = torch.mm(self.W.T, self.W)
        WW_pos = separate_positive(WW)
        WW_neg = separate_negative(WW)
        
        XW_pos = separate_positive(XW)
        H1 = (XW_pos + torch.mm(self.H.T, WW_neg)).T
        
        XW_neg = separate_negative(XW)
        H2 = (XW_neg + torch.mm(self.H.T, WW_pos)).T + 10**-9
        
        self.H *= torch.sqrt(H1/H2)    