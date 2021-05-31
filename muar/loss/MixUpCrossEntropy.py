import torch
import torch.nn.functional as F

class MixUpCrossEntropy:
    "Cross entropy that works if there is a probability of MixUp being applied."
    def __init__(self, reduction:bool=True):
        """ 
        Args: 
            reduction (bool): True if mean is applied after loss.
        """
        self.reduction = 'mean' if reduction else 'none'
        self.criterion = F.cross_entropy
        
    def __call__(self, logits:torch.Tensor, y:torch.LongTensor):
        """
        Args:
            logits (torch.Tensor): Output of the model.
            y (torch.LongTensor): Targets of shape (batch_size) or (batch_size, 3).
        """
        assert len(y.shape) == 1 or y.shape[1] == 3, 'Invalid targets.'

        if len(y.shape) == 1:
            loss = self.criterion(logits, y, reduction=self.reduction)
        
        elif y.shape[1] == 3:
            loss_a = self.criterion(logits, y[:, 0].long(), reduction=self.reduction)
            loss_b = self.criterion(logits, y[:, 1].long(), reduction=self.reduction)
            loss = ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b)

        if self.reduction == 'mean': return loss.mean()
        return loss