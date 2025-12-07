import torch.nn as nn
class MultiSequential(nn.Module):
    """
    A class used to represent a Sequential module that is similar to nn.Sequential but takes multiple constant/same arguments
    alongside input in the forward function
    """
    def __init__(self, *layers):
      super().__init__()
      self.layers = nn.ModuleList(layers)

    # forward function can now receive the same extra arguments
    def forward(self, x, *extra):
      for layer in self.layers:
          x = layer(x, *extra)
      return x