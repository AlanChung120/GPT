import torch.nn as nn

class FeedForward(nn.Module):
  """
  A class used to represent a network with a simple linear layer followed by a non-linearity (per token level (independently))
    Allows tokens to think/compute on the data gathered from the self-attention (communication between tokens)
  """

  def __init__(self, nSize):
    super().__init__()
    # network with a linear layer follwed by a non-linearity (multi-layer perceptron)
    self.network = nn.Sequential(
      nn.Linear(nSize, nSize),
      nn.ReLU(),
    )
  
  # forward pass of the network
  def forward(self, x):
    return self.network(x)