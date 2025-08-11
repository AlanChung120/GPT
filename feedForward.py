import torch.nn as nn

class FeedForward(nn.Module):
  """
  A class used to represent a network with a simple linear layer followed by a non-linearity and linear projection (per token level (independently)) 
    Allows tokens to think/compute on the data gathered from the self-attention (communication between tokens)
  """

  def __init__(self, nSize):
    super().__init__()
    # network with a linear layer follwed by a non-linearity (multi-layer perceptron) and linear projection
    self.network = nn.Sequential(
      nn.Linear(nSize, 4 * nSize), # increase the inner dimension (residual block on the side of the residual pathway) by 4
      nn.ReLU(),
      nn.Linear(4 * nSize, nSize), # Linear Projection of the outcome back into the residual pathway
    )
  
  # forward pass of the network
  def forward(self, x):
    return self.network(x)