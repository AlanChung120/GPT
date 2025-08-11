import torch.nn as nn
from attentionHead import MultiHeadAttention
from feedForward import FeedForward

class Block(nn.Module):
  """
  A class used to represent a transformer block to repeat: communication (attention) followed by computation (feedforward)
  """
  
  # nEmbed is the dimension of the inputs (previously calculated) into self-attention (embedding dimension)
  # required: nEmbed = headSize (input into output)
  # blockSize T: number of time, sequential characters in a context chunk
  def __init__(self, headSize, numHeads, nEmbed, blockSize):
    super().__init__()
    # numHeads heads of smaller one head of self-attention models to apply multiple parallel one head of self-attentions (communication)
    self.saHeads = MultiHeadAttention(numHeads, headSize, nEmbed, blockSize) # (B, T, headSize)
    # a simple feed forward network (computation)
    self.feedForward = FeedForward(headSize) # results are same dimensions: (B, T, headSize)
  
  # one forward pass of the block (B, T, C/nEmbed/headSize) -> (B, T, headSize)
  def forward(self, x):
    # apply multiple heads of self-attention 
    # perform residual connetions (shortcuts) which allow bypassing multiple layers to optimize deep neural networks
    # Mitigates vanishing gradient problem (early layers have trouble learning due to diminishing gradients as they propogate backwards many layers)
    # We allow residual connection to skip and allow direct path for gradients to flow thus preventing vanshing
    # fork off do calculations (not taking the shortcut), comeback and add (project) to original input (residual pathway/connection)
    # network learns the residuals (difference between input and output) rather than the output itself
    x = x + self.saHeads(x) # (B, T, C/nEmbed/headSize) (residual pathway) + (B, T, headSize) (fork off) = (B, T, headSize)
    # apply a feed forward network
    x = x + self.feedForward(x)  # (B, T, C/nEmbed/headSize) (residual pathway) + (B, T, headSize) (fork off) = (B, T, headSize)
    return x # (B, T, headSize)
