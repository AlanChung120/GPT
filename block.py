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
    x = x + self.saHeads(x) # (B, T, C/nEmbed/headSize) + (B, T, headSize) = (B, T, headSize)
    # apply a feed forward network
    x = x + self.feedForward(x)  # (B, T, C/nEmbed/headSize) + (B, T, headSize) = (B, T, headSize)
    return x # (B, T, headSize)
