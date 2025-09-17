import torch.nn as nn
from selfAttentionHead import MultiHeadSelfAttention
from feedForward import FeedForward

class EncoderBlock(nn.Module):
  """
  A class used to represent a transformer block to repeat: communication (attention) followed by computation (feedforward)
  """
  
  # nEmbed is the dimension of the inputs (previously calculated) into self-attention (embedding dimension)
  # required: nEmbed = headSize (input into output)
  # blockSize T: number of time, sequential characters in a context chunk
  def __init__(self, headSize, numHeads, nEmbed, blockSize, dropout):
    super().__init__()
    # numHeads heads of smaller one head of self-attention models to apply multiple parallel one head of self-attentions (communication)
    self.saHeads = MultiHeadSelfAttention(numHeads, headSize, nEmbed, blockSize, dropout, False) # (B, T, headSize)
    # a simple feed forward network (computation)
    self.feedForward = FeedForward(headSize, dropout) # results are same dimensions: (B, T, headSize)
    # layer norm to normalize (subtract mean divide by std) rows (all features within a single data point in a batch) to N(0, 1) and scale (gamma) and shift (beta) 
    # contrast to batch normalization which normalizes column (singular feature/neuron across batch dimension) to N(0, 1) and scale (gamma) and shift (beta)
    # gamma and beta are learnable parameters, this improves training stability and speed
    # batch/blockSize/time act as batch dimensions (per token transformation, normalizes the features into unit N(0, 1))
    self.layerNorm1 = nn.LayerNorm(headSize) # results are same dimensions: (B, T, headSize)
    self.layerNorm2 = nn.LayerNorm(headSize) # results are same dimensions: (B, T, headSize)
  
  # one forward pass of the block (B, T, C/nEmbed/headSize) -> (B, T, headSize)
  def forward(self, x):
    # apply multiple heads of self-attention 
    # perform residual connections (shortcuts) which allow bypassing multiple layers to optimize deep neural networks
    # Mitigates vanishing gradient problem (early layers have trouble learning due to diminishing gradients as they propogate backwards many layers)
    # We allow residual connection to skip and allow direct path for gradients to flow thus preventing vanshing
    # fork off do calculations (not taking the shortcut), comeback and add (project) to original input (residual pathway/connection)
    # network learns the residuals (difference between input and output) rather than the output itself
    # apply layer norm before transformation (changed from the original transformer model)
    x = x + self.saHeads(self.layerNorm1(x)) # (B, T, C/nEmbed/headSize) (residual pathway) + (B, T, headSize) (fork off) = (B, T, headSize)
    # apply a feed forward network
    x = x + self.feedForward(self.layerNorm2(x))  # (B, T, C/nEmbed/headSize) (residual pathway) + (B, T, headSize) (fork off) = (B, T, headSize)
    return x # (B, T, headSize)
