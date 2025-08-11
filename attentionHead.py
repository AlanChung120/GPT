import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337) # set seed for consistency

class AttentionHead(nn.Module):
  """
  A class used to represent a one head of self-attention (self because keys and values all come from the same source as queries (x) otherwise it is cross-attention)
    Attention is communication mechanism of a directed graph that aggregates via the weighted sum from all nodes that point to them (all the preceding tokens in our case)
    no notion of space, just set of vectors and we postionally encode them (positionEmbeddingTable)
    - self-attention (keys and values are produced from the same source as queries) 
    - cross-attention (queries are produced from x but keys and values come from an external source (encoder module))
    - encoder block(all tokens communciate with each other, need for sentiment of sentence)
    - decoder block (triangular masking to prevent communcation from the future, used in language modelling/autoregressive to "decode")
    instead of equal weighted aggregation we implement a batch/data/token dependent affinity (how related, how much to aggregate/learn) matrix aggregation
    each token has a query vector (what I am looking for based on token identity and position) and a key vector (what I contain (token identity and position))
    affinity of token x with token y (y precedes x) = dot product between query vector of x with key vector of y
    if key and query vector match (key is what query is looking for) higher the dot product (same high components), higher the affinity, higher the weight (relative))
  """

  def __init__(self, headSize, nEmbed, blockSize):
    super().__init__()
    # headSize is the size of the key/query vectors
    # Arbitrary module to learn the weights to convert to appropriate vectors
    self.key = nn.Linear(nEmbed, headSize, bias=False) # Linear module (nEmbed, headSize) for the key vector (no bias = matrix multiply with some fixed weights)
    self.query = nn.Linear(nEmbed, headSize, bias=False) # Linear module (nEmbed, headSize) for the query vector (no bias = matrix multiply with some fixed weights)
    self.value = nn.Linear(nEmbed, headSize, bias=False) # Linear module (nEmbed, headSize) for the value vector (no bias = matrix multiply with some fixed weights)
    # register buffer (not a parameter) a T by T lower triangular 1s matrix (tril)
    self.register_buffer('tril', torch.tril(torch.ones(blockSize, blockSize)))
  
  # Forward pass of a singular head of self-attention (B, T, C) -> (B, T, headSize) 
  def forward(self, x):
    # B = batch size (compute in parallel)
    # T = time, block size, sequential characters in a context chunk
    # C = channel, nEmbed (also headSize in most cases)
    B, T, C = x.shape

    # Produce key and query vector in paraellel (no communication between tokens)
    k = self.key(x) # key vector for all tokens (B, T, C) -> (B, T, headSize) (headSize vector to store identity and position)
    q = self.query(x) # query vector for all tokens (B, T, C) -> (B, T, headSize) (headSize vector to store what identity and position to look for)

    B, T, headSize = k.shape

    # Dot product between query vector and key vector for all tokens is the new weight matrix (affinities between two tokens for all possible pairs of tokens)
    # entry (col, row): row-th query vector dot prodcut col-th key vector
    # scaled attention divides weightMatrix by sqrt(headSize) which will scale weightMatrix variance to query and key variance (control the variance)
    # which then the softmax will stay diffuse and not saturate too much to the extreme (converge to one hot vector)
    weightMatrix = q @ k.transpose(-2, -1) * headSize**-0.5 # transpose last two dimensions: (B, T, headSize) @ (B, headSize, T) -> (B, T, T) B T by T matrices

    # Filter/mask upper triangle of tril (lower triangular 1s matrix) which are all 0 with -inf (-inf represents that tokens from the future is not considered)
    # decoder block: use of triangular masking (tokens from the future is not considered), encoder block would delete this line
    weightMatrix = weightMatrix.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # lower triangular affinities and upper triangular -inf
    # softmax (normalization operation) each row (exponentiate (-inf -> 0, 0 -> 1, inf -> inf) all the entries/affinities and divide by the sum of its row of exponentiated entries)
    weightMatrix = F.softmax(weightMatrix, dim=-1) # each row sum to 1 (For batch b: i-th row of matrix is the weights for the i-th token in the sequence) (B, T, T)
    v = self.value(x) # value vector for all tokens (B, T, C) -> (B, T, headSize) (headSize vector for all tokens that store what private x stores (position and identity))
    out = weightMatrix @ v  # (B, T, T) @ (B, T, headSize) -> (B, T, headSize) (for a single head)

    return out
  
class MultiHeadAttention(nn.Module):
  """ 
  A class used to represent a multiple heads (communication channels) of self-attention in parallel (and concatenating the results)
    muliple independent communication channels to allow many different types of communication (attention) between tokens (ex. consonants, vowels)
    and decode them into the output
  """
  # nEmbed is the dimension of the inputs (previously calculated) into self-attention (embedding dimension)
  # blockSize T: number of time, sequential characters in a context chunk
  def __init__(self, numHeads, headSize, nEmbed, blockSize):
    super().__init__()
    # smaller head size for multi-head self attention
    multiHeadSize = headSize // numHeads
    # multiple smaller single head of self-attention in paraellel (numHeads * multiHeadSize = headSize for channel dimension consistency)
    self.heads = nn.ModuleList((AttentionHead(multiHeadSize, nEmbed, blockSize)) for _ in range(numHeads)) 
    # Linear Projection of the outcome back into the residual pathway
    self.proj = nn.Linear(headSize, headSize)
  
  # run multiple single head of self-attention in paraellel (B, T, C) -> (B, T, headSize)
  def forward(self, x):
    # run multiple single head of self-attention and concatenate the outputs over the channel dimension (C)
    out = torch.cat([head(x) for head in self.heads], dim=-1)
    # perform Linear Projection of the outcome back into the residual pathway
    out = self.proj(out)
    return out