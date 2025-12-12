import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337) # set seed for consistency

class AttentionHead(nn.Module):
  """
  A class used to represent a one head of attention (self when keys and values all come from the same source as queries (x) otherwise it is cross-attention)
  Attention is communication mechanism of a directed graph that aggregates via the weighted sum from all nodes that point to them (all the preceding tokens in our case)
  no notion of space, just set of vectors and we postionally encode them (positionEmbeddingTable)
  - self-attention (keys and values are produced from the same source as queries) 
  - cross-attention (queries are produced from x but keys and values come from an external source (encoder module))
  - encoder block: no masking (all tokens communciate with each other (even future), need for sentiment of sentence, communication at all levels)
  - decoder block: triangular masking (triangular masking to prevent communication from the future, used in language modelling/autoregressive to "decode")
  instead of equal weighted aggregation we implement a batch/data/token dependent affinity (how related, how much to aggregate/learn) matrix aggregation
  each token has a query vector (what I am looking for based on token identity and position) and a key vector (what I contain (token identity and position))
  affinity of token x with token y (y precedes x) = dot product between query vector of x with key vector of y
  if key and query vector match (key is what query is looking for) higher the dot product (same high components), higher the affinity, higher the weight (relative))
  """

  def __init__(self, headSize, nEmbed, dropout, blockSize=None):
    super().__init__()
    # headSize is the size of the key/query vectors
    # Arbitrary module to learn the weights to convert to appropriate vectors
    self.key = nn.Linear(nEmbed, headSize, bias=False) # Linear module (nEmbed, headSize) for the key vector (no bias = matrix multiply with some fixed weights)
    self.query = nn.Linear(nEmbed, headSize, bias=False) # Linear module (nEmbed, headSize) for the query vector (no bias = matrix multiply with some fixed weights)
    self.value = nn.Linear(nEmbed, headSize, bias=False) # Linear module (nEmbed, headSize) for the value vector (no bias = matrix multiply with some fixed weights)
    # register buffer (not a learnable parameter) a blockSize by blockSize lower triangular 1s matrix where 1s represents the entries that are allowed to communicate 
    # only used when blockSize is not None (decoder block self-attention) because blockSize is only used to create tril
    self.register_buffer('tril', torch.tril(torch.ones(blockSize, blockSize)) if blockSize is not None else None)
    # dropout is a regularization technique to prevent overfitting
    # dropout randomly shuts off some subset of neurons every pass, thus training ensemble of subnetworks which is then merged
    self.dropout = nn.Dropout(dropout)
  
  # Forward pass of a singular head of attention (B, T, C) -> (B, T, headSize) 
  # paddedMask: True/False (is padded value) matrix for filtering out padded value for attention (B, T/S)
  # if external provided then it is cross-attention from external otherwise it is self-attention
  def forward(self, x, paddedMask, external=None):
    # B = batch size (compute in parallel)
    # T = time, block size, sequential characters in a context chunk (=S if encoder block attention)
    # C = channel, nEmbed (also headSize in most cases)
    B, T, C = x.shape

    # if encoder output provided
    if external is not None:
      # B = batch size (compute in parallel)
      # S = encoder input size, prompt size
      # C = headSize
      B, S, C = external.shape

    # Produce key, query, and value vector in paraellel (no communication between tokens)
    # gets keys and values from external source if provided (cross) if not get it from itself (self)
    k = self.key(external) if external is not None else self.key(x)  # key vector for all tokens (B, T/S, C) -> (B, T/S, headSize) (headSize vector to store identity and position)
    q = self.query(x) # query vector for all tokens (B, T, C) -> (B, T, headSize) (headSize vector to store what identity and position to look for)
    v = self.value(external) if external is not None else self.value(x) # value vector for all tokens (B, T/S, C) -> (B, T/S, headSize) (headSize vector for all tokens that store what private x stores (position and identity))

    B, T, headSize = k.shape

    # Dot product between query vector and key vector for all tokens is the new weight matrix (affinities between two tokens for all possible pairs of tokens)
    # entry (col, row): row-th query vector dot prodcut col-th key vector
    # scaled attention divides weightMatrix by sqrt(headSize) which will scale weightMatrix variance to query and key variance (control the variance)
    # which then the softmax will stay diffuse and not saturate too much to the extreme (converge to one hot vector)
    weightMatrix = q @ k.transpose(-2, -1) * headSize**-0.5 # transpose last two dimensions: (B, T, headSize) @ (B, headSize, T/S) -> (B, T, T/S) B T by T/S matrices
    # If tril is set (decoder block self-attention) filter/mask appropriately using tril matrix set above (encoder/decoder step)
    # tril is shrunk to T dimension (for T < blockSize case but T = blockSize) but still keeps the lower triangle 1s matrix form
    # filter/mask upper triangle of tril (lower triangular 1s matrix) which are all 0 with -inf (-inf represents that tokens from the future is not considered)
    if self.tril is not None:
      weightMatrix = weightMatrix.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # lower triangular affinities and upper triangular -inf
    # Extra masking for the padded values (for every query (2nd dimension), key (last dimension, what paddedMask is based off) is appropriately masked out)
    weightMatrix = weightMatrix.masked_fill(paddedMask.unsqueeze(1), float('-inf')) # (B, T, T/S) (unsqueeze -> (B, T/S) to (B, 1, T/S))
    # key (last dimension) softmax (row normalization operation) (exponentiate (-inf -> 0, 0 -> 1, inf -> inf) all the entries/affinities and divide by the sum of its row of exponentiated entries)
    weightMatrix = F.softmax(weightMatrix, dim=-1) # each row sum to 1 (For batch b: i-th row of matrix is the weights for the i-th token in the sequence) (B, T, T/S)
    weightMatrix = torch.nan_to_num(weightMatrix, nan=0.0) # convert NaN to 0
    # perform dropout (randomly prevent some nodes from communicating)
    weightMatrix = self.dropout(weightMatrix)
    out = weightMatrix @ v  # (B, T, T/S) @ (B, T/S, headSize) -> (B, T, headSize) (for a single head)

    return out
  
class MultiHeadAttention(nn.Module):
  """ 
  A class used to represent a multiple heads (communication channels) of attention in parallel (and concatenating the results)
    muliple independent communication channels to allow many different types of communication (attention) between tokens (ex. consonants, vowels)
    and decode them into the output
  """
  # nEmbed is the dimension of the inputs (previously calculated) into attention (embedding dimension)
  # blockSize T: number of time, sequential characters in a context chunk
  def __init__(self, numHeads, headSize, nEmbed, dropout, blockSize=None):
    super().__init__()
    # smaller head size for multi-head attention
    multiHeadSize = headSize // numHeads
    # multiple smaller single head of attention in paraellel (numHeads * multiHeadSize = headSize for channel dimension consistency)
    self.heads = nn.ModuleList((AttentionHead(multiHeadSize, nEmbed, dropout, blockSize)) for _ in range(numHeads)) 
    # Linear Projection of the outcome back into the residual pathway
    self.proj = nn.Linear(headSize, headSize)
    # dropout is a regularization technique to prevent overfitting, dropout right before residual connection into residual pathway 
    # dropout randomly shuts off some subset of neurons every pass, thus training ensemble of subnetworks which is then merged
    self.dropout = nn.Dropout(dropout)
  
  # run multiple single head of attention in paraellel (B, T/S, C) -> (B, T/S, headSize)
  def forward(self, x, paddedMask, external=None):
    # run multiple single head of attention and concatenate the outputs over the channel dimension (C)
    out = torch.cat([head(x, paddedMask, external) for head in self.heads], dim=-1)
    # perform Linear Projection of the outcome back into the residual pathway and dropout
    out = self.dropout(self.proj(out))
    return out