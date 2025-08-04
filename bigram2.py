import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337) # set seed for consistency

class BigramModel(nn.Module):
  """
  A class used to represent a Bigram Language Model
  (predicts the probablilty of a sequence of tokens by considering the preceding token for each token: P(token | preceding token))
  """

  def __init__(self, nEmbed, vocabSize, blockSize):
    super().__init__()
    # initialize the embedding table to initial values p = 1/C -> logit(p) = log(1/C / (1 - (1/C))) = log(1/(C - 1))
    # logits: scores/confidence level in the next token but instead of probablilties (0,1) it maps to real numbers (-inf, inf)
    # making regression easier (negative = 0.5 >, positive = 0.5 < , zero = 0.5)
    # (1) -> (1 * C)
    # object representing a look up table of vocabSize by nEmbed that stores the nEmbed logits for all possible context tokens (preceding)
    self.tokenEmbeddingTable = nn.Embedding(vocabSize, nEmbed) # (vocabSize, C) encode token identity
    # object representing a look up table of blockSize by nEmbed that stores the nEmbed logits for all possible token positions
    self.positionEmbeddingTable = nn.Embedding(blockSize, nEmbed) # (T, C) encode token position
    # language modelling head (C, vocabSize) linear module to turn nEmbed dimension back to vocabSize dimension (vocabSize possible next tokens)
    self.lmHead = nn.Linear(nEmbed, vocabSize) # (C, vocabSize)

  # forward function is implicitly called when the instance (object) is called directly
  # forward pass/evaluation of the model -> contexts is the input, targets is the target output
  def forward(self, contexts, targets=None, device=None):
    # B = batch size (compute in parallel)
    # T = time, block size, sequential characters in a context chunk
    # C = channel, nEmbed
    # vocabSize = all possible next tokens
    B, T = contexts.shape

    # contexts and targets are (B, T) -> for given context token contexts[i][j] the target token is targets[i][j]
    # returns a (B, T, vocabSize) tensor given the contexts by getting positional and identity embeddings returned by the embedding tables by going 
    # through all the context tokens in contexts and all the positions and adding the embeddings and converting to vocabSize logits for next possible tokens
    # get preceding token (context) embedding by inputting contexts into tokenEmbeddingTable
    tokenEmbedding = self.tokenEmbeddingTable(contexts) # (B, T) -> (B, T, C)
    # get positional embedding by inputting (0, 1, .., T - 1) tensor into the positionEmbeddingTable
    positionEmbedding = self.positionEmbeddingTable(torch.arange(T, device=device)) # (T) -> (T, C)
    # encode both positional and prececing token (context) embedding 
    x = tokenEmbedding + positionEmbedding # (B, T, C) + (B (B copies of positionEmbedding automatically added), T, C) = (B, T, C)
    # convert C=nEmbed dimension back to vocabSize dimension to get the logits for all possible next tokens
    logits = self.lmHead(x) # (B, T, C) -> (B, T, vocabSize)

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    # single head self-attention (communication between tokens) (self because keys and values all come from the same source as queries (x) otherwise it is cross-attention)
    # instead of equal weighted aggregation we implement a batch/data/token dependent affinity (how related, how much to aggregate/learn) matrix aggregation
    # each token has a query vector (what I am looking for based on token identity and position) and a key vector (what I contain (token identity and position))
    # affinity of token x with token y (y precedes x) = dot product between query vector of x with key vector of y
    # if key and query vector match (key is what query is looking for) higher the dot product (same high components), higher the affinity, higher the weight (relative))
    C = 32
    headSize = 16 # size of the key/query vectors
    # Arbitrary module to learn the weights to convert to appropriate vectors
    key = nn.Linear(C, headSize, bias=False) # Linear module (C, headSize) for the key vector (no bias = matrix multiply with some fixed weights)
    query = nn.Linear(C, headSize, bias=False) # Linear module (C, headSize) for the query vector (no bias = matrix multiply with some fixed weights)
    value = nn.Linear(C, headSize, bias=False) # Linear module (C, headSize) for the value vector (no bias = matrix multiply with some fixed weights)
    # Produce key and query vector in paraellel (no communication between tokens)
    k = key(x) # key vector for all tokens (B, T, C) -> (B, T, headSize) (headSize vector to store identity and position)
    q = query(x) # query vector for all tokens (B, T, C) -> (B, T, headSize) (headSize vector to store what identity and position to look for)
    # Dot product between query vector and key vector for all tokens is the new weight matrix (affinities between two tokens for all possible pairs of tokens)
    # entry (col, row): row-th query vector dot prodcut col-th key vector
    weightMatrix = q @ k.transpose(-2, -1) # transpose last two dimensions: (B, T, headSize) @ (B, headSize, T) -> (B, T, T) B T by T matrices
    # T by T lower triangular 1s matrix
    tril = torch.tril(torch.ones(T, T))
    # Filter/mask upper triangle of tril (lower triangular 1s matrix) which are all 0 with -inf (-inf represents that tokens from the future is not considered)
    # decoder block: use of triangular masking (tokens from the future is not considered), encoder block would delete this line
    weightMatrix = weightMatrix.masked_fill(tril == 0, float('-inf')) # lower triangular affinities and upper triangular -inf
    # softmax (normalization operation) each row (exponentiate (-inf -> 0, 0 -> 1, inf -> inf) all the entries/affinities and divide by the sum of its row of exponentiated entries)
    weightMatrix = F.softmax(weightMatrix, dim=-1) # each row sum to 1 (For batch b: i-th row of matrix is the weights for the i-th token in the sequence)
    v = value(x) # value vector for all tokens (B, T, C) -> (B, T, headSize) (headSize vector for all tokens that store what private x stores (position and identity))
    out = weightMatrix @ v  # (B, T, T) @ (B, T, headSize) -> (B, T, headSize) (for a single head)


    #------------------------------------------------------------------------------------------------------------------------------------------------------------

    if targets is None:
      loss = None # no loss if targets is unavailable
    else:
      # get logits dimensions
      B, T, vocabSize = logits.shape
      # change the logits shape to match the requirements of the cross entropy loss function
      logits = logits.view(B*T, vocabSize)
      # change the targets shape to match the requirements of the cross entropy loss function
      targets = targets.view(B*T)

      # functioncal form of cross entropy loss
      # negative log likelihood loss to calculate loss/error/difference between logits and the correct targets
      # the correct target token should have a very high number under logits
      loss = F.cross_entropy(logits, targets)

    return logits, loss
  
  # generate maxNewTokens tokens given context tokens contexts (B, T)
  def generate(self, contexts, maxNewTokens):
    seq = contexts # initialize the sequence of tokens with the current context
    # generate batchSize next tokens in parallel for maxNewTokens tokens
    for _ in range(maxNewTokens):
      # get the predictions in the form of logits
      logits, loss = self(seq) # call the forward function
      # get the most recent (last time step) token for all batches (WILL FIX: not ideal to only look at last token)
      lastLogits = logits[:, -1, :] # (B, 1, vocabSize)
      # convert the logits into probabilities using softmax (logits for each vocabSize -> probability distribution of length vocabSize)
      probs = F.softmax(lastLogits, dim=-1) # (B, 1, vocabSize)
      # sample from the probability distribution (of length vocabSize) so we have a sampled next token for each batch
      nextTokens = torch.multinomial(probs, num_samples=1) # (B, 1)
      seq = torch.cat((seq, nextTokens), dim=1) # append the next token to the running sequence (B, T+1)
    return seq # (B, T + maxNewTokens)