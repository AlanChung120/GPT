import torch
from torch.nn import functional as F
import torch.nn as nn
torch.manual_seed(1337) # set seed for consistency

class BigramModel(nn.Module):
  """
  A class used to represent a Bigram Language Model
  (predicts the probablilty of a sequence of tokens by considering the preceding token for each token: P(token | preceding token))
  """

  def __init__(self, vocabSize):
    super().__init__()
    # initialize the embedding table to initial values p = 1/C -> logit(p) = log(1/C / (1 - (1/C))) = log(1/(C - 1))
    # object representing a look up table of vocabSize by vocabSize that stores the logits for all possible next tokens given a context token
    # logits: scores/confidence level in the next token but instead of probablilties (0,1) it maps to real numbers (-inf, inf)
    # making regression easier (negative = 0.5 >, positive = 0.5 < , zero = 0.5)
    # (1) -> (1 * C)
    self.tokenEmbeddingTable = nn.Embedding(vocabSize, vocabSize) # (C, C)

  # forward function is implicitly called when the instance (object) is called directly (B, T) -> (B, T, vocabSize)
  # forward pass/evaluation of the model -> contexts is the input, targets is the target output
  def forward(self, contexts, targets=None):
    # B = batch size (compute in parallel)
    # T = time, block size, sequential characters in a context chunk
    # C = channel, all possible next tokens, vocabSize

    # contexts and targets are (B, T) -> for given context token contexts[i][j] the target token is targets[i][j]
    # returns a (B, T, C) tensor given the contexts storing all the logits returned by the embedding table by going 
    # through all the context tokens in contexts
    logits = self.tokenEmbeddingTable(contexts)

    if targets is None:
      loss = None # no loss if targets is unavailable
    else:
      # get logits dimensions
      B, T, C = logits.shape
      # change the logits shape to match the requirements of the cross entropy loss function
      logits = logits.view(B*T, C)
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
      lastLogits = logits[:, -1, :] # (B, 1, C)
      # convert the logits into probabilities using softmax (logits for each C -> probability distribution of length C)
      probs = F.softmax(lastLogits, dim=-1) # (B, 1, C)
      # sample from the probability distribution (of length C) so we have a sampled next token for each batch
      nextTokens = torch.multinomial(probs, num_samples=1) # (B, 1)
      seq = torch.cat((seq, nextTokens), dim=1) # append the next token to the running sequence (B, T+1)
    return seq # (B, T + maxNewTokens)

if __name__ == '__main__':
  # self-attention (communicate between all preceding tokens)------------------------------------------------------------------------------------------------------------
  B = 4
  T = 8
  C = 2
  x = torch.randn(B, T, C) # each token has some C-length vector storing information
  xbow = torch.zeros((B, T, C)) # x bag of words: averaging up word/token stored in T locations (counting frequency)
  for b in range(B): # go through all the batches
    for t in range(T): # go through all the timeline
      xprev = x[b, :t+1] # at this batch and current token and every token that precedes it with vocabSize information on each token (t, C)
      xbow[b, t] = torch.mean(xprev, 0) # average out t C-length vectors (component wise) and store the C-length vector under this batch and current timeline/token (t)

  # or weighted aggregation method
  # T by T lower triangular 1s matrix
  avgWeightMatrix = torch.tril(torch.ones(T, T))
  # scale it so that all the rows in avgWeightMatrix add up to one (lower triangular average matrix) (divide each element by its row sum)
  avgWeightMatrix = avgWeightMatrix / avgWeightMatrix.sum(1, keepdim=1)
  # matrix multiplication by avgWeightMatrix does the above self-attention calculation: for a batch for each column (c) t-th row is the vertical average up to t-th row
  xbow2 = avgWeightMatrix @ x # (B (B copies of avgWeightMatrix automatically added), T, T) @ (B, T, C) -> (B, T, C) (evaluate all batches in paraellel)
  # so a batch will have a T by C matrix where for each component/column c the row t is the vertical average of the component/column values up to the row t
  torch.allclose(xbow, xbow2) # true

  # or softmax method
  # T by T lower triangular 1s matrix
  tril = torch.tril(torch.ones(T, T))
  # T by T zero matrix (0s represents how many tokens from the past are we averaging up (aggregation))
  avgWeightMatrix = torch.zeros((T, T)) # 0s represent affinities (how much preceding token affects/interests the token) (will not be 0 but will be data/token dependent)
  # Filter/mask upper triangle of tril (lower triangular 1s matrix) which are all 0 with -inf (-inf represents that tokens from the future is not considered)
  avgWeightMatrix = avgWeightMatrix.masked_fill(tril == 0, float('-inf')) # lower triangular 0s and upper triangular -inf
  # softmax (normalization operation) each row (exponentiate (-inf -> 0, 0 -> 1, inf -> inf) all the entries and divide by the sum of its row of exponentiated entries)
  avgWeightMatrix = F.softmax(avgWeightMatrix, dim=-1)
  # avgWeightMatrix is the same matrix as above
  xbow3 = avgWeightMatrix @ x  # (B (B copies of avgWeightMatrix automatically added), T, T) @ (B, T, C) -> (B, T, C)
  torch.allclose(xbow, xbow3) # true

  # self-attention summary (weighted aggregation): matrix multiply by a a lower triangular with lower triangular values representing affinities-------------------------