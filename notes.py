import torch
from torch.nn import functional as F

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