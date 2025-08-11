import torch
import torch.nn as nn
from torch.nn import functional as F
from block import Block
torch.manual_seed(1337) # set seed for consistency

class BigramModel(nn.Module):
  """
  A class used to represent a Bigram Language Model
  (predicts the probablilty of a sequence of tokens by considering the preceding token for each token: P(token | preceding token))
  """
  # required: nEmbed = headSize (input into output for blocks)
  def __init__(self, nEmbed, vocabSize, blockSize, headSize, numHeads):
    super().__init__()
    # initialize the embedding table to initial values p = 1/C -> logit(p) = log(1/C / (1 - (1/C))) = log(1/(C - 1))
    # logits: scores/confidence level in the next token but instead of probablilties (0,1) it maps to real numbers (-inf, inf)
    # making regression easier (negative = 0.5 >, positive = 0.5 < , zero = 0.5)
    # (1) -> (1 * C)
    # object representing a look up table of vocabSize by nEmbed that stores the nEmbed logits for all possible context tokens (preceding)
    self.tokenEmbeddingTable = nn.Embedding(vocabSize, nEmbed) # (vocabSize, C) encode token identity
    # object representing a look up table of blockSize by nEmbed that stores the nEmbed logits for all possible token positions
    self.positionEmbeddingTable = nn.Embedding(blockSize, nEmbed) # (T, C) encode token position
    # multiple iteration of self-attention (communication) and feed forward (computation) blocks to intersperse them
    self.blocks = nn.Sequential(
      Block(headSize, numHeads, nEmbed, blockSize), 
      Block(headSize, numHeads, nEmbed, blockSize), 
      Block(headSize, numHeads, nEmbed, blockSize), 
    ) # 3 * (B, T, nEmbed/headSize)
    # language modelling head (headSize, vocabSize) linear module (decoder) to turn headSize dimension back to vocabSize dimension (vocabSize possible next tokens)
    self.lmHead = nn.Linear(headSize, vocabSize) # (headSize, vocabSize)

  # forward function is implicitly called when the instance (object) is called directly (B, T) -> (B, T, vocabSize)
  # forward pass/evaluation of the model -> contexts is the input, targets is the target output
  def forward(self, contexts, targets=None, device=None):
    # B = batch size (compute in parallel)
    # T = time, block size, sequential characters in a context chunk
    # C = channel, nEmbed (=headSize in this case)
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
    # run the transformer blocks
    x = self.blocks(x) # (B, T, C) -> (B, T, headSize)
    # convert headSize (which is C and nEmbed in most cases) dimension back to vocabSize dimension to get the logits for all possible next tokens
    logits = self.lmHead(x) # (B, T, headSize) -> (B, T, vocabSize)

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
  def generate(self, contexts, maxNewTokens, blockSize):
    seq = contexts # initialize the sequence of tokens with the current context
    # generate batchSize next tokens in parallel for maxNewTokens tokens
    for _ in range(maxNewTokens):
      # crop seq to get last blockSize tokens for the positionEmbeddingTable (otherwise it will run out of scope; it only has embeddings for blockSize)
      seqBlock = seq[:, -blockSize:] # (B, blockSize, vocabSize)
      # get the predictions in the form of logits
      logits, loss = self(seqBlock) # call the forward function
      # get the most recent (last time step) token for all batches (WILL FIX: not ideal to only look at last token)
      lastLogits = logits[:, -1, :] # (B, 1, vocabSize)
      # convert the logits into probabilities using softmax (logits for each vocabSize -> probability distribution of length vocabSize)
      probs = F.softmax(lastLogits, dim=-1) # (B, 1, vocabSize)
      # sample from the probability distribution (of length vocabSize) so we have a sampled next token for each batch
      nextTokens = torch.multinomial(probs, num_samples=1) # (B, 1)
      seq = torch.cat((seq, nextTokens), dim=1) # append the next token to the running sequence (B, T+1)
    return seq # (B, T + maxNewTokens)