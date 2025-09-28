import torch
import torch.nn as nn
from torch.nn import functional as F
from encoderBlock import EncoderBlock
torch.manual_seed(1337) # set seed for consistency

class Encoder(nn.Module):
  """
  A class used to represent a encoder Model (just run the encoder to test) like a block to contextualize input 
  for cross-head attention for decoder
  """
  # required: nEmbed = headSize (input into output for blocks)
  def __init__(self, nEmbed, vocabSize, blockSize, headSize, numHeads, numLayers, dropout):
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
    self.blocks = nn.Sequential(*[EncoderBlock(headSize, numHeads, nEmbed, blockSize, dropout) for _ in range(numLayers)]) # numLayers * (B, T, nEmbed/headSize) 

  # forward function is implicitly called when the instance (object) is called directly (B, T) -> (B, T, vocabSize)
  # forward pass/evaluation of the model -> contexts is the input
  def forward(self, device, contexts):
    # B = batch size (compute in parallel)
    # T = time, prompt size, sequential characters in the prompt
    # C = channel, nEmbed (=headSize in this case)
    # vocabSize = all possible next tokens
    B, T = contexts.shape

    # contexts is (B, T)
    # returns a (B, T, headSize) tensor given the contexts by getting positional and identity embeddings returned by the embedding tables by going 
    # through all the context tokens in contexts and all the positions and adding the embeddings and converting to vocabSize logits for next possible tokens
    # get preceding token (context) embedding by inputting contexts into tokenEmbeddingTable
    tokenEmbedding = self.tokenEmbeddingTable(contexts) # (B, T) -> (B, T, C)
    # get positional embedding by inputting (0, 1, .., T - 1) tensor into the positionEmbeddingTable
    positionEmbedding = self.positionEmbeddingTable(torch.arange(T, device=device)) # (T) -> (T, C)
    # encode both positional and prececing token (context) embedding 
    x = tokenEmbedding + positionEmbedding # (B, T, C) + (B (B copies of positionEmbedding automatically added), T, C) = (B, T, C)
    # run the transformer blocks
    x = self.blocks(x) # (B, T, C) -> (B, T, headSize)
    
    return x