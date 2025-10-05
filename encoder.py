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
    # object representing a look up table of vocabSize by nEmbed that stores the nEmbed logits for all possible prompt tokens
    self.tokenEmbeddingTable = nn.Embedding(vocabSize, nEmbed) # (vocabSize, C) encode token identity
    # object representing a look up table of blockSize by nEmbed that stores the nEmbed logits for all possible token positions
    self.positionEmbeddingTable = nn.Embedding(blockSize, nEmbed) # (S, C) encode token position
    # multiple iteration of self-attention (communication) and feed forward (computation) blocks to intersperse them
    self.blocks = nn.Sequential(*[EncoderBlock(headSize, numHeads, nEmbed, blockSize, dropout) for _ in range(numLayers)]) # numLayers * (B, S, nEmbed/headSize) 

  # forward function is implicitly called when the instance (object) is called directly (B, S) -> (B, S, vocabSize)
  # forward pass/evaluation of the model -> prompts is the input
  def forward(self, device, prompts):
    # B = batch size (compute in parallel)
    # S = prompt size, sequential characters in the prompt
    # C = channel, nEmbed (=headSize in this case)
    # vocabSize = all possible next tokens
    B, S = prompts.shape

    # prompts is (B, S)
    # returns a (B, S, headSize) tensor given the prompts by getting positional and identity embeddings returned by the embedding tables by going 
    # through all the prompts tokens in prompts and all the positions and adding the embeddings and converting to vocabSize logits for the tokens
    # get preceding token (prompts) embedding by inputting prompts into tokenEmbeddingTable
    tokenEmbedding = self.tokenEmbeddingTable(prompts) # (B, S) -> (B, S, C)
    # get positional embedding by inputting (0, 1, .., S - 1) tensor into the positionEmbeddingTable
    positionEmbedding = self.positionEmbeddingTable(torch.arange(S, device=device)) # (S) -> (S, C)
    # encode both positional and prececing token (prompts) embedding 
    x = tokenEmbedding + positionEmbedding # (B, S, C) + (B (B copies of positionEmbedding automatically added), S, C) = (B, S, C)
    # run the transformer blocks
    x = self.blocks(x) # (B, S, C) -> (B, S, headSize)
    
    return x