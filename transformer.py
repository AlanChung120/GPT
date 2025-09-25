import torch
import torch.nn as nn
from torch.nn import functional as F
from decoder import Decoder
from encoder import Encoder
torch.manual_seed(1337) # set seed for consistency

class Transformer(nn.Module):
  """
  A class used to represent a transformer model (encoder + decoder model, see tranformer_model.png)
  """
  # required: nEmbed = headSize (input into output for blocks)
  def __init__(self, nEmbed, vocabSize, blockSize, headSize, numHeads, numLayers, dropout):
    super().__init__()
    # encoder model (see encoder.py)
    self.encoder = Encoder(nEmbed, vocabSize, blockSize, headSize, numHeads, numLayers, dropout)
    # decoder model (see decoder.py)
    self.decoder = Decoder(nEmbed, vocabSize, blockSize, headSize, numHeads, numLayers, dropout)

  # forward function is implicitly called when the instance (object) is called directly (B, T) -> (B, T, vocabSize)
  # forward pass/evaluation of the model ->  prompts (B, T) is the encoder input, contexts (B, T) and targets (B, T) is the decoder input
  def forward(self, device, prompts, contexts, targets=None):
    # run the encoder to get the encoded prompt to input to the decoder (see encoder.py)
    encodedPrompts = self.encoder(device, prompts) # (B, T) -> (B, T, vocabSize)
    # run the decoder (see decoder.py)
    logits, loss = self.decoder(device, contexts, targets, encodedPrompts) # (B, T) and (B, T, vocabSize) -> (B, T, vocabSize)

    return logits, loss
  
  # generate maxNewTokens tokens given prompt tokens prompts (B, T) and context tokens contexts (B, T) 
  def generate(self, prompts, contexts, maxNewTokens, blockSize, device):
    seq = contexts # initialize the sequence of tokens with the current context
    # generate batchSize next tokens in parallel for maxNewTokens tokens
    for _ in range(maxNewTokens):
      # crop seq to get last blockSize tokens for the positionEmbeddingTable (otherwise it will run out of scope; it only has embeddings for blockSize)
      seqBlock = seq[:, -blockSize:] # (B, blockSize, vocabSize)
      # get the predictions in the form of logits
      logits, loss = self(device, prompts, seqBlock) # call the forward function
      # get the most recent (last time step) token for all batches
      lastLogits = logits[:, -1, :] # (B, 1, vocabSize)
      # convert the logits into probabilities using softmax (logits for each vocabSize -> probability distribution of length vocabSize)
      probs = F.softmax(lastLogits, dim=-1) # (B, 1, vocabSize)
      # sample from the probability distribution (of length vocabSize) so we have a sampled next token for each batch
      nextTokens = torch.multinomial(probs, num_samples=1) # (B, 1)
      seq = torch.cat((seq, nextTokens), dim=1) # append the next token to the running sequence (B, T+1)
    return seq # (B, T + maxNewTokens)