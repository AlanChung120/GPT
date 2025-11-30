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
  def __init__(self, nEmbed, vocabSize, maxEncoderSize, blockSize, headSize, numHeads, numLayers, dropout):
    super().__init__()
    # encoder model (see encoder.py)
    self.encoder = Encoder(nEmbed, vocabSize, maxEncoderSize, headSize, numHeads, numLayers, dropout)
    # decoder model (see decoder.py)
    self.decoder = Decoder(nEmbed, vocabSize, blockSize, headSize, numHeads, numLayers, dropout)

  # forward function is implicitly called when the instance (object) is called directly (B, S) and (B, T) -> (B, T, vocabSize)
  # forward pass/evaluation of the model -> prompts (B, S) is the encoder input, contexts (B, T) and targets (B, T) is the decoder input
  def forward(self, device, prompts, contexts, targets=None):
    encPaddedMask = (prompts == 0) # True/False (is padded value) matrix for filtering out padded value for attention (B, S)
    decPaddedMask = (contexts == 0) # True/False (is padded value) matrix for filtering out padded value for attention (B, T)
    # run the encoder to get the encoded prompt to input to the decoder (see encoder.py)
    encodedPrompts = self.encoder(device, prompts, encPaddedMask) # (B, S) -> (B, S, headSize)
    # run the decoder (see decoder.py)
    logits, loss = self.decoder(device, contexts, encPaddedMask, decPaddedMask, targets, encodedPrompts) # (B, T) and (B, S, headSize) -> (B, T, vocabSize)

    return logits, loss
  
  # generate maxNewTokens tokens given prompt tokens prompts (B, S) and context tokens contexts (B, T) 
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
  

  # generate until end of sequence token (eosToken) given prompt tokens prompts (1, S) and context tokens contexts (1, T) (no batchSize parallel work)
  def generateUntil(self, prompts, contexts, eosToken, blockSize, device):
    seq = contexts # initialize the sequence of tokens with the current context
    while True:
      # crop seq to get last blockSize tokens for the positionEmbeddingTable (otherwise it will run out of scope; it only has embeddings for blockSize)
      seqBlock = seq[:,-blockSize:] # (1, blockSize, vocabSize)
      # get the predictions in the form of logits
      logits, loss = self(device, prompts, seqBlock) # call the forward function
      # get the most recent (last time step) token
      lastLogits = logits[:, -1, :] # (1, 1, vocabSize)
      # convert the logits into probabilities using softmax (logits for each vocabSize -> probability distribution of length vocabSize)
      probs = F.softmax(lastLogits, dim=-1) # (1, 1, vocabSize)
      # sample from the probability distribution (of length vocabSize) so we have a sampled next token for each batch
      nextTokens = torch.multinomial(probs, num_samples=1) # (1, 1)
      seq = torch.cat((seq, nextTokens), dim=1) # append the next token to the running sequence (1, T+1)
      if (nextTokens[0, 0] == eosToken):
        break
    return seq # (1, T + new tokens)