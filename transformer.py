import torch
import torch.nn as nn
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
    # encoder model
    self.encoder = Encoder(nEmbed, vocabSize, blockSize, headSize, numHeads, numLayers, dropout)
    # decoder model
    self.decoder = Decoder(nEmbed, vocabSize, blockSize, headSize, numHeads, numLayers, dropout)

  # forward function is implicitly called when the instance (object) is called directly (B, T) -> (B, T, vocabSize)
  # forward pass/evaluation of the model ->  prompt (B, T) is the encoderInput, contexts (B, T) and targets (B, T) is the decoderInput
  def forward(self, device, prompt, contexts, targets):
    external = self.encoder(device, prompt)
    logits, loss = self.decoder(device, contexts, targets, external)

    return logits, loss
  
  # generate maxNewTokens tokens given context tokens contexts (B, T)
  def generate(self, contexts, maxNewTokens, blockSize, device):
    self.decoder.generate(contexts, maxNewTokens, blockSize, device)