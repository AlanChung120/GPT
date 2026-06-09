import torch
from languageModel import LanguageModel
from transformer import Transformer
import pickle

# Pad data for dimension consistency in getBatch (0 means next line which is end of sequence)
def padData(lst, maxLength):
  if len(lst) < maxLength:
    return lst + [0] * (maxLength - len(lst))
  else:
    return lst

def getBatch(type):
  data = trainData if type == 'train' else valData # data based on type
  promptIndices = []
  indices = []
  # set list of random int between 0 to len(answer) - blocksize for batchSize random (prompt, answer) tuples (indices of answers to indicate start of a block)
  for _ in range(0, batchSize):
    promptIdx = torch.randint(len(data), ()) # index for the random (prompt, answer) tuple
    promptIndices.append(promptIdx)
    indices.append(torch.randint(len(data[promptIdx][1]) - blockSize, ())) # random starting index for the answer tensor
  prompts = torch.stack([data[promptIdx][0] for promptIdx in promptIndices])
  # these are all batchSize list of blockSize lists (batchSize amount of blockSize different contexts and targets for each of these contexts)
  x = torch.stack([data[promptIndices[idx]][1][indices[idx]:indices[idx]+blockSize] for idx in range(batchSize)]) # list of list of context in the chunk (current element and all preceding element is the context)
  y = torch.stack([data[promptIndices[idx]][1][indices[idx]+1:indices[idx]+blockSize+1] for idx in range(batchSize)]) # list of list of target in the chunk (current element is the target)
  prompts, x, y = prompts.to(device), x.to(device), y.to(device)
  return prompts, x, y

# estimate loss through average over multiple batches to reduce noise (batch dependent) (more accurate)
@torch.no_grad() # not call backward step (for pytorch efficiency)
def estimateLoss():
  lossEstimates = {}
  # disables dropout batchnorm layers
  model.eval() # switch to eval mode (bc some layers wil behave differently ex. dropout batchnorm layers)
  # for both splits
  for split in ['train', 'val']:
    losses = torch.zeros(estimateIters) # to store all losses from estimateIters iterations
    # average out the loss over multiple batches (estimateIters batches)
    for k in range(estimateIters):
      prompts, X, Y = getBatch(split)
      logits, loss = model(device, prompts, X, Y)
      losses[k] = loss.item() # store the loss
    lossEstimates[split] = losses.mean() # average out estimateIters iterations of losses
  # enables dropout batchnorm layers
  model.train() # switch to train mode (bc some layers wil behave differently ex. dropout batchnorm layers)
  return lossEstimates 

if __name__ == '__main__':
  torch.manual_seed(1337) # set seed for consistency

  # hyperparameters----------------------------------------------------------------------------------
  # beginning of sequence token and end of sequence token (for decoder/generator)
  BOS = '<'
  EOS = '>'
  # maximum input of encoder length/prompt length
  MAXPROMPTLENGTH = 4
  # train in chunks for efficiency
  batchSize = 4 # independent chunks to process in parallel (GPU efficient)
  # context-target based chunk training: the chunk contains information for every element in the chunk,
  # that element can act as a target and everything preceding can act as the context. So it contains each
  # element's positional information (context (as low as one characater) -> target)
  # Allow the model to see many different context sizes
  blockSize = 2 # maximum context length (chunk length) (that the model will be used to)
  epochs = 40000
  printInterval = 1000
  learningRate = 3e-5 # bigger the neural network the lower
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  estimateIters = 100 # number of iterations to calculate mean loss to estimate loss
  nEmbed = 4 # embedding dimensions (intermediate step)
  # should be equal to nEmbed to execute multiple iteration of blocks (output of self-attention back into input)
  attentionHeadSize = nEmbed # head size for one head of self-attention (attentionHeadSize % attentionNumHeads == 0)
  attentionNumHeads = 4 # number of self-attention heads to run in parallel
  numLayers = 2 # number of block layers
  dropout = 0.2 # dropout rate

  # read in the file
  with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

  lm = LanguageModel(text, BOS, EOS)

  # encode the text data into list of (prompt, answer) tuple. prompt and answer is stored as a torch tensor (multi-dimensional array in pytorch)
  data = [] # list of (prompt, answer) tuple for training/testing
  currentPrompt = [] # prompt of the current line
  for line in text.splitlines():  # go line by line (prompt and answer is separated by \n)
    if len(currentPrompt) == 0: # if prompt is not set then we set it
      currentPrompt = torch.tensor(padData(lm.encode(line), MAXPROMPTLENGTH), dtype=torch.long) # pad all prompts to MAXPROMPTLENGTH
    else: # if prompt is set then we add the (prompt, answer) tuple to data
      answer = torch.tensor(padData(lm.encode(line), blockSize + 1), dtype=torch.long) # pad for answers less than blockSize + 1 (+ 1 because last context needs a target)
      data.append((currentPrompt, answer))
      currentPrompt = []

  # split data into train data and validation sets (prevent and get a sense of overfitting) (len(data) >  1)
  split = int(0.9 * len(data)) # first part of the data will be train then rest of it will be validation
  trainData = data[:split]
  valData = data[split:]
  
  model = Transformer(nEmbed, lm.vocabSize, MAXPROMPTLENGTH, blockSize, attentionHeadSize, attentionNumHeads, numLayers, dropout).to(device)
  # optimizer: method of updating the parameters using the gradients, ADAM (adaptive learning rate)
  optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)

  for epoch in range(epochs):
    
    # every printInterval we print out the losses on train and validation sets
    if (epoch + 1) % printInterval == 0:
      losses = estimateLoss()
      print(f'epoch {epoch + 1}/{epochs}, train loss={losses['train'].item():.4f}, val loss={losses['val']:.4f}')

    # get batches of data
    prompts, xBatch, yBatch = getBatch('train')

    # forward pass: evaluate the logits (prediction scores) and the loss (want to minimize this)
    logits, loss = model(device, prompts, xBatch, yBatch)
    
    # backward step
    optimizer.zero_grad(set_to_none=True) # zero out gradients from previous epoch
    loss.backward() # calculate back propagation (gradients (derivative of loss function wrt parameter) for all the parameters)
    optimizer.step() # step (optimize parameters) towards negative of gradient (calculated in the previous step) scaled with learning rate

  # save language model
  LMFILE = "lm.pkl"
  with open(LMFILE, "wb") as file:
    pickle.dump(lm, file)
  print(f'Language Model saved to {LMFILE}')
  
  # training data to save
  trainingData = {
    "modelState": model.state_dict(),
    "maxPromptLength": MAXPROMPTLENGTH,
    "blockSize": blockSize,
    "nEmbed": nEmbed,
    "attentionHeadSize": attentionHeadSize,
    "attentionNumHeads": attentionNumHeads,
    "numLayers": numLayers,
    "dropout": dropout,
  }

  # save to a py torch file
  MODELFILE = "model.pth"
  torch.save(trainingData, MODELFILE)

  print(f'Training complete. File saved to {MODELFILE}')