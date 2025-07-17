import torch
from languageModel import LanguageModel
from bigram import BigramModel

def getBatch(type):
  data = trainData if type == 'train' else valData # data based on type
  # list of batchSize random int between 0 to len(data) - blocksize (indices of data to indicate start of a block)
  indices = torch.randint(len(data) - blockSize, (batchSize, ))
  # these are all batchSize list of blockSize lists
  x = torch.stack([data[idx: idx+blockSize] for idx in indices]) # list of list of context in the chunk (current element and all preceding element is the context)
  y = torch.stack([data[idx+1:idx+blockSize+1] for idx in indices]) # list of list of target in the chunk (current element is the target)
  x, y = x.to(device), y.to(device)
  return x, y

# estimate loss through average over multiple batches to reduce noise (batch dependent) (more accurate)
@torch.no_grad() # not call backward step (for pytorch efficiency)
def estimateLoss():
  lossEstimates = {}
  # eval = train (no dropout batchnorm layers) in the bigram model so it does nothing
  model.eval() # switch to eval mode (bc some layers wil behave differently ex. dropout batchnorm layers)
  # for both splits
  for split in ['train', 'val']:
    losses = torch.zeros(estimateIters) # to store all losses from estimateIters iterations
    # average out the loss over multiple batches (estimateIters batches)
    for k in range(estimateIters):
      X, Y = getBatch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item() # store the loss
    lossEstimates[split] = losses.mean() # average out estimateIters iterations of losses
  # eval = train (no dropout batchnorm layers) in the bigram model so it does nothing
  model.train() # switch to train mode (bc some layers wil behave differently ex. dropout batchnorm layers)
  return lossEstimates 

if __name__ == '__main__':
  torch.manual_seed(1337) # set seed for consistency

  # hyperparameters
  # train in chunks for efficiency
  batchSize = 32 # independent chunks to process in parallel (GPU efficient)
  # context-target based chunk training: the chunk contains information for every element in the chunk,
  # that element can act as a target and everything preceding can act as the context. So it contains each
  # element's positional information (context (as low as one characater) -> target)
  # Allow the model to see many different context sizes
  blockSize = 8 # maximum context length (chunk length) (that the model will be used to)
  epochs = 3000
  printInterval = 300
  learningRate = 1e-2
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  estimateIters = 200 # number of iterations to calculate mean loss to estimate loss
  maxNewTokens = 500

  # read in the file (1,000,000 characters)
  with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

  lm = LanguageModel(text)

  # encode the entire text data and store it into a torch tensor (multi-dimensional array in pytorch)
  data = torch.tensor(lm.encode(text), dtype=torch.long)

  # split data into train data and validation sets (prevent and get a sense of overfitting)
  split = int(0.9 * len(data)) # first part of the data will be train then rest of it will be validation
  trainData = data[:split]
  valData = data[split:]
  
  model = BigramModel(lm.vocabSize).to(device)
  # optimizer: method of updating the parameters using the gradients, ADAM (adaptive learning rate)
  optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)

  for epoch in range(epochs):
    
    # every printInterval we print out the losses on train and validation sets
    if (epoch + 1) % printInterval == 0:
      losses = estimateLoss()
      print(f'epoch {epoch + 1}/{epochs}, train loss={losses['train'].item():.4f}, val loss={losses['val']:.4f}')

    # get batches of data
    xBatch, yBatch = getBatch('train')

    # forward pass: evaluate the logits (prediction scores) and the loss (want to minimize this)
    logits, loss = model(xBatch, yBatch)
    
    # backward step
    optimizer.zero_grad(set_to_none=True) # zero out gradients from previous epoch
    loss.backward() # calculate back propagation (gradients (derivative of loss function wrt parameter) for all the parameters)
    optimizer.step() # step (optimize parameters) towards negative of gradient (calculated in the previous step) scaled with learning rate

  # generate from the model
  context = torch.zeros((1, 1), dtype=torch.long, device=device) # feed the new line character "\n" (0) as the starting sequence/context
  print(lm.decode(model.generate(context, maxNewTokens)[0].tolist())) # generate from the initial context get the first batch and decode it