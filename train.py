import torch
from languageModel import LanguageModel

if __name__ == '__main__':
  # read in the file (1,000,000 characters)
  with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

  lm = LanguageModel(text)
  lm.showChars()

  # encode the entire text data and store it into a torch tensor (multi-dimensional array in pytorch)
  data = torch.tensor(lm.encode(text), dtype=torch.long)

  # split data into train data and validation sets (prevent and get a sense of overfitting)
  split = int(0.9 * len(data)) # first part of the data will be train then rest of it will be validation
  trainData = data[:split]
  valData = data[split:]

  # train in chunks for efficiency
  batchSize = 4 # independent chunks to process in parallel (GPU efficient)
  # context-target based chunk training: the chunk contains information for every element in the chunk
  # that element can act as a target and everything preceding can act as the context. So it contains each
  # elemtent's positional information (context (as low as one characater) -> target)
  # Allow the model to see many different context sizes
  blockSize = 8 # maximum context length (chunk length) (that the model will be used to)

  torch.manual_seed(1337) # set seed for consistency