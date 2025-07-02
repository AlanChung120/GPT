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