import torch

if __name__ == '__main__':
  # read in the file (1,000,000 characters)
  with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()
  
  # get all the sorted unique (set) characters in the text 
  chars = sorted(list(set(text)))
  vocabSize = len(chars) # all possible characters/elements in the sequence of text
  print(''.join(chars)) # print all possible characters/elements

  # tokenize: convert text to sequence of numbers in a char level (according to the vocabulary chars)
  
  # create a mapping from characters to integers (alphabetical order)
  charToInt = { ch:i for i,ch in enumerate(chars) }
  intToChar = { i:ch for i,ch in  enumerate(chars) }

  # encoder: takes the string and turns it into its corresponding tokenized list of integers
  encode = lambda str: [charToInt[char] for char in str]
  # decoder: takes the list of integers and turns it into its corresponding string
  decode = lambda ints: ''.join([intToChar[int] for int in ints])

  # encode the entire text data and store it into a torch tensor (multi-dimensional array in pytorch)
  data = torch.tensor(encode(text), dtype=torch.long)

  # split data into train data and validation sets (prevent and get a sense of overfitting)
  split = int(0.9 * len(data)) # first part of the data will be train then rest of it will be validation
  trainData = data[:split]
  valData = data[split:]