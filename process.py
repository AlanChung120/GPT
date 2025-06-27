

if __name__ == '__main__':
  # read in the file (1,000,000 characters)
  with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()
  
  # get all the sorted unique (set) characters in the text 
  chars = sorted(list(set(text)))
  vocabSize = len(chars) # all possible characters/elements in the sequence of text
  print(''.join(chars)) # print all possible characters/elements

  # tokenize: convert text to sequence of numbers (according to the vocabulary chars)