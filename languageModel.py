class LanguageModel:
  """
  A class used to represent a language model and to tokenize
  
  Fields
  -----------
  bos : str
    the character representing the beginning of sequence token
  eos : str
    the character representing the end of sequence token
  chars : list[str]
    list of all the characters in the language model
  vocabSize : int
    number of characters in the language model
  charToInt: dict[str, int]
    translation dictionary from character to integer in the language model
  intToChar: dict[int, str]
    translation dictionary from integer to character in the language model
  """
  bos = ''
  eos = ''
  chars = []
  vocabSize = 0
  charToInt = {}
  intToChar = {}

  def __init__(self, text, bos, eos):
    self.bos = bos
    self.eos = eos

    self.chars = sorted(list(set(text))) # get all the sorted unique (set) characters in the text 
    self.vocabSize = len(self.chars) # count of all possible characters/elements in the sequence of text

    # create a mapping from characters to integers and vice versa (alphabetical order)
    self.charToInt = { ch:i for i,ch in enumerate(self.chars) }
    self.intToChar = { i:ch for i,ch in enumerate(self.chars) }
  
  # prints chars
  def showChars(self):
    print(''.join(self.chars)) # print all possible characters/elements

  # tokenize: convert text to sequence of numbers in a char level (according to the vocabulary chars)
  # encoder: takes the string and turns it into its corresponding tokenized list of integers
  def encode(self, str):
    return [self.charToInt[char] for char in str]
  
  # decoder: takes the list of integers and turns it into its corresponding string
  def decode(self, ints):
    return ''.join([self.intToChar[int] for int in ints])

  # Get the index of the beginning of sequence token (integer)
  def getBosIndex(self):
    return self.charToInt[self.bos] 
  
  # Get the index of the end of sequence token (integer)
  def getEosIndex(self):
    return self.charToInt[self.eos]
