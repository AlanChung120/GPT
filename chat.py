import torch
from transformer import Transformer
import pickle

if __name__ == '__main__':
  
  # load trained data
  FILE = "model.pth"
  trainingData = torch.load(FILE)

  # load language model
  with open("lm.pkl", "rb") as file:
    lm = pickle.load(file)

  # load parameters
  modelState = trainingData["modelState"]
  MAXPROMPTLENGTH = trainingData["maxPromptLength"]
  blockSize = trainingData["blockSize"]
  nEmbed = trainingData["nEmbed"]
  attentionHeadSize = trainingData["attentionHeadSize"]
  attentionNumHeads = trainingData["attentionNumHeads"]
  numLayers = trainingData["numLayers"]
  dropout = trainingData["dropout"]

  # set device and model
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Transformer(nEmbed, lm.vocabSize, MAXPROMPTLENGTH, blockSize, attentionHeadSize, attentionNumHeads, numLayers, dropout).to(device)

  # load the learned parameters from the file
  model.load_state_dict(modelState)

  # set the model to evaluation mode (bc some layers wil behave differently ex. dropout batchnorm layers)
  model.eval()

  # chat loop
  botName = "ChatBot"
  print("Let's chat! type 'quit' or 'q' to exit")
  while True:
    prompt = input("You: ")
    if prompt == "quit" or prompt == "q":
      break
    elif len(prompt) > MAXPROMPTLENGTH:
      print("Prompt is too long. Try again")
      continue

    encodedPrompt = torch.tensor([lm.encode(prompt)], dtype=torch.long)

    encodedPrompt = encodedPrompt.to(device)

    # generate from the model (\n is the beginning of the sequence and end of the sequence token for decoder (generator))
    context = torch.zeros((1, 1), dtype=torch.long, device=device) # feed the new line character "\n" (0) as the starting sequence/context
    eosToken = 0 # set the new line character "\n" (0) as the ending token
    response = lm.decode(model.generateUntil(encodedPrompt, context, eosToken, blockSize, device)[0].tolist()[1:-1]) # generate from the initial context get the first batch and decode it
    if response == '':
      print(f'{botName}: Not trained to do this yet.')
    else:
      print(f'{botName}: {response}')
