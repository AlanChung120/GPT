# GPT

Finished and uploaded June 27, 2026. Used Python 3.12.5. This is a simplified GPT clone created using principles of transformers and attention. It trains the GPT using input.txt which has the format prompt and response separated by new line (you can adjust to fit your use). For the given training data (list of prompt and response), it randomly batches them and goes through the encoder which encodes the prompt using self attention. Then it sends the encoded prompt and response to the decoder where the potential response is trained using self and cross attention (with encoded prompt). For more information refer to transformer_model.png. Finally user can chat with the GPT that responds based on the previously trained data.
To Train:
Run the train.py file using installed Python (will depend on the training file input.txt)
To Run:
Run the chat.py file using installed Python.
