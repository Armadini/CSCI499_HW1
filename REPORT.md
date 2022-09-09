# Final Report

## Implementation Decisions

### Research:
Unfortunately, my computer is very slow so I was unable to train a model on the full dataset. When trying to do so, I waited over 5 minutes for a single epoch to finish and it still had not. However, I was able to train smaller models with subsets of the dataset. As such, I don't have specific design choices in terms of hyperparameters, but I will analyze their theoretical value:

- Embedding size: This did not need to be too large as I thought that the meaning of words was not overwhelmingly important. It seemed that the model was really just looking for two main pices of information, so a large embedding size for extracting word meaning was not necessary.
- Hidden Dimension: This was also not too large as the model was just looking for two pieces of info
- LSTM: I only had one LSTM layer in order to save training time. I could've stacked multiple on top of each other form additional sequence passes, but I also felt that was uncessecary for this task
- Linear heads for each prediction: While I coudl've trained two seperate models entirely for each task, I felt that one LSTM was enough to capture info about the action and target of the instruction, and the hidden dimension was also large enough. As such, I only used the final linear layer to decode the final hidden state into two labels


### Engineering:
The data loaders were purposefully split into their own file. This was to allow for more functionality as I created custom Dataset classes. Moreover, within the dataloaders, I added a "debug" option which I have found to be very helpful in past AI projects. This effectively reduced the size of the dataset allowing me to train quickly.