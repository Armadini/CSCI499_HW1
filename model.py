# IMPLEMENT YOUR MODEL CLASS HERE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class ArmansSuperDuperLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, actionset_size, objectset_size):
        super(ArmansSuperDuperLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to action space
        self.hidden2action = nn.Linear(hidden_dim, actionset_size)
        # The linear layer that maps from hidden state space to object space
        self.hidden2object = nn.Linear(hidden_dim, objectset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out[:, -1, :]
        # Actions
        action_space = self.hidden2action(lstm_out)
        action_scores = F.log_softmax(action_space, dim=1)
        # Objects
        object_space = self.hidden2object(lstm_out)
        object_scores = F.log_softmax(object_space, dim=1)
        return action_scores, object_scores