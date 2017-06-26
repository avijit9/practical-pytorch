import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms




# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        pdb.set_trace()
        input = self.encoder(input.view(1, -1))
        output, hidden = self.lstm(input.view(1, 1, -1))
        output = self.decoder(output.view(1, -1))
        return output, hidden
"""
# Not needed for new pytorch version

    def init_hidden(self):
    	return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
"""
