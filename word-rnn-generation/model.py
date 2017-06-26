import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import pdb



# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.fc_in = nn.Linear(input_size,hidden_size)
        #self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

<<<<<<< HEAD
    def forward(self, x):
        
        x = self.encoder(x)
        output, hidden = self.lstm(x)
        pdb.set_trace()
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
"""
# Not needed for new pytorch version

    def init_hidden(self):
    	return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
"""
=======
    def forward(self, x, hidden):
        #input = self.encoder(input.view(1, -1))
        t,n = x.size(0), x.size(0)
        x = x.contiguous().view(t*n, -1)
        output, hidden = self.fc_in(input)
        output = output.contiguous().view(t, n, -1)
        #output, hidden = self.lstm(input.view(1, 1, -1), hidden)
        output, hidden = self.lstm(output)
        output = output.contiguous().view(t*n,-1)
        output = self.decoder(output)
        output = output.contiguous().view(t, n, -1)
        #output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
    	return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

# RNN model
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.GRU(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
    	return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
>>>>>>> c579ced07c46c40a3f0a8fab862abc4935ef0675
