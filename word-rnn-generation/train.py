import argparse
import time
import os
from six.moves import cPickle
from utils import TextDataLoader
from model import *
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import pdb
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print("Imported all libraries successfully!")

parser = argparse.ArgumentParser(description='PyTorch Word-RNN Example')
# Data directory
parser.add_argument('--data_dir', type=str, default='../data/',
	help='data directory containing input.txt')
# Batch Size
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
# Number of epochs to train
parser.add_argument('--epochs', type=int, default=19, metavar='N',
                    help='number of epochs to train (default: 2)')
# Whether to use cuda or not
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
# Seed
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# Sequence Length
parser.add_argument('--seq_length', type=int, default=25,
	help='RNN sequence length')
# Input Encoding
parser.add_argument('--input_encoding', type=str, default=None,
	help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')
# RNN's hidden state Size
parser.add_argument('--rnn_size', type=int, default=256, help='size of RNN hidden state')
# Number of layers in RNN
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the RNN')
# RNN/GRU/LSTM?
parser.add_argument('--model', type=str, default='lstm', help='rnn, gru, or lstm')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Intialize custom dataloader
textLoader = TextDataLoader(args.data_dir, args.batch_size, args.seq_length)

text_generation_model  = LSTM(textLoader.vocab_size, args.rnn_size, textLoader.vocab_size, args.num_layers)

if args.cuda:
	text_generation_model = text_generation_model.cuda()
num_batches = textLoader.num_batches

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(text_generation_model.parameters() , lr = 0.001)

train_size  = textLoader.batch_size*args.seq_length
def train(epoch):
	text_generation_model.train()
	textLoader.reset_batch_pointer()
	
	# hidden = text_generation_model.init_hidden()

	for i in range(num_batches):
		train_loss = 0
		data, target = textLoader.next_batch()
		data = Variable(torch.from_numpy(data).cuda())
		target = Variable(torch.from_numpy(target).cuda())
		


		optimizer.zero_grad()

		output, hidden = text_generation_model(data)
		pdb.set_trace()
		loss = criterion(output, target)
		train_loss += loss.data[0]
		loss.backward()
		optimizer.step()

		if (i+1) % 100 == 0:
                    print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, num_epochs, i+1, num_batches, loss.data[0]))

	# print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


for epoch in range(args.epochs):
	train(epoch)
