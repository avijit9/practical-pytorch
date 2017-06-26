import os
import collections
import numpy as np
import re
import itertools
import codecs
from six.moves import cPickle

"""
TextDataLoader class mostly based on https://github.com/hunkim/word-rnn-tensorflow/blob/master/utils.py

"""
class TextDataLoader():
	def __init__(self, data_dir, batch_size, seq_length, encoding = None):
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.seq_length = seq_length

		input_file = os.path.join(data_dir,'input.txt')
		vocab_file = os.path.join(data_dir,'vocab.pkl')
		tensor_file = os.path.join(data_dir,'data.npy')

		print("reading text file")

		self.preprocess(input_file, vocab_file, tensor_file, encoding)

		self.create_batches()
		self.reset_batch_pointer()


	def build_vocab(self, sentences):
		"""
		Build a vocabulary mapping from word to index based on the sentences.
		return both forward and inverse mapping
		"""
		# Build the vocab
		word_counts =  collections.Counter(sentences)

		# Mapping from index to word (get the indices of most common words)
		vocab_inv = [x[0] for x in word_counts.most_common()] # Do we need this?
		vocab_inv = list(sorted(vocab_inv))

		# Mapping from word to index

		vocab = {x: i for i,x in enumerate(vocab_inv)}

		return [vocab, vocab_inv]



	def preprocess(self,input_file, vocab_file, tensor_file, encoding):
		with codecs.open(input_file, "r", encoding = encoding) as f:
			data = f.read()

			# Get the words
			text = data.split()

			self.vocab, self.words = self.build_vocab(text) # vocab: dictionary and words: frequent words

			self.vocab_size = len(self.words)

			with open(vocab_file, 'wb') as f:
				cPickle.dump(self.words, f)

			self.tensor = np.array(list(map(self.vocab.get, text)))

			np.save(tensor_file,self.tensor)

	def create_batches(self):
		self.num_batches = int(self.tensor.size / (self.batch_size*self.seq_length))
		if self.num_batches == 0:
			assert False, "Not enough data!"
		# To prevent the number of words/batch_size*seq_length not a integer
		self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
		x_data = self.tensor
		y_data = np.copy(self.tensor)

		y_data[:-1] = x_data[1:]
		y_data[-1] = x_data[0]

		self.x_batches = np.split(x_data.reshape(self.batch_size, -1), self.num_batches, 1)
		self.y_batches = np.split(y_data.reshape(self.batch_size, -1), self.num_batches, 1)

	def next_batch(self):
		x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
		self.pointer += 1
		return x, y

	def reset_batch_pointer(self):
		self.pointer = 0





