import re
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
import torchtext
from torch.utils.data import Dataset, DataLoader
import seaborn as sns

from utils import get_sizeof, setup_matplotlib_style


def read_file_to_str_array(filename):
	with open(filename, "r", encoding="utf-8") as f:
		track_list = f.readlines()
		track_list_one_string = ''.join(track_list)
		track_list_one_string = re.sub('(\n){4,}', '\n\n', track_list_one_string)
		track_list = track_list_one_string.split('\n\n')
	return track_list


def tokenize_track(track):
	tokens = list(track)
	tokens.insert(0, '<BOS>')
	tokens.append('<EOS>')
	return tokens


def tokenize_track_array(track_array):
	output_array = []
	for track in track_array:
		output_array.append(tokenize_track(track))
	return output_array


class ABCDataset(Dataset):

	def __init__(self, abc_file_dir, max_len):
		self.max_len = max_len
		self.tracks_str_array = read_file_to_str_array(abc_file_dir)
		# -2 to ensure space for <BOS> sequence <EOS>
		self.tracks_str_array = list(filter(lambda s: len(s) <= self.max_len - 2, self.tracks_str_array))

		def yield_tokens(tracks_string_array):
			for track in tracks_string_array:
				yield list(track)

		self.BOS_IDX, self.EOS_IDX, self.UNK_IDX, self.PAD_IDX = 0, 1, 2, 4

		specials = ["<BOS>", "<EOS>", "<UNK>", "<PAD>"]

		VOCAB_MIN_FREQ = 100
		self.vocabulary = torchtext.vocab.build_vocab_from_iterator(
			yield_tokens(self.tracks_str_array),
			min_freq=VOCAB_MIN_FREQ,
			specials=specials,
			special_first=True
		)
		self.vocabulary.set_default_index(self.UNK_IDX)
		self.text_to_int_encoder = lambda t: self.vocabulary.lookup_indices(t)
		self.int_to_text_decoder = lambda t: self.vocabulary.lookup_tokens(t)

		self.tokenized_array = tokenize_track_array(self.tracks_str_array)

	def __len__(self):
		return len(self.tracks_str_array)

	def __getitem__(self, index):
		X = self.tokenized_array[index]
		# pad to max_len
		padding = ["<PAD>"] * (self.max_len - len(X))
		X.extend(padding)
		# integer encode, convert to np
		X = np.array(self.text_to_int_encoder(X))
		y = np.roll(X, 1)
		return X, y

	def size_summary(self):
		print("Dataset size summary:")
		for var in self.__dict__:
			print(f"\tself.{var} = {get_sizeof(getattr(self, var))}")


class ABCInMemoryDataset(Dataset):

	def __init__(self, abc_file_dir, max_len, min_len=0, cut_or_filter="filter"):
		self.tracks_str_array = read_file_to_str_array(abc_file_dir)
		self.max_len = max_len
		self.min_len = min_len
		if cut_or_filter == "filter":
			self.tracks_str_array = list(
				filter(lambda s: (len(s) <= self.max_len - 2) and (len(s) > min_len), self.tracks_str_array))
		elif cut_or_filter == "cut":
			self.tracks_str_array = list(
				filter(lambda x: len(x) > min_len, map(lambda s: s[:max_len - 2], self.tracks_str_array)))

		def yield_tokens(tracks_string_array):
			for track in tracks_string_array:
				yield list(track)

		self.PAD_IDX, self.BOS_IDX, self.EOS_IDX, self.UNK_IDX = 0, 1, 2, 4

		specials = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", ]

		VOCAB_MIN_FREQ = 100
		self.vocabulary = torchtext.vocab.build_vocab_from_iterator(
			yield_tokens(self.tracks_str_array),
			min_freq=VOCAB_MIN_FREQ,
			specials=specials,
			special_first=True
		)
		self.vocabulary.set_default_index(self.UNK_IDX)
		self.text_to_int_encoder = lambda t: self.vocabulary.lookup_indices(t)
		self.int_to_text_decoder = lambda t: self.vocabulary.lookup_tokens(t)

		self.data = []
		self.targets = []

		for track in self.tracks_str_array:
			# x = BOS t1 t2 t3 t4 ... t_n
			# y = t1 t2 t3 t4 t5 ... EOS
			x = list(track)
			y = list(track)
			y.append("<EOS>")
			x.insert(0, "<BOS>")
			# integer encode, convert to np
			x = torch.tensor(self.text_to_int_encoder(x), device=DEVICE)
			y = torch.tensor(self.text_to_int_encoder(y), device=DEVICE)
			self.data.append(x)
			self.targets.append(y)

	def __len__(self):
		return len(self.tracks_str_array)

	def __getitem__(self, item):
		# the data, unpadded
		return self.data[item], self.targets[item]


def investigate_data(dataset):
	print(len(dataset.tracks_str_array))

	track_lengths = list(map(lambda x: len(x), dataset.tracks_str_array))
	track_lengths = np.array(track_lengths)
	longest_track_idx = np.argmax(track_lengths)
	shortest_track_idx = np.argmin(track_lengths)
	print(f"Shortest track [idx = {shortest_track_idx}] : {dataset.tracks_str_array[shortest_track_idx]}")
	print(np.mean(track_lengths))

	print(f"Total tracks: {len(dataset.tracks_str_array)}")
	print(
		f"Track length (chars): min={np.min(track_lengths)}, mean={np.mean(track_lengths)}, max={np.max(track_lengths)}")

	len_sorted = sorted(track_lengths)
	print(f"Top 25 smallest lengths: {len_sorted[:25]}, top 25 largest: {len_sorted[-25:]}")

	LEN_TRESH = 800
	longer_than_thresh = len(track_lengths[track_lengths > LEN_TRESH])
	print(
		f"Found {longer_than_thresh} values longer than {LEN_TRESH} ({100 * longer_than_thresh / len(track_lengths):.2f}% of total)")

	is_long = track_lengths > LEN_TRESH

	with open(f"files_longer_than_{LEN_TRESH}.abc", "w", encoding="utf-8") as f:
		for i in range(len(is_long)):
			if is_long[i]:
				f.write(f"Song {i}")
				f.writelines(dataset.tracks_str_array[i])
				f.write("\n\n")

	setup_matplotlib_style()
	plt.figure()
	plt.title("Distribution of tune lengths in the dataset\n(token=character)")
	plt.ylabel("Number of tunes")
	plt.xlabel("Number of tokens in a tune")
	plt.hist(track_lengths, bins=np.arange(0, LEN_TRESH, step=50), edgecolor='black', linewidth=0.5)
	plt.show()

	sns.boxplot(x=track_lengths)
	plt.ylim([0, 20000])
	plt.show()

def split_train_valid_test_dataloaders(
		dataset,
		train_percent,
		valid_percent,
		batch_size,
		collate_fn
):
	len_training = int(train_percent * len(dataset))
	len_valid = int(valid_percent * len(dataset))
	len_test = len(dataset) - len_training - len_valid
	train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
		dataset, [len_training, len_valid, len_test], generator=torch.Generator().manual_seed(123_567)
	)

	dataloader_kwargs = {
		"batch_size": batch_size,
		"shuffle": True,
		# "pin_memory": True,
		"drop_last": True,
		"collate_fn": collate_fn
	}
	train_data_loader = DataLoader(
		train_dataset,
		**dataloader_kwargs
	)
	valid_data_loader = DataLoader(
		valid_dataset,
		**dataloader_kwargs
	)
	test_data_loader = DataLoader(
		test_dataset,
		**dataloader_kwargs
	)
	return train_data_loader, valid_data_loader, test_data_loader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
	path_to_abc = '../data_processed/abc_parsed_cleanup4.abc'
	start_time = time.perf_counter()
	dataset = ABCInMemoryDataset(path_to_abc, min_len=0, max_len=100000, cut_or_filter="filter")
	print(f"Took {time.perf_counter() - start_time} to load the dataset")
	investigate_data(dataset)





