import datetime
import os
import pickle
import pprint
import re
import time
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader

from dataset import ABCInMemoryDataset, split_train_valid_test_dataloaders
from utils import setup_matplotlib_style, is_valid_abc


class RNNSimpleGenerator(nn.Module):
	def __init__(
			self,
			vocab,
			input_features,
			hidden_size,
			n_layers,
			dropout_prob
	):
		super().__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.dropout_prob = dropout_prob

		self.vocab = vocab
		# self.embedding = nn.Embedding()
		self.lstm = nn.LSTM(
			input_size=input_features,
			hidden_size=self.hidden_size,
			num_layers=self.n_layers,
			bias=True,
			batch_first=True,
			dropout=dropout_prob if hidden_size > 1 else 0,
			bidirectional=False
		)

		if hidden_size == 1:
			self.dropout = nn.Dropout(dropout_prob)
		else:
			self.dropout = nn.Dropout(0)
		self.n_classes = len(self.vocab)
		self.fc = nn.Linear(self.hidden_size, self.n_classes)

	def forward(self, X, len_X, initial_hidden):
		# X is padded from pad_collate
		# One hot or embed
		X = F.one_hot(X.long(), num_classes=self.n_classes).float()
		# Pack
		X = pack_padded_sequence(X, len_X, batch_first=True, enforce_sorted=False).to(DEVICE)
		X, h = self.lstm(X, initial_hidden)
		# Undo pack
		X, len_X = pad_packed_sequence(X, batch_first=True)
		X = self.dropout(X)
		X = self.fc(X)
		# X = self.softmax(X)
		return X, len_X, h

	def init_hidden(self, batch_size, device):
		# n_states = 2 for LSTM - hidden state, cell state
		return torch.zeros(size=(self.n_layers, batch_size, self.hidden_size)).to(device)


def pad_collate(batch):
	(x_batch, y_batch) = zip(*batch)

	len_x = torch.tensor([len(x) for x in x_batch]).long()  # .to(DEVICE)
	len_y = torch.tensor([len(y) for y in y_batch]).long()  # .to(DEVICE)
	# TODO MAKE SURE THIS VALUE CORRESPONDS TO THE PAD_IDX
	x_pad = pad_sequence(x_batch, batch_first=True, padding_value=0).to(DEVICE)
	y_pad = pad_sequence(y_batch, batch_first=True, padding_value=0).to(DEVICE)

	return x_pad, y_pad, len_x, len_y



def padded_crossentropy_loss(predictions, len_predictions, targets, len_targets):
	batch_ce_loss = torch.tensor(0.0, device=DEVICE)
	for i in range(predictions.size(0)):
		ce = F.cross_entropy(
			predictions[i][:len_predictions[i]],
			targets[i][:len_targets[i]],
			reduction="sum", ignore_index=0)
		ce = ce / len_predictions[i].cuda()
		batch_ce_loss += ce

	return batch_ce_loss / predictions.size(0)


def train():
	dataset = ABCInMemoryDataset(CONFIG["path_to_abc"], max_len=CONFIG["max_len"], cut_or_filter="cut")

	train_data_loader, valid_data_loader, test_data_loader = split_train_valid_test_dataloaders(
		dataset, train_percent=0.8, valid_percent=0.1, batch_size=CONFIG["batch_size"], collate_fn=pad_collate)

	model = RNNSimpleGenerator(
		input_features=len(dataset.vocabulary),
		vocab=dataset.vocabulary,
		hidden_size=CONFIG["hidden_size"],
		n_layers=CONFIG["n_recurrent_layers"],
		dropout_prob=CONFIG["dropout_prob"]
	).to(DEVICE)

	optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode='min', factor=0.1, patience=2, verbose=True, threshold=0.0001,
		threshold_mode='rel', cooldown=0, min_lr=0.0000001, eps=1e-08
	)

	loss_fn = padded_crossentropy_loss  # nn.NLLLoss()#loss_fn = nn.CrossEntropyLoss()

	n_classes = len(dataset.vocabulary)

	history = {
		"train": {
			"loss": []
		},
		"val": {
			"loss": []
		}
	}

	model.train()
	epoch_times = []
	for epoch in range(CONFIG["epochs"]):
		torch.cuda.empty_cache()
		epoch_start_time = time.perf_counter()
		h0 = model.init_hidden(batch_size=CONFIG["batch_size"], device=DEVICE)
		c0 = model.init_hidden(batch_size=CONFIG["batch_size"], device=DEVICE)

		n_batches = len(train_data_loader)
		minibatch_losses = torch.tensor(np.zeros(n_batches), device=DEVICE)
		for i, (batch_X, batch_y, len_X, len_y) in enumerate(tqdm.tqdm(train_data_loader)):
			# this is equivalent to optimizer.zero_grad()
			# reset gradients every minibatch!
			# set_to_none supposedly is more efficient as per
			# https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
			model.zero_grad(set_to_none=True)

			# detach hidden state - we don't want to backpropagate through batches
			h0 = h0.detach()
			c0 = c0.detach()

			# pass data through a model
			output_packed, len_packed, (h, c) = model(batch_X, len_X, (h0, c0))
			targets = batch_y.long()

			# calculate loss and backpropagate gradients
			loss = loss_fn(output_packed, len_packed, targets, len_y)
			minibatch_losses[i] = loss.data
			loss.backward()

			# Prevent exploding gradients
			nn.utils.clip_grad_norm_(model.parameters(), 5)
			# Adjust learning weights
			optimizer.step()

			if i % 50 == 0:
				tqdm.tqdm.write(f"Epoch {epoch + 1}/{CONFIG['epochs']} - Loss: {loss.item()}")

		# on epoch end
		# average out the per-batch losses
		history["train"]["loss"].append(torch.nanmean(minibatch_losses).item())

		# Evaluate on validation dataset without gradients
		with torch.no_grad():
			h0 = model.init_hidden(batch_size=CONFIG["batch_size"], device=DEVICE)
			c0 = model.init_hidden(batch_size=CONFIG["batch_size"], device=DEVICE)

			n_batches = len(valid_data_loader)
			val_minibatch_losses = torch.tensor(np.zeros(n_batches), device=DEVICE)
			for i, (batch_X, batch_y, len_X, len_y) in enumerate(valid_data_loader):
				# batch_X = F.one_hot(batch_X.long(), num_classes=n_classes).float()#.to(DEVICE)
				# batch_y = F.one_hot(batch_y.long(), num_classes=n_classes).float()#.to(DEVICE)
				h0 = h0.detach()
				c0 = c0.detach()
				output_packed, len_packed, (h, c) = model(batch_X, len_X, (h0, c0))
				# predictions = rearrange(output, 'b t c -> b c t')
				targets = batch_y.long()
				val_loss = loss_fn(output_packed, len_packed, targets, len_y)
				val_minibatch_losses[i] = val_loss.data

		history["val"]["loss"].append(torch.nanmean(val_minibatch_losses).item())
		# tb_writer.add_scalar("Validation loss", history["val"]["loss"][epoch], epoch)
		epoch_time = time.perf_counter() - epoch_start_time
		epoch_times.append(epoch_time)
		tqdm.tqdm.write(
			f"Epoch {epoch + 1}: [{epoch_time:.2f}s] Train loss: {history['train']['loss'][epoch]} | Val loss: {history['val']['loss'][epoch]}")

		scheduler.step(val_loss)
	# on train end
	state = {
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"CONFIG": CONFIG,
		"vocab": model.vocab,
		"model": model
	}

	folder_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
	os.mkdir(folder_name)
	save_filename = f"{model.__class__.__name__}.pth"
	torch.save(state, f"{folder_name}/{save_filename}")
	pickle.dump(history, open(f"{folder_name}/history.pkl", "wb"))
	with open(f"{folder_name}/params.txt", "w") as f:
		f.writelines(pprint.pformat(CONFIG))
	with open(f"{folder_name}/time.txt", "w") as f:
		f.writelines(str(epoch_times))

	with open(f"{folder_name}/experimet_summary.txt", "a") as f:
		f.writelines(folder_name + "\n")
		f.writelines(pprint.pformat(CONFIG) + "\n")
		f.write(f"Time : {np.array(epoch_times).mean() :.2f}\n")
		f.write("\n\n")

	print(f"Experiment saved to {folder_name}")



def try_generate(model, eos):
	start_character = "<BOS>"
	attempt = 0
	tries = []
	char2int = lambda t: model.vocab.lookup_indices(t)
	int2char = lambda t: model.vocab.lookup_tokens(t)
	with torch.no_grad():
		while True:
			generated_tokens = [start_character]
			attempt += 1

			h = model.init_hidden(batch_size=1, device=DEVICE)
			c = model.init_hidden(batch_size=1, device=DEVICE)
			generated_tok = None
			while generated_tok != eos:
				x = torch.tensor(char2int([generated_tokens[-1]])).expand(1, 1)  # 1 element batch
				if x.ndimension() == 1:
					len_x = torch.tensor([len(x)]).long()
				else:
					len_x = [len(t) for t in x]
					len_x = torch.tensor(len_x).long()
				y_pred, len_y, (h, c) = model(x, len_x, (h, c))
				# unpad
				for i in range(y_pred.size(0)):
					y_pred[i] = y_pred[i][:len_y[i]]
				# y_pred = (Batch, time step, character (1-hot))
				p = F.softmax(y_pred, dim=2).cpu().numpy().squeeze()

				generated_tok = np.random.choice(y_pred.size(2), p=p)
				generated_tok = int2char([generated_tok])[0]
				generated_tokens.append(generated_tok)
			generated_tune = "".join(generated_tokens).replace(start_character, "").replace(eos, "")
			if is_valid_abc(generated_tune):
				break
			else:
				tries.append(generated_tune)
		return generated_tune, tries, attempt


def evaluate(model_path):
	print(f"Loading model {model_path}")
	state = torch.load(model_path)
	CONFIG = state["CONFIG"]
	model = state["model"]
	model.load_state_dict(state["model_state_dict"])
	model.eval()

	eos_tok = "<EOS>"

	TRIES = 500
	print(model)
	print(f"Evaluating model for {TRIES} tries")
	base_folder = Path(model_path).parent
	tunes_out_folder = base_folder.joinpath("output_tunes")
	Path.mkdir(tunes_out_folder, exist_ok=True)

	eval_start = time.perf_counter()
	eval_times = []
	n_of_attempts_list = []
	for i in tqdm.trange(TRIES):
		eval_attempt_start = time.perf_counter()
		generated_tune, tries, n_of_attempts = try_generate(model, eos_tok)
		eval_times.append(time.perf_counter() - eval_attempt_start)
		with open(f"{tunes_out_folder}/tune_{i}_correct_attempts_{n_of_attempts}.abc", "w") as out:
			out.writelines(generated_tune)
		with open(f"{tunes_out_folder}/tune_{i}_incorrect_attempts_{n_of_attempts}.abc", "w") as out:
			out.writelines(tries)
		n_of_attempts_list.append(n_of_attempts)

	eval_end = time.perf_counter()

	with open(f"{base_folder}/n_of_tries.txt", "w") as f:
		f.writelines(str(n_of_attempts_list))

	with open(f"{base_folder}/eval_times.pkl", "wb") as f:
		pickle.dump(eval_times, f)

	eval_times = np.array(eval_times)

	hist = pickle.load(open(base_folder.joinpath("history.pkl"), "rb"))
	plt.plot(np.arange(1, len(hist["train"]["loss"]) + 1), hist["train"]["loss"], label="Training loss")
	plt.plot(np.arange(1, len(hist["val"]["loss"]) + 1), hist["val"]["loss"], label="Validation loss")
	plt.legend()
	plt.title(f"Minimum traning loss: {np.array(hist['train']['loss']).min() :.5f}\nMinimum validation loss: {np.array(hist['val']['loss']).min():.5f}" )
	plt.xlabel("Epoch")
	plt.ylabel("Loss (Crossentropy)")
	plt.savefig(f"{Path(model_path).parent}/training_val_losses.png")
	plt.show()

	eval_summary_str = f"Took {eval_end - eval_start:.2f}s to evaluate. [min = {eval_times.min()}, avg = {eval_times.mean()}, max = {eval_times.max()}]"
	with open(f"{base_folder}/model_summary.txt", "w") as f:
		f.writelines(pprint.pformat(CONFIG))
		f.write("\n")
		f.writelines(str(model))
		f.writelines(eval_summary_str)

	print(eval_summary_str)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CONFIG = OrderedDict([
	("path_to_abc", "../data_processed/abc_parsed_cleanup2.abc"),
	("batch_size", 64),
	("lr", 0.001),
	("hidden_size", 1024),
	("n_recurrent_layers", 3),
	("dropout_prob", 0.4),
	("epochs", 10),
	("cut_or_filter", "cut"),
	("max_len", 700)
])


if __name__ == "__main__":
	setup_matplotlib_style()
	# mode = "train"
	mode = "eval"
	model_path = "experiment_20220721-150209/RNNSimpleGenerator.pth"
	if mode == "train":
		train()
	elif mode == "eval":
		evaluate(model_path)
	else:
		hist = pickle.load(open("experiment_20220715-161305_no_L_metatag/history.pkl", "rb"))

		print(hist)