import datetime
import os
import pprint
import re
import sys
from itertools import permutations
from prettytable import PrettyTable
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib import rc
import pickle
import numpy as np
from torch.nn.utils.rnn import pad_sequence

meter_pattern = r"M: ?((C\|?)|([0-9][0-9]?/[0-9][0-9]?))\n"
length_pattern = r"L: ?1/(1|2|4|8|16|32|64|128|256|512)\n"
key_pattern = r"K: ?([A-G][b#]?(mix|dor|phr|lyd|loc|m)?)\n"
tune = r"[\d\D]+"

header_permutations = permutations([meter_pattern, key_pattern, length_pattern])

header_regex = []
for p in header_permutations:
    header_regex.append("".join(p))

header_regex = "|".join(["(" + s + ")" for s in header_regex])
header_regex = "(" + header_regex + ")"
header_regex += tune
valid_abc_pattern = re.compile(header_regex)


def setup_matplotlib_style():
    plt.style.use(['high-vis'])
    rc('figure', **{'dpi': 150, 'figsize': (12, 7)})
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})
    rc('text', usetex=True)
    rc('legend', **{'fontsize': 18})

    rc('axes', **{'grid': True, 'axisbelow': 'True'})
    rc('grid', **{'linestyle': '-.', 'alpha': 0.3, 'color': 'k'})
    rc('xtick', **{'direction': 'in', 'top': True, 'bottom': True})
    rc('ytick', **{'direction': 'in', 'left': True, 'right': True})


def get_sizeof(var, suffix='B'):
    """ by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
    num = sys.getsizeof(var)
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def is_valid_abc(tune):
    return bool(re.search(valid_abc_pattern, tune))


# base_folder.joinpath("history.pkl")
def plot_training_history_from_pickle(base_folder, pkl_name):
    hist = pickle.load(open(f"{base_folder}/{pkl_name}", "rb"))
    plt.plot(np.arange(1, len(hist["train"]["loss"]) + 1), hist["train"]["loss"], label="Training loss")
    plt.plot(np.arange(1, len(hist["val"]["loss"]) + 1), hist["val"]["loss"], label="Validation loss")
    plt.legend()
    plt.title(
        f"Minimum traning loss: {np.array(hist['train']['loss']).min() :.5f}\nMinimum validation loss: {np.array(hist['val']['loss']).min():.5f}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Crossentropy)")
    plt.savefig(f"{base_folder}/training_val_losses.png")
    plt.show()


def save_training_attempt(model, CONFIG, history, picklable_extra=None, np_extra=None, txt_extra=None):
    state = {
        "model_state_dict": model.state_dict(),
        "CONFIG": CONFIG,
        "model": model
    }

    folder_name = f"experiment_{model.__class__.__name__}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.mkdir(folder_name)
    save_filename = f"{model.__class__.__name__}.pth"
    torch.save(state, f"{folder_name}/{save_filename}")
    pickle.dump(history, open(f"{folder_name}/history.pkl", "wb"))

    with open(f"{folder_name}/model_parameters.txt", "w") as f:
        total_params = 0
        table = PrettyTable(["Modules", "Parameters"])
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            param_count = param.numel()
            table.add_row([name, param_count])
            total_params += param_count
        f.writelines(repr(table)+"\n")
        f.write(f"Total parameters: {total_params}\n")


    with open(f"{folder_name}/time.txt", "w") as f:
        f.writelines(str(history["epoch_times"]))
    with open(f"{folder_name}/experimet_summary.txt", "a") as f:
        f.writelines(folder_name + "\n")
        f.writelines(pprint.pformat(CONFIG) + "\n")
        f.write(f"Time : {np.array(history['epoch_times']).mean() :.2f}\n")
        f.write("\n\n")
    print(f"Experiment saved to {folder_name}")

    if np_extra is not None:
        for name, value in np_extra.items():
            np.save(f"{folder_name}/{name}.npy", value)

    if txt_extra is not None:
        for name, value in txt_extra.items():
            with open(f"{folder_name}/{name}.txt", "w") as f:
                f.writelines(value)

    if picklable_extra is not None:
        for name, value in picklable_extra.items():
            pickle.dump(value, open(f"{folder_name}/{name}.pkl", "wb"))

    return f"{folder_name}/{save_filename}"

def save_evaluation_attempt():
    ...


def init_history():
    return {
        "train": {
            "loss": []
        },
        "val": {
            "loss": []
        },
        "epoch_times": []
    }


def pad_collate(batch, device, pad_idx):
    (x_batch, y_batch) = zip(*batch)

    len_x = torch.tensor([len(x) for x in x_batch]).long()
    len_y = torch.tensor([len(y) for y in y_batch]).long()
    # TODO MAKE SURE THIS VALUE CORRESPONDS TO THE PAD_IDX
    x_pad = pad_sequence(x_batch, batch_first=True, padding_value=pad_idx).to(device)
    y_pad = pad_sequence(y_batch, batch_first=True, padding_value=pad_idx).to(device)

    return x_pad, y_pad, len_x, len_y

