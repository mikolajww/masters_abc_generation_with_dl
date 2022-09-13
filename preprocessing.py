import datetime
import pprint
import re
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt

import utils

root_folder = r"../data/to_be_processed"
target_file = f"abc_combined_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.abc"


metas_to_ignore = [
    r"(^[ABCDFGHImNOPQRrSsTUVWwXZ]:.*$)",  # Header fields other than K L M
    r"(%.*$)", # comments starting with %
    r"(`)", # backticks - effectively whitespace
    r"""("[^"]+")""", # quoted chords - backing track
    r"\{[^\}].*\}", # grace notes
    r"(transpose=?-?[0-9]{1,2})", # transpositions
    r"(clef=treble)|(clef=bass)|(clef=none)|(clef=alto)|(clef=tenor)|(clef=[A-G])" # non-standard clefs
]

to_ignore = "|".join(metas_to_ignore)


songs = []
# get some statistics
n_files = {".abc": 0, ".txt": 0}
n_abc_songs = 0
missing_m, missing_l, invalid = 0, 0, 0

counter = Counter()

def try_open(path, encoding, songs):
    global n_abc_songs, missing_m, missing_l, invalid
    with open(path, "r", encoding=encoding) as f:
        lines = "".join(f.readlines()).split("X:")
        tunes_in_file = list(
            map(lambda x:
                x[x.index("\n") + 1:],
                filter(lambda x:
                       len(x) > 0,
                       lines)
                )
        )
        for tune in tunes_in_file:
            cleaned_tune = ""
            for line in tune.split("\n"):
                if line.startswith("%"):
                    continue
                cleaned_line = re.sub(to_ignore, "", line)
                cleaned_line = re.sub("([Mm]ajor)|([Ii]onian)|([Mm]aj)", "", cleaned_line, re.IGNORECASE)
                cleaned_line = re.sub("([Mm]inor)|([Aa]eolian)|([Mm]in)", "m", cleaned_line, re.IGNORECASE)
                cleaned_line = re.sub("([Mm]ixolydian)|([Mm]ixolyd)|(Mix)", "mix", cleaned_line, re.IGNORECASE)
                cleaned_line = re.sub("([Dd]orian)|(Dor)", "dor", cleaned_line, re.IGNORECASE)
                cleaned_line = re.sub("([Pp]hrygian)|(Phr)", "phr", cleaned_line, re.IGNORECASE)
                cleaned_line = re.sub("([Ll]ydian)|(Lyd)", "lyd", cleaned_line, re.IGNORECASE)
                cleaned_line = re.sub("([Ll]ocrian)|(Loc)", "loc", cleaned_line, re.IGNORECASE)
                if cleaned_line.strip(): # line is not empty or whitespace
                    if m := re.match("(^K:.*$)|(^M:.*$)|(^L:.*$)", cleaned_line):
                        cl_str = m.string.replace(" ", "").replace("\n", "").lower()
                        counter[cl_str] += 1
                    cleaned_tune += cleaned_line.replace(" ", "") + "\n"
            if not cleaned_tune:
                # empty song - nothing was left
                continue
            if not re.search(utils.meter_pattern, cleaned_tune, re.MULTILINE):
                missing_m += 1
                cleaned_tune = "M:4/4\n" + cleaned_tune
            if not re.search(utils.length_pattern, cleaned_tune, re.MULTILINE):
                missing_l += 1
                cleaned_tune = "L:1/8\n" + cleaned_tune
            if not utils.is_valid_abc(cleaned_tune):
                if len(cleaned_tune) > 5000:
                    print("Long")
                invalid += 1
                continue
            songs.append(cleaned_tune +"\n\n")
            n_abc_songs += 1

encodings_to_try = [None, "utf-8", "raw-unicode-escape"]

for ext in [".abc", ".txt"]:

    for i, path in enumerate(Path(root_folder).rglob(f"*{ext}")):
        for e in encodings_to_try:
            try:
                try_open(path, None, songs)
                break
            except UnicodeDecodeError as e:
                continue
            except Exception as exc:
                print(f"Unexpected exception {exc.__class__} : {exc}")
                continue
        # print(f"{i} - {path}")
        n_files[ext] += 1

with open(target_file, "w", encoding="utf-8") as f:
    f.writelines(songs)

with open(target_file + "_stats.txt", "w") as f:
    f.writelines(pprint.pformat(counter))
    f.writelines(f"\n{missing_m=},{missing_l=}, {invalid=}")
    f.writelines(f"\n{n_abc_songs=}, {n_files=}")

key_counter = Counter()
length_counter = Counter()
meter_counter = Counter()
for k, v in counter.items():
    if k.startswith("k"):
        k = re.sub("maj", "", k, re.IGNORECASE)
        k = re.sub("min", "m", k, re.IGNORECASE)
        k = re.sub("k:", "", k, re.IGNORECASE)
        key_counter[k] += v
    if k.startswith("l"):
        k = re.sub("l:", "", k, re.IGNORECASE)
        length_counter[k] += v
    if k.startswith("m"):
        k = re.sub("m:", "", k, re.IGNORECASE)
        meter_counter[k] += v

# C = 4/4    C| = 2/2
meter_counter["4/4"] += meter_counter["c"]
del meter_counter["c"]

meter_counter["2/2"] += meter_counter["c|"]
del meter_counter["c|"]

utils.setup_matplotlib_style()

# plt.figure()
# keys = key_counter.most_common(10)
# x_keys = [a[0].title() for a in keys]
# y_keys = [a[1] for a in keys]
# plt.suptitle("Most common key signatures in the dataset")
# plt.bar(x_keys, y_keys, color='blue', width=1.0, edgecolor='black', linewidth=1.5)
# plt.xlabel("Number of tunes")
# plt.ylabel("Key signature")
# plt.savefig(r"C:\Users\Mikolaj\Desktop\Magisterka\abc_music_generation_with_dl\key_signatures.png")
#
# plt.figure()
# meters = meter_counter.most_common(10)
# x_meters = [a[0].title() for a in meters]
# y_meters = [a[1] for a in meters]
# plt.suptitle("Most common time signatures in the dataset")
# plt.bar(x_meters, y_meters, color='blue', width=1.0, edgecolor='black', linewidth=1.5)
# plt.xlabel("Number of tunes")
# plt.ylabel("Meter")
# plt.savefig(r"C:\Users\Mikolaj\Desktop\Magisterka\abc_music_generation_with_dl\meter_signatures.png")
#
# plt.figure()
# lengths = length_counter.most_common(5)
# x_lengths = [a[0].title() for a in lengths]
# y_lengths = [a[1] for a in lengths]
# plt.suptitle("Most common base note lengths in the dataset")
# plt.bar(x_lengths, y_lengths, color='blue', width=1.0, edgecolor='black', linewidth=1.5)
# plt.xlabel("Number of tunes")
# plt.ylabel("Base note length")
# plt.savefig(r"C:\Users\Mikolaj\Desktop\Magisterka\abc_music_generation_with_dl\length_signatures.png")


print(f"Found {sum(n_files.values())} ({n_files=}) files ")
print(f"Found {n_abc_songs} total songs")
print(f"{missing_l=}, {missing_m=}, {invalid=}")
