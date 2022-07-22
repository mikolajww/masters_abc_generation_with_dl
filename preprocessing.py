import re
from pathlib import Path
from collections import Counter
root_folder = r"../../data/to_be_processed"
target_file = "abc_combined.abc"


metas_to_ignore = [
    r"(^[ABCDFGHImNOPQRrSsTUCWwXZ]:.*$)",  # Header fields other than K L M
    r"(%.*$)", # comments starting with %
    r"(`)", # backticks - effectively whitespace
    r"""("[^"]+")""", # quoted chords - backing track
    r"\{[^\}].*\}", # grace notes
    r"(transpose=?-?[0-9]{1,2})", # transpositions
    r"(clef=treble)|(clef=bass)|(clef=none)|(clef=alto)|(clef=tenor)" # non-standard clefs
]

to_ignore = "|".join(metas_to_ignore)


songs = []
# get some statistics
n_files = {".abc": 0, ".txt": 0}
n_abc_songs = 0


def try_open(path, encoding, songs):
    global n_abc_songs
    with open(path, "r", encoding=encoding) as f:
        found_first_song = False
        abc_data = "" #"<*SOF*>\n"
        for line in f:
            if not line.isspace():
                if re.match(r"^X:", line):
                    if not found_first_song:
                        abc_data += "\n" #"<*SOS*>\n"
                        found_first_song = True
                    else:
                        abc_data += "\n" #"<*EOS*>\n<*SOS*>\n"
                    n_abc_songs += 1
                cleaned_line = re.sub(to_ignore, "", line)
                cleaned_line = re.sub("(Major)|(ionian)", "", cleaned_line, re.IGNORECASE)
                cleaned_line = re.sub("(Minor)|(aeolian)", "m", cleaned_line, re.IGNORECASE)
                cleaned_line = re.sub("(mixolydian)", "mix", cleaned_line, re.IGNORECASE)
                cleaned_line = re.sub("(dorian)", "dor", cleaned_line, re.IGNORECASE)
                cleaned_line = re.sub("(phrygian)", "phr", cleaned_line, re.IGNORECASE)
                cleaned_line = re.sub("(lydian)", "lyd", cleaned_line, re.IGNORECASE)
                cleaned_line = re.sub("(locrian)", "loc", cleaned_line, re.IGNORECASE)
                if not cleaned_line.isspace():
                    abc_data += cleaned_line.replace(" ", "")
        songs.append(abc_data+"\n") #+ "<*EOS*>\n") #+ "\n<*EOF*>\n")

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
        print(f"{i} - {path}")
        n_files[ext] += 1

with open(target_file, "w", encoding="utf-8") as f:
    f.writelines(songs)

print(f"Found {sum(n_files.values())} ({n_files=}) files ")
print(f"Found {n_abc_songs} total songs")