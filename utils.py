import re
import sys
from itertools import permutations

import matplotlib.pyplot as plt
from matplotlib import rc

meter_pattern = r"M: ?[0-9]/[0-9]\n"
length_pattern = r"L: ?1/(1|2|4|8|16|32|64|128|256|512)\n"
key_pattern = r"K: ?((C\|?)|([A-G][b#]?(mix|dor|phr|lyd|loc|m)?))\n"
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

	return bool(re.match(valid_abc_pattern, tune))