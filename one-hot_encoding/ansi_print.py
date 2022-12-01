import torch

class colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0;37m'

def a_print(text, truth):
    for pair in zip(text, truth):
        if pair[0] == pair[1]: color = colors.GREEN
        else: color = colors.RED
        print(f'{color}{pair[0]}', end='')
    print(colors.RESET)
