import torch

class colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0;37m'

def a_print(text, truth, err_color):
    for pair in zip(text, truth):
        if pair[0] == pair[1]: color = colors.GREEN
        else: 
            if err_color == 'red':
                color = colors.RED
            if err_color == 'yellow':
                color = colors.YELLOW
        print(f'{color}{pair[0]}', end='')
    print(colors.RESET)
