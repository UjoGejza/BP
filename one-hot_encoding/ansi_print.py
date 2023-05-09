# ansi_print.py
# Author: Sebastián Chupáč
# prints text based on truth, positionlly dependent 1:1, # is inserted typo printed with different color
import torch

colors = {
    'green': '\033[92m',
    'red' : '\033[91m',
    'yellow' : '\033[93m',
    'blue' : '\033[0;34m',
    'white' : '\033[0;37m'}

def a_print(text, truth, corr_color, err_color):
    for pair in zip(text, truth):
        if pair[0] == pair[1]: color = colors[corr_color]
        else: 
            color = colors[err_color]
            if pair[1] == '#':
                color = colors['blue']
        print(f'{color}{pair[0]}', end='')
    print(colors['white'])
