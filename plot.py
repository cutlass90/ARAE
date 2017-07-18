import os
import json
import argparse

import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def to_image(x):
    return x.reshape(28, 28)

def sample(samples, step=0):

    fig = plt.figure(figsize=(4, 4))
    for i, x in enumerate(samples):
        ax = fig.add_subplot(10, 10, i+1) 
        ax.imshow(to_image(x), cmap='gray', aspect='auto', interpolation='bicubic')
        ax.set_axis_off()

    # remove spacings between subplots
    fig.subplots_adjust(hspace=0, wspace=0, left=0, bottom=0, right=1, top=1)
    os.makedirs('pics/', exist_ok=True)
    fig.savefig('pics/sample_{}.png'.format(step))
    plt.close()
    print('Sample saved.')