from global_vars import OUTPUT_DIR

from matplotlib import pyplot as plt
import numpy as np

import pickle
import os


plt.style.use('ggplot')


def print_error(error, title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    x = [i for i in range(len(error))]
    ax.plot(x, error, label='Validation error')
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Error, %')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name))


best_accuracy = 0

with open(OUTPUT_DIR + '/primary_train/val_acc_history.pkl', 'rb') as f:
    accuracy = pickle.load(f)

accuracy_arr = np.zeros(len(accuracy))
for i, val in enumerate(accuracy):
    accuracy_arr[i] = val.item()

error = (1 - accuracy_arr) * 100
name = 'primary_training.png'
title = 'Primary training'
print_error(error, title, name)

best_accuracy = max(best_accuracy, max(accuracy_arr))

with open(OUTPUT_DIR + '/additional_train/val_acc_history.pkl', 'rb') as f:
    accuracy = pickle.load(f)

accuracy_arr = np.zeros(len(accuracy))
for i, val in enumerate(accuracy):
    accuracy_arr[i] = val.item()

error = (1 - accuracy_arr) * 100
name = 'additional_training.png'
title = 'Additional training'
print_error(error, title, name)

best_accuracy = max(best_accuracy, max(accuracy_arr))
print('Best accuracy achieved:', best_accuracy)