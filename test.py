import numpy as np
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
def plot_hist():
    labels_name = ['PM3', 'Rel. Gen.', 'Méth. Math.', 'Phy stat', 'MQ2', 'Ato', 'Thermo', 'Ondes élec.',
                   'Optique', 'MQ1', 'Ondes']
    target = np.array([80,93,127,64,72,28,40,37,35,39,36])
    dictionary = dict(zip(labels_name, target))
    sorted_dict = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))
    labels_name, target = zip(*sorted_dict.items())
    freq_series = pd.Series(target)
    my_colors = [(x / 26.0, x / 40.0, 0.75) for x in range(25)]
    plt.figure(figsize=(12, 8))
    ax = freq_series.plot(kind="bar", color=my_colors)
    ax.set_title("Nombre de pages de maths de devoirs produites par cours")
    ax.set_xlabel("Cours")
    ax.set_ylabel("Pages de math")
    ax.set_xticklabels(labels_name)
    rects = ax.patches

    # Make some labels.
    labels = [ '%.0f' % elem for elem in target]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 0.1, label, ha="center", va="bottom"
        )

    plt.show()

plot_hist()