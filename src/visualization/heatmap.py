import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import os

np.random.seed(seed=None)

save_dir = "./reports/figures/heatmap"


def heatmap(probs, keys, models, n_images, ex):
    choice_list = np.random.choice(keys, n_images, replace=True, p=probs)
    choice_list = [list(map(int, c.split(";"))) for c in choice_list]
    choice_list = pd.DataFrame(choice_list, columns=models)
    choice_list[r"$\Gamma^*$"] = (choice_list[models].sum(axis=1) >= 2)*1

    norm = matplotlib.colors.Normalize(-1, 1)
    colors = [[norm(-1.0), "r"], [norm(1.0), "g"]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    choice_list = choice_list.T

    plt.figure(figsize=(10, 2))
    sns.heatmap(choice_list, cmap=cmap, cbar=False, xticklabels=False)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, ex), dpi=300)
    plt.show()


ex = "EX1"
probs = [0.4, 0.2, 0.2, 0.2]
keys = ["1;0;1", "1;1;0", "0;1;1", "0;1;0"]
models = ["DenseNet161", "ResNeXt(32x4d)", "WideResNet50(2)"]
n_images = 1770
heatmap(probs, keys, models, n_images, ex)


ex = "EX2"
probs = [0.2, 0.5, 0.1, 0.1, 0.1]
keys = ["1;0;1", "1;1;1", "0;1;1", "1;1;0", "0;1;0"]
models = ["ResNet34", "ResNet50", "VGG19"]
n_images = 820
heatmap(probs, keys, models, n_images, ex)

ex = "EX3"
probs = [0.4, 0.2, 0.2, 0.2]
keys = ["1;0;1", "1;1;0", "0;1;0", "0;1;1"]
models = ["MobileNetV2", "VGG11", "VGG13"]
n_images = 1770
heatmap(probs, keys, models, n_images, ex)

ex = "EX4"
probs = [0.1, 0.5, 0.1, 0.2, 0.1]
keys = ["1;0;1", "1;1;1", "1;1;0", "0;1;1", "0;1;0"]
models = ["ResNet34", "ResNet101", "VGG16"]
n_images = 820
heatmap(probs, keys, models, n_images, ex)