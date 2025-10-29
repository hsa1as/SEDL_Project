import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
def plot_weights_heatmap(t: torch.Tensor, path: str):
    a = (t.detach() != 0.).to(dtype=torch.int).cpu().numpy()
    cmap = ListedColormap(['black', 'white'])
    plt.imshow(a, aspect='auto', cmap=cmap, interpolation='nearest')
    plt.savefig(path, bbox_inches='tight')
    plt.close()