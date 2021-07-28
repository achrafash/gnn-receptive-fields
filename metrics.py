import os
import os.path as osp
import pandas as pd
import numpy as np
import torch

from data import load_dataset
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_k_hop_neighborhood(k: int, adj: torch.tensor) -> list:
    """Returns the list of k-hop dense adjacency matrices

    Args:
        - k (int): maximum depth of neighborhood to compute
        - adj [num_nodes, num_nodes]: dense adjacency matrix

    :rtype: List[torch.Tensor]
    """

    k_hop_neb = adj.clone().unsqueeze(0)
    pow_A = adj.clone().to(device)

    for _ in tqdm(range(k - 1)):
        pow_A = torch.mm(adj, pow_A)
        k_neb = torch.where(torch.where(pow_A > 0, 1, 0) - sum(k_hop_neb) > 0, 1, 0)
        k_hop_neb = torch.cat((k_hop_neb, k_neb.unsqueeze(0)), dim=0)

    return k_hop_neb.to(device)


def scale(X: torch.Tensor):
    """Returns the scaled features of the graph

    Args:
        - X [num_nodes, num_features]
    """

    m = X.mean(0)
    s = X.std(0)
    ones = torch.ones(s.shape).to(device)
    s = torch.where(s == 0, ones, s)
    return (X - m) / s


def centroids(X: torch.Tensor, y: torch.Tensor):
    """Returns the label representation by averaging its nodes' features

    Args:
        - X [num_nodes, num_features]: node features
        - y [num_nodes]: labels
    """
    num_classes = y.max().item() + 1

    # group nodes by label
    obs = {}
    for i in range(X.size(0)):
        if obs.get(y[i].item()):
            obs[y[i].item()] += [X[i]]
        else:
            obs[y[i].item()] = [X[i]]

    return torch.stack([sum(obs[c]) / len(obs[c]) for c in range(num_classes)], 0)


def homophily_index(y: torch.Tensor, neb: torch.tensor, mask=None):
    """Computes the homophily index for a given depth

    Args:
        - y [num_nodes]: labels of all nodes
        - neb [num_nodes, num_nodes]: neighbors to consider (can be the adjacency matrix or k-hop neighborhood)
        - mask [num_nodes]: "train", "test" or "val" to consider only these specific neighbors
    """
    num_nodes = y.size(0)

    if mask == None:
        mask = torch.ones(num_nodes)

    masked_neb = neb * mask.to(device)
    yy = y.unsqueeze(1).expand(-1, num_nodes).to(device)

    return (((masked_neb.long() * yy.t()) == yy) * (masked_neb == 1)).sum(1) / ((masked_neb == 1).sum(1) + 1e-8)


def corrcoef(x: torch.Tensor, y: torch.Tensor):
    """Mimics `np.corrcoef`

    Args
        - x: 2D torch.Tensor
        - y: 1D torch.Tensor

    Returns
        c : torch.Tensor
            if x.size() = (5, 100), then return val will be of size (5,5)

    -------
    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref:
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013

    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.corrcoef(x)
        >>> th_corr = corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    if x == [] or x.size(0) == 0:
        return torch.tensor(0.0).to(device)

    # calculate covariance matrix of rows
    mean_x = x.mean(1).unsqueeze(1).expand(x.shape)  # [num_nodes]
    mean_y = y.mean()  # (1)

    c = torch.matmul((x - mean_x), (y - mean_y).t()) / y.size(0)  # covariance

    # normalize covariances
    std_x = torch.std(x, dim=1)
    std_y = torch.std(y)

    c = c / (std_x * std_y)

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c


def graph_correlation(neb: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    """Returns the list of correlations between the barycenter representation of
    labels and the neighbor features.

    Args:
        - neb [num_nodes, num_nodes]: dense adjacency matrix
        - x [num_nodes, num_features]: node features
        - y [num_nodes, num_features]: label representation associated with the target node

    :rtype: list [num_nodes]: correlation (scalar) for every node
    """
    num_nodes = x.size(0)
    return torch.stack([corrcoef(x=x[neb[i] == 1], y=y[i]).abs().mean().to(device) for i in range(num_nodes)], 0)


def confidence(values: torch.Tensor):
    """Returns the 95% confidence interval of the array of values"""
    q = 1.96
    m = values.mean()
    s = values.std()

    return m - q * s / np.sqrt(len(values)), m + q * s / np.sqrt(len(values))


def graph_summary(dataset, K=20):
    graph = dataset[0].to(device)

    all_neb = get_k_hop_neighborhood(K, to_dense_adj(graph.edge_index).squeeze(0))

    x = scale(graph.x)
    scaled_centroids = centroids(x, graph.y)
    y = torch.stack([scaled_centroids[graph.y[i]] for i in range(graph.num_nodes)]).to(device)

    data = pd.DataFrame(
        {
            "k": [],
            "homophily_neighbors": [],
            "homophily_neighborhood": [],
            "correlation_neighbors": [],
            "correlation_neighborhood": [],
        }
    )

    idx = 0
    for k in tqdm(range(1, K + 1)):
        homo_neighborhood = homophily_index(y=graph.y, neb=all_neb[:k].sum(0))
        homo_neighborhood_conf = confidence(homo_neighborhood)

        homo_neighbors = homophily_index(y=graph.y, neb=all_neb[k - 1])
        homo_neighbors_conf = confidence(homo_neighbors)

        corr_neighbors = graph_correlation(all_neb[k - 1], x=x, y=y)
        corr_neighbors_conf = confidence(corr_neighbors)

        corr_neighborhood = graph_correlation(all_neb[:k].sum(0), x=x, y=y)
        corr_neighborhood_conf = confidence(corr_neighborhood)

        data.loc[idx] = {
            "k": k,
            "homophily_neighbors": homo_neighbors_conf[0].item(),
            "homophily_neighborhood": homo_neighborhood_conf[0].item(),
            "correlation_neighbors": corr_neighbors_conf[0].item(),
            "correlation_neighborhood": corr_neighborhood_conf[0].item(),
        }
        idx += 1
        data.loc[idx] = {
            "k": k,
            "homophily_neighbors": homo_neighbors_conf[1].item(),
            "homophily_neighborhood": homo_neighborhood_conf[1].item(),
            "correlation_neighbors": corr_neighbors_conf[1].item(),
            "correlation_neighborhood": corr_neighborhood_conf[1].item(),
        }
        idx += 1

    # _, ax = plt.subplots(1, 2, figsize=(24,12))
    # sns.lineplot(ax=ax[0], x='k', y='value',
    #              hue='variable',
    #              data=pd.melt(data[['k', 'homophily_neighbors', 'homophily_neighborhood']], ['k'])).set(xlabel="depth", ylabel="index", title="Homophily")
    # sns.lineplot(ax=ax[1], x='k', y='value',
    #              hue='variable',
    #              data=pd.melt(data[['k', 'correlation_neighbors', 'correlation_neighborhood']], ['k'])).set(xlabel="depth", ylabel="correlation", title="Correlation")

    return all_neb, data


if __name__ == "__main__":
    path = osp.join(os.getcwd(), "data")
    pubmed = load_dataset(path, "PubMed")
    _, pubmed_summary = graph_summary(pubmed)
    pubmed_summary.to_csv("./pubmed_summary.csv")
