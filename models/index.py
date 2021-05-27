from torch.nn import Sequential
from torch.nn.modules.activation import ReLU
from torch_geometric.nn import GCNConv, GATConv


def create_gcn_model(
    n_layers,
    in_channels,
    hidden_dim,
    out_channels
):
    layers = []
    input_size = in_channels
    output_size = hidden_dim[0]

    for l in range(n_layers):
        layers.append(GCNConv(input_size, output_size))
        layers.append(ReLU())

        input_size = output_size
        if l == n_layers - 2:
            output_size = out_channels
        else:
            output_size = hidden_dim[l]

    return Sequential(*layers)


def create_gat_model(
    n_layers,
    in_channels,
    out_channels
):
    return Sequential(GATConv(), ReLU())
