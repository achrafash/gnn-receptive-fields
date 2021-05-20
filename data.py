from torch_geometric.datasets import Planetoid, TUDataset, PPI, QM9, Amazon, Reddit2


def load_dataset(path, name, transform=None, params=None):
    '''
    name:
        - 'Cora'
        - 'CiteSeer'
        - 'PubMed'
        - 'PPI'
        - 'ENZYMES'
        - 'QM9'
        - 'Amazon'
    '''

    if name == 'Cora':
        return Planetoid(path, 'Cora', transform=transform)
    elif name == 'CiteSeer':
        return Planetoid(path, 'CiteSeer', transform=transform)
    elif name == 'PubMed':
        return Planetoid(path, 'PubMed', transform=transform)
    elif name == 'PPI':
        return PPI(path, transform=transform)
    elif name == 'ENZYMES':
        return TUDataset(path, 'ENZYMES', transform=transform)
    elif name == 'QM9':
        return QM9(path, transform=transform)
    elif name == 'Amazon':
        if params['goods']:
            goods = params['goods']
        else:
            goods = 'Computers'
        return Amazon(path, name=goods, transform=transform)
    elif name == 'Reddit':
        return Reddit2(path, transform=transform)
