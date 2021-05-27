# GNN: Linearly growing Receptive Fields
Repository for my 2021 research internship at Dascim Polytechnique
## Benchmark

### Models
- GAT: Graph Attention Neural Networks (torch_geometric.nn.models.GAT)
- GCN: Graph Convolutional Neural Networks (torch_geometric.nn.models.GCN)
- JK-Net: Jumping Knowledge Network (torch_geometric.nn.models.JumpingKnowledge)
- AdaGCN: Adaboost with GCN
- MADReg & AdaEdge

### Datasets
- Short-range graphs
    - Cora
    - CiteSeer
    - Amazon (Product Classification)
    - Reddit (Subreddit prediction)
- Long-range graphs
    - QM9
    - ENZYMES
    - NCI1

## Colab Snipets
- Using .py modules hosted in github:
```
!wget <github_path_to_python_module>
```
- Installing packages once:
```
import os, sys
import os.path as osp
from google.colab import drive
drive.mount('/content/mnt')
nb_path = '/content/notebooks'
os.symlink('/content/mnt/My Drive/Colab Notebooks', nb_path)
sys.path.insert(0, nb_path)
```
```
!pip install --target=$nb_path __package_name__
```