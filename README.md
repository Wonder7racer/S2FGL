# $S^2$FGL: Spatial Spectral Federated Graph Learning

## Abstract

Federated Graph Learning (FGL) combines the privacy-preserving capabilities of federated learning (FL) with the strong graph modeling capability of Graph Neural Networks (GNNs). Current research addresses subgraph-FL only from the structural perspective, neglecting the propagation of graph signals on spatial and spectral domains of the structure. From a spatial perspective, subgraph-FL introduces edge disconnections between clients, leading to disruptions in label signals and a degradation in the class knowledge of the global GNN. From a spectral perspective, spectral heterogeneity causes inconsistencies in signal frequencies across subgraphs, which makes local GNNs overfit the local signal propagation schemes. As a result, spectral client drifts occur, undermining global generalizability. To tackle the challenges, we propose a global knowledge repository to mitigate label signal disruption and a frequency alignment to address spectral client drifts. The combination of **S**patial and **S**pectral strategies forms our framework $S^2$FGL.

<img width="1467" height="648" alt="S2FGL111" src="https://github.com/user-attachments/assets/aa90b51b-c86f-4b01-a28d-1388c0553c15" />

## Citation

``` latex
@improceedings{s2fgl,
  title={S2FGL: Spatial Spectral Federated Graph Learning},
  author={Tan, Zihan and Huang, Suyuan and Wan, Guancheng and Huang, Wenke and Li, He and Ye, Mang},
  booktitle=ICML,
  year={2025}
}
```
