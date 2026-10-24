# Baseline attribution

The GRACE baseline follows **Deep Graph Contrastive Representation Learning**
(Zhu et al., 2020), https://arxiv.org/abs/2006.04131.
The authors' reference repository is https://github.com/CRIPAC-DIG/GRACE.

This compact benchmark retains the local benchmark's two-layer GCN,
PReLU projection head and symmetric two-view InfoNCE objective. It is not
a verbatim distribution of the upstream repository.

Data loaders use PyTorch Geometric and the public Planetoid and Geom-GCN
dataset distributions. Their original citations and terms remain applicable.
No third-party datasets, pretrained models or copied upstream source trees
are bundled in this archive.
