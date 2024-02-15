## Pycallingcards

[![](https://readthedocs.org/projects/pycallingcards/badge/?version=latest)](https://pycallingcards.readthedocs.io/en/latest/)

<p align="center">
    <img src="https://github.com/The-Mitra-Lab/pycallingcards/assets/64772552/4eb2fa43-1f69-4a35-8b7e-72568e0f95e7", width="350">
</p>

**Pycallingcards** is a package for calling cards data analysis developed and maintained by [Mitra Lab](http://genetics.wustl.edu/rmlab/) at Washington University in St. Louis.

Calling cards is a sequencing technology to assay TF binding which could be done in vitro and in vivo at both bulk and single-cell level. To know more about calling cards technology, please check [Moudgil et al](https://www.sciencedirect.com/science/article/pii/S009286742030814X?via%3Dihub) and [Wang et al](http://genetics.wustl.edu/rmlab/files/2012/09/Calling-Cards-for-DNA-binding.pdf).


Pycallingcards is composed of five different part: datasets, reading (rd), preprocessing (pp), tools (tl) and plotting (pl).
For single-cell calling cards anaysis, Pycallingcards interacts with [Scanpy](https://scanpy.readthedocs.io/en/stable/) and the main structure of Pycallingcards also follows the Scanpy.

- **Datasets** contains four main published datasets for callingcards data.
- **Reading (rd)** includes several functions to read and save qbed and peak data.
- **Preprocessing (pp)** helps to preprocess data from qbed data to call peaks, make annotation, make Anndata object and filter peaks.
- **Tools (tl)** calls motif of the peaks, completes differential peaks and pair differential peaks with gene expression,
- **Plotting (pl)** proveides an allround plottting system. It could plot genome areas, link with [WashU Epigenome Browser](http://epigenomegateway.wustl.edu/browser/), show signal comparison with Chip-seq(BigWig file), display differential peaks, demonstrate potenial binding-gene expression relationships.


## Documentation.

Please check [**here**](https://pycallingcards.readthedocs.io/en/latest/) for detailed documentation.

## Installation

```shell
pip install pycallingcards
```

Install via GitHub

```shell
pip install "git+https://github.com/The-Mitra-Lab/pycallingcards.git" --upgrade
```

# Docker

Link to dockerhub of pycallingcards [**here**](https://hub.docker.com/r/juanruguo/pycallingcards).

Link to tips and codes for using pycallingcards docker  [**here**](https://github.com/The-Mitra-Lab/Pycallingcards_docker).

## Development

Use pre-commit to format code at `git commit`.

```shell
pip install pre-commit
pre-commit install
```

## Cite

```bibtex
@article{guo2024pycallingcards,
  title={Pycallingcards: an integrated environment for visualizing, analyzing, and interpreting calling cards data},
  author={Guo, Juanru and Zhang, Wenjin and Chen, Xuhua and Yen, Allen and Chen, Lucy and Shively, Christian A and Li, Daofeng and Wang, Ting and Dougherty, Joseph D and Mitra, Robi D},
  journal={Bioinformatics},
  pages={btae070},
  year={2024},
  publisher={Oxford University Press}
}
```
