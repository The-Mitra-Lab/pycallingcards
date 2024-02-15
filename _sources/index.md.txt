# Pycallingcards: Calling Cards Data Analysis in Python

Welcome to **Pycallingcards**!

**Pycallingcards** is a package for calling cards data analysis developed and maintained by [Mitra Lab](http://genetics.wustl.edu/rmlab/) at Washington University in St. Louis.

Calling cards is a sequencing technology to assay TF binding which could be done in vitro and in vivo at both bulk and single-cell level. To know more about calling cards technology, please check [Moudgil et al](https://www.sciencedirect.com/science/article/pii/S009286742030814X?via%3Dihub) and [Wang et al](http://genetics.wustl.edu/rmlab/files/2012/09/Calling-Cards-for-DNA-binding.pdf).


Pycallingcards is composed of five different part: datasets, reading (rd), preprocessing (pp), tools (tl) and plotting (pl).
For single-cell calling cards anaysis, Pycallingcards interacts with [Scanpy](https://scanpy.readthedocs.io/en/stable/) and the main structure of Pycallingcards also follows the Scanpy.

- **Datasets** contains four main published datasets for callingcards data.
- **Reading (rd)** includes several functions to read and save qbed/ccf and peak data.
- **Preprocessing (pp)** helps to preprocess data from qbed/ccf data to call peaks, make annotation, make Anndata object and filter peaks.
- **Tools (tl)** calls motif of the peaks, completes differential peaks and pair differential peaks with gene expression,
- **Plotting (pl)** proveides an allround plottting system. It could plot genome areas, link with [WashU Epigenome Browser](http://epigenomegateway.wustl.edu/browser/), show signal comparison with Chip-seq(BigWig file), display differential peaks, demonstrate potenial binding-gene expression relationships.


If you find a model useful for your research, please consider citing the [Pycallingcards manuscript](to be done). Thank you!

```{eval-rst}
.. card:: Installation :octicon:`plug;1em;`
    :link: installation
    :link-type: doc

    Click here to view a brief *Pycallingcards* installation guide and prerequisites.
```

```{eval-rst}
.. card:: Tutorials :octicon:`play;1em;`
    :link: tutorials/index
    :link-type: doc

    End-to-end tutorials showcasing key features in the package.
```

```{eval-rst}
.. card:: User guide :octicon:`info;1em;`
    :link: user_guide/index
    :link-type: doc

    User guide provides some detail information of *Pycallingcards*.
```

```{eval-rst}
.. card:: API reference :octicon:`book;1em;`
    :link: api/index
    :link-type: doc

    Detailed descriptions of *Pycallingcards* API and internals.
```

```{eval-rst}
.. card:: GitHub :octicon:`mark-github;1em;`
    :link: https://github.com/The-Mitra-Lab/pycallingcards

    Ask questions, report bugs, and contribute to *Pycallingcards* at our GitHub repository.
```


*This documentation was heavily inspired and adapted from the [scvi-tools documentation](https://docs.scvi-tools.org/en/stable/). Go check them out!*



```{toctree}
:hidden: true
:maxdepth: 3
:titlesonly: true

installation
tutorials/index
user_guide/index
api/index
release_notes/index
references
GitHub <https://github.com/The-Mitra-Lab/pycallingcards>




```
