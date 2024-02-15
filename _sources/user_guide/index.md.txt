# User guide
:::{figure} /\_static/overview0.png
:align: center
:alt: Overview of tasks
:class: img-fluid
:::

Pycallingcards is a python package designed for both single-cell and bulk calling cards data analysis. In the user guide, we provide an overview of the package.

:::{figure} /\_static/overview1.png
:align: center
:alt: Overview of tasks
:class: img-fluid
:::

## Single-cell

Pycallingcards interfaces with Scanpy which completes preprocessing, clustering and differential expression analysis for scRNA-seq data. Pycallingcards reads the qbed file and call peaks, make annotation and create a cell by peak anndata object. Two Anndata objects can combine into one Mudata object. Differential peak analysis is then followed and pair analysis is completed to find out potential hypothesis for the relationship between transcription factor bindings and gene expression in different clusters. Visualization is designed during every step to understand the data.


## Bulk

The basic processes are peak calling, annotation, differential peak analysis and visualization. If bulk RNA-seq for the two conditions are provided, calling cards data would be paired with them to discover potential connections between them.
