# User guide

Pycallingcards is a python package designed for both single-cell and bulk calling cards data analysis. In the user guide, we provide an overview of the package.

## Single-cell 
:::{figure} /\_static/overview1.png
:align: center
:alt: Overview of tasks
:class: img-fluid
:::


Pycallingcards interfaces with Scanpy which completes preprocessing, clustering and differential expression analysis for scRNA-seq data. Pycallingcards reads the qbed/ccf file and call peaks, make annotation and create a cell by peak anndata object. Differential peak analysis is then followed and pair analysis is completed to find out potential hypothesis for the relationship between transcription factor bindings and gene expression in different clusters. Visualization is designed during every step to understand the data. 


## Bulk
:::{figure} /\_static/overview2.png
:align: center
:alt: Overview of tasks
:class: img-fluid
:::

The basic processes are peak calling, annotation, differential peak analysis and visualization. If bulk RNA-seq for the two conditions are provided, calling cards data would be paired with them to discover potential connections between them.