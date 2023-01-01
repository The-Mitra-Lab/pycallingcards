# Tutorials

To get started, we recommend trying following along with our tutorials.


:::{note}
For questions about using Pycallingcards, or broader questions about modeling data, please use our raise commit.
:::



### Single-cell calling cards data without background

```{toctree}
:maxdepth: 1

notebooks/Mouse_cortex_Example
```

<span style=" font-size: 0.8em;"> This is a brd4 single-cell(sc) callingcards dataset in mouse cortex from [Moudgil et al., Cell. (2020)](https://doi.org/10.1016/j.cell.2020.06.037) and could be downloaded from [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148448).

<span style=" font-size: 0.8em;"> In this tutorial, we will call peaks, make annotation, do differential peak analysis, pair peaks with genes.



### Single-cell calling cards data with background

```{toctree}
:maxdepth: 1

notebooks/K562HCT116_SP1

```

<span style=" font-size: 0.8em;"> In this data, we test transcription factor SP1 bindings in celline K562 and HCT116 by single-cell(sc) calling cards techenology. The data is from [Moudgil et al., Cell. (2020)](https://doi.org/10.1016/j.cell.2020.06.037) and could be download from [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148448).

<span style=" font-size: 0.8em;"> In this tutorial, we will call peaks with backgound, make annotation, compare peaks with Chip-seq reference data and do differential peak analysis.


### Bulk calling cards data without background

```{toctree}
:maxdepth: 1

notebooks/Brd4_bulk
```

<span style=" font-size: 0.8em;"> This is a brd4 bulk callingcards dataset in mouse cortex from [Kfoury et al., PNAS. (2021)](https://www.pnas.org/doi/10.1073/pnas.2017148118) and could be downloaded from [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156821).

<span style=" font-size: 0.8em;"> In this tutorial, we will call peaks, make annotation, do differential peak analysis, pair bindings with gene expression.

### Bulk calling cards data with background

```{toctree}
:maxdepth: 1

notebooks/SP1_bulk

```

<span style=" font-size: 0.8em;"> This is a transcription factor SP1 binding bulk calling cards data in cre-driver mouseline and bulk brd4 data is also sequenced as backgound. This data contain two time points: day 10(P10) and day 28(P28). The data are from [Cammack et al., PNAS. (2020)](https://www.pnas.org/doi/10.1073/pnas.1918241117) and could be downloaded from [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128493).


<span style=" font-size: 0.8em;"> In this tutorial, we will call peaks, make annotation, do differential peak analysis.
