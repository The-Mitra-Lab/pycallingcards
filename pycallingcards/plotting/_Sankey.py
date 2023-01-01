import numpy as np
import pandas as pd
from d3blocks import D3Blocks


def sankey(
    result: pd.DataFrame,
    peakToGene: list = ["Peak", "Gene", "Pvalue_peak"],
    geneToCluster: list = ["Gene", "Cluster", "Score_gene"],
    figsize: list = [1500, 1500],
    path: str = "sankey.html",
):

    """\
    Plot ranking of peaks. This function uses  `d3blocks <https://github.com/d3blocks/d3blocks>`__,
    please install it before using.

    :param result:
        pd.DataFrame of result gain from cc.tl.pair_peak_gene_sc with 'Peak' and 'Gene' columns.
    :param peakToGene: Default is `['Peak','Gene','Pvalue_peak']`.
        The name of column names for source, targt and  weight for the peak to gene part.
    :param geneToCluster: Default is `['Gene','Cluster','Score_gene']`.
        The name of column names for source, targteand  weight for the gene to cluster part.
    :param figsize: Default is `[1500,1500]`.
        The size of the figure.
    :param path: Default is `'sankey.html'`.
       The path to save the file. Please make sure that it ends with html.


    :example:
    >>> import pycallingcards as cc
    >>> import scanpy as sc
    >>> adata_ccf = sc.read("Mouse-Cortex_CCF.h5ad")
    >>> adata = cc.datasets.mousecortex_data(data="RNA")
    >>> ccf_data = cc.datasets.mousecortex_data(data="ccf")
    >>> peak_data = cc.pp.callpeaks(ccf_data, method = "CCcaller", reference = "mm10",  maxbetween = 2000, pvalue_cutoff = 0.01,
                lam_win_size = 1000000,  pseudocounts = 1, record = True)
    >>> peak_annotation = cc.pp.annotation(peak_data, reference = "mm10")
    >>> peak_annotation = cc.pp.combine_annotation(peak_data,peak_annotation)
    >>> sc.tl.rank_genes_groups(adata,'cluster')
    >>> result = cc.tl.pair_peak_gene_sc(adata_ccf,adata,peak_annotation)
    >>> cc.pl.sankey(result)

    """

    print(
        "This function use D3Blocks. Please make sure D3Blocks is installed and result would be saved as html file."
    )

    d3 = D3Blocks(chart="Sankey", frame=True)

    df = pd.DataFrame(
        np.concatenate((np.array(result[peakToGene]), np.array(result[geneToCluster]))),
        columns=["source", "target", "weight"],
    )

    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df["weight"] = df["weight"].astype(str)

    d3.set_node_properties(df)

    d3.set_edge_properties(df, color="target", opacity="target")

    d3.show(filepath=path, figsize=figsize)
