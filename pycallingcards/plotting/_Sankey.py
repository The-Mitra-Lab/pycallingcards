from d3blocks import D3Blocks
import pandas as pd
import numpy as np

def sankey(
    result: pd.DataFrame,
    peakToGene: list[str,str,str] = ['Peak','Gene','Pvalue_peak'],
    geneToCluster: list[str,str,str] = ['Gene','Cluster','Score_gene'],
    figsize: list = [1500,1500],
    path: str = 'sankey.html'
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
    >>> cc.pl.sankey(result)
    
    """

    print('This function use D3Blocks. Please make sure D3Blocks is installed and result would be saved as html file.')

    d3 = D3Blocks(chart='Sankey', frame=True)

    df = pd.DataFrame(np.concatenate((np.array(result[peakToGene]),np.array(result[geneToCluster]))),columns=["source","target","weight"])
 
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df["weight"] = df["weight"].astype(str)

    d3.set_node_properties(df)
    
    d3.set_edge_properties(df, color='target', opacity='target')

    d3.show(filepath=path,figsize = figsize)