from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib import pyplot as plt
from scipy.sparse import lil_matrix
from tqdm import tqdm


def GWAS(
    data: pd.DataFrame,
    chr_name: list = ["Chr_liftover", "Start_liftover", "End_liftover"],
    return_name: str = "GWAS",
) -> pd.DataFrame:

    """\
    Calculate the GWAS result for the peak in the data.
    GWAS data is downloaded from `GWAS Catalog <https://www.ebi.ac.uk/gwas/docs/file-downloads>`__.


    :param data:
        The pd.DataFrame. Should contain either 3 colums for [chr,start,end] or 1 column like 'chr8_64645834_64659215'.
    :param chr_name: Default is `['Chr_liftover','Start_liftover','End_liftover']`.
       If the data contain either 3 columns for [chr,start,end], input the column names as list eg ['Chr_liftover','Start_liftover','End_liftover'].
       If the data contain either 1 column like 'chr8_64645834_64659215', input the column name as list eg ['Peak'].
    :param return_name: Default is `'GWAS'`.
        The name of the column for the result.



    :example:
    >>> import pycallingcards as cc
    >>> adata_cc = cc.datasets.mousecortex_data(data="CC")
    >>> result = cc.tl.pair_peak_gene_bulk(adata_cc,"https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/deseq_MF.csv")
    >>> GWAS_result = cc.tl.GWAS(result, chr_name = ['Peak'])


    """

    GWAS = pd.read_csv(
        "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GWAS.csv"
    )

    result = data.copy()
    if len(chr_name) == 3:
        data = data[chr_name]
    elif len(chr_name) == 1:
        data = data[chr_name].str.split("_", expand=True)

    diseases = []
    for binding in range(len(result)):
        if data.iloc[binding, 0] != "":
            chrom = data.iloc[binding, 0]
            start = data.iloc[binding, 1]
            end = data.iloc[binding, 2]
            gwasbind = list(
                GWAS[
                    (GWAS["CHR_ID"] == str(chrom)[3:])
                    & (GWAS["CHR_POS"] >= int(start))
                    & (GWAS["CHR_POS"] <= int(end))
                ]["DISEASE/TRAIT"].unique()
            )

            if gwasbind != []:
                diseases.append(["; ".join(gwasbind)])
            else:
                diseases.append([""])
        else:
            diseases.append([""])

    finalresult = pd.DataFrame(diseases).set_index(data.index)
    finalresult.columns = [return_name]
    return pd.concat([result, finalresult], axis=1)


def GWAS_adata_sc(
    adata: AnnData,
    number: int = 100,
    bindings: list = ["Chr", "Start", "End"],
    clusters: list = None,
    cluster_name: str = "cluster",
) -> pd.DataFrame:

    """\
    Plot GWAS results for different cell types for single-cell calling cards data. It considers the relative number of insertions in each cell type.
    GWAS data is downloaded from `GWAS Catalog <https://www.ebi.ac.uk/gwas/docs/file-downloads>`__.


    :param adata:
        The anndata object of scCC data.
    :param number: Default is `100`.
       The minimun total number for each SNP.
    :param bindings: Default is ['Chr', 'Start', 'End'].
        The name for binding information.
    :param clusters: Default is None.
        The cluter to consider.
        If None, it will use all the clusters in adata.obs[cluster_name]
    :param cluster_name: Default is `'cluster'`.
        The name of cluster in adata.obs.



    :example:
    >>> import pycallingcards as cc
    >>> adata_cc = cc.datasets.mousecortex_data(data="CC")
    >>> adata_cc = cc.tl.liftover(adata_cc, bindings = ['Chr_liftover', 'Start_liftover', 'End_liftover'])
    >>> cc.tl.GWAS_adata_sc(adata_cc)

    """

    GWAS = pd.read_csv(
        "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GWAS.csv"
    )

    diseases = list(GWAS["DISEASE/TRAIT"].unique())
    disease_dict = {}

    for i in range(len(GWAS["DISEASE/TRAIT"].unique())):
        disease_dict[diseases[i]] = i

    bind_dis = lil_matrix(np.zeros((adata.shape[1], len(diseases))))
    data = adata.var[bindings]

    for binding in tqdm(range(adata.shape[1])):

        chrom = data.iloc[binding, 0]
        start = data.iloc[binding, 1]
        end = data.iloc[binding, 2]
        if str(start) == "None" or str(start) == "nan" or str(start) == "":
            gwasbind = []
        else:
            gwasbind = GWAS[
                (GWAS["CHR_ID"] == str(chrom)[3:])
                & (GWAS["CHR_POS"] >= int(start))
                & (GWAS["CHR_POS"] <= int(end))
            ]

        for dis in range(len(gwasbind)):
            bind_dis[binding, disease_dict[gwasbind.iloc[dis, 3]]] += 1

    cell_dis = adata.X.dot(bind_dis)

    if clusters == None:
        clusters = list(adata.obs[cluster_name].unique())

    final_result = []
    for cluster in clusters:
        final_result.append(
            cell_dis[adata.obs[cluster_name] == cluster].sum(axis=0).tolist()[0]
        )

    final_result = np.array(final_result).T
    total = final_result.sum(axis=1)
    diseases = np.array(diseases)
    diseases = diseases[total > number]
    final_result = final_result[total > number]
    final_result = pd.DataFrame(final_result, index=diseases, columns=clusters)

    return final_result


def GWAS_adata_bulk(
    adata: AnnData, number: int = 100, bindings: list = ["Chr", "Start", "End"]
) -> pd.DataFrame:

    """\
    Plot GWAS results for different cell types for bulk calling cards data. It considers the relative number of insertions in each group.
    GWAS data is downloaded from `GWAS Catalog <https://www.ebi.ac.uk/gwas/docs/file-downloads>`__.


    :param adata:
        The anndata object of bulk CC data.
    :param number: Default is `100`.
       The minimun total number for each SNP.
    :param bindings: Default is ['Chr', 'Start', 'End'].
        The name for binding information.

    :example:
    >>> import pycallingcards as cc
    >>> adata_cc = cc.datasets.mouse_brd4_data(data="CC")
    >>> adata_cc = cc.tl.liftover(adata_cc)
    >>> cc.tl.GWAS_adata_bulk(adata_cc, bindings = ['Chr_liftover', 'Start_liftover', 'End_liftover'])

    """

    GWAS = pd.read_csv(
        "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GWAS.csv"
    )

    diseases = list(GWAS["DISEASE/TRAIT"].unique())
    disease_dict = {}

    for i in range(len(GWAS["DISEASE/TRAIT"].unique())):
        disease_dict[diseases[i]] = i

    bind_dis = lil_matrix(np.zeros((adata.shape[1], len(diseases))))
    data = adata.var[bindings]

    for binding in tqdm(range(adata.shape[1])):

        chrom = data.iloc[binding, 0]
        start = data.iloc[binding, 1]
        end = data.iloc[binding, 2]

        if str(start) == "None" or str(start) == "nan" or str(start) == "":
            gwasbind = []
        else:
            gwasbind = GWAS[
                (GWAS["CHR_ID"] == str(chrom)[3:])
                & (GWAS["CHR_POS"] >= int(start))
                & (GWAS["CHR_POS"] <= int(end))
            ]

        for dis in range(len(gwasbind)):
            bind_dis[binding, disease_dict[gwasbind.iloc[dis, 3]]] += 1

    sample_dis = adata.X.dot(bind_dis)

    final_result = np.array(sample_dis.todense().T)
    total = final_result.sum(axis=1).T
    diseases = np.array(diseases)
    diseases = diseases[total > number]
    final_result = final_result[total > number]

    names = list(adata.obs.index)

    final_result = pd.DataFrame(final_result, index=diseases, columns=names)

    return final_result
