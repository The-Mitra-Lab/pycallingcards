import copy
from typing import Iterable, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData


def pair_peak_gene_sc(
    adata_cc: AnnData,
    adata: AnnData,
    peak_annotation: pd.DataFrame,
    pvalue_cutoff_cc: float = 0.001,
    pvalue_cutoff_rna: float = 0.001,
    lfc_cutoff: float = 3,
    score_cutoff: float = 3,
    group_cc: str = "binomtest",
    group_adata: str = "rank_genes_groups",
    group_name: str = "cluster",
) -> pd.DataFrame:

    """\
    Pair related peaks and genes.
    Find out significant binding peaks for one cluster and then see whether the annotated genes are also significantly expressed.
    :param adata_cc:
        Anndata for callingcards
    :param adata:
        Anndata for RNA.
    :param peak_annotation:
        peak_annotation get from cc.pp.annotation and cc.pp.combine_annotation
    :param pvalue_cutoff_cc: Default is `0.001`.
        The cut off value for the pvalues of adata_cc.
    :param pvalue_cutoff_rna: Default is `0.001`.
        The cut off value for the pvalues of adata.
    :param score_cutoff: Default is `3`.
        The cut off value for the logfoldchange of adata_cc and the cut of score value for adata.
    :param group_cc: Default is `'binomtest'`.
        The name of target result in adata_cc.uns.
    :param group_adata: Default is `'rank_genes_groups'`.
        The name of target result in adata.uns.
    :param group_name: Default is `'rank_genes_groups'`.
        The name of the cluster in adata_cc.obs.
    :return:
        pd.DataFrame with paired genes and peaks for different groups.
    :example:
    >>> import pycallingcards as cc
    >>> import scanpy as sc
    >>> adata_cc = sc.read("Mouse-Cortex_cc.h5ad")
    >>> adata = cc.datasets.mousecortex_data(data="RNA")
    >>> qbed_data = cc.datasets.mousecortex_data(data="qbed")
    >>> peak_data = cc.pp.callpeaks(qbed_data, method = "CCcaller", reference = "mm10",  maxbetween = 2000, pvalue_cutoff = 0.01,
                lam_win_size = 1000000,  pseudocounts = 1, record = True)
    >>> peak_annotation = cc.pp.annotation(peak_data, reference = "mm10")
    >>> peak_annotation = cc.pp.combine_annotation(peak_data,peak_annotation)
    >>> sc.tl.rank_genes_groups(adata,'cluster')
    >>> cc.tl.pair_peak_gene_sc(adata_cc,adata,peak_annotation)
    """

    pvalue_cc = np.array(adata_cc.uns[group_cc]["pvalues"].tolist())
    name_cc = np.array(adata_cc.uns[group_cc]["names"].tolist())
    lfg_cc = np.array(adata_cc.uns[group_cc]["logfoldchanges"].tolist())
    score_adata = np.array(adata.uns[group_adata]["scores"].tolist())
    pvalue_adata = np.array(adata.uns[group_adata]["pvals"].tolist())
    name_adata = np.array(adata.uns[group_adata]["names"].tolist())
    peak_annotation["name"] = (
        peak_annotation["Chr"].astype(str)
        + "_"
        + peak_annotation["Start"].astype(str)
        + "_"
        + peak_annotation["End"].astype(str)
    )
    rnalists = list(adata.var.index)

    possible_group = list(adata_cc.obs[group_name].unique())
    possible_group.sort()

    result = []
    for cluster in range(pvalue_cc.shape[1]):
        for peak in range(pvalue_cc.shape[0]):
            if (
                pvalue_cc[peak, cluster] <= pvalue_cutoff_cc
                and abs(lfg_cc[peak, cluster]) >= lfc_cutoff
            ):
                rnalist = list(
                    peak_annotation[peak_annotation["name"] == name_cc[peak, cluster]][
                        ["Gene Name1", "Gene Name2"]
                    ].iloc[0]
                )
                for rna in list(set((rnalist))):
                    if rna in rnalists:
                        if (
                            pvalue_adata[:, cluster][name_adata[:, cluster] == rna]
                            <= pvalue_cutoff_rna
                            and abs(
                                score_adata[:, cluster][name_adata[:, cluster] == rna]
                            )
                            >= score_cutoff
                        ):

                            result.append(
                                [
                                    possible_group[cluster],
                                    name_cc[peak, cluster],
                                    lfg_cc[peak, cluster],
                                    pvalue_cc[peak, cluster],
                                    rna,
                                    score_adata[:, cluster][
                                        name_adata[:, cluster] == rna
                                    ][0],
                                    pvalue_adata[:, cluster][
                                        name_adata[:, cluster] == rna
                                    ][0],
                                ]
                            )
            else:
                continue

    return pd.DataFrame(
        result,
        columns=[
            "Cluster",
            "Peak",
            "Logfoldchanges",
            "Pvalue_peak",
            "Gene",
            "Score_gene",
            "Pvalue_gene",
        ],
    )


def pair_peak_gene_bulk(
    adata_cc: AnnData,
    deresult: Union[str, pd.DataFrame],
    pvalue_cutoff_cc: float = 0.001,
    pvalue_cutoff_rna: float = 0.001,
    lfc_cutoff_cc: float = 3,
    lfc_cutoff_rna: float = 3,
    group_cc: str = "fisher_exact",
    name_cc: str = "logfoldchanges",
) -> pd.DataFrame:

    """\
    Pair related peaks and genes.
    Find out significant binding peaks for one cluster and then see whether the annotated genes are also significantly expressed.

    :param adata_cc:
        Anndata for callingcards
    :param deresult:
        result form DEseq2 could be a pandas dataframe or the path to the csv file.
    :param pvalue_cutoff_cc: Default is `0.001`.
        The cut off value for the pvalues of adata_cc.
    :param pvalue_cutoff_rna: Default is `0.001`.
        The cut off value for the pvalues of adata.
    :param lfc_cutoff_cc: Default is `3`.
        The cut off value for the logfoldchange for name_cc of adata_cc.
    :param lfc_cutoff_rna: Default is `3`.
        The cut off value for the logfoldchange of rna.
    :param group_cc: Default is `'fisher_exact'`.
        The name of target result in adata_cc.uns.
    :param name_cc: Default is `''logfoldchanges'`.
        The name of target result in adata.uns[group_cc].


    :return:
        pd.DataFrame with paired genes and peaks for different groups.

    :example:
    >>> import pycallingcards as cc
    >>> adata_cc = cc.datasets.mouse_brd4_data(data="CC")
    >>> cc.tl.pair_peak_gene_bulk(adata_cc,"https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/deseq_MF.csv")
    """

    if type(deresult) == str:
        rnade = pd.read_csv(deresult, index_col=0)

    de_dic = {}
    ind = 0
    de_dic["names"] = np.array(adata_cc.uns[group_cc]["names"].tolist())[:, ind]
    de_dic["pvalues"] = np.array(adata_cc.uns[group_cc]["pvalues"].tolist())[:, ind]
    de_dic[name_cc] = np.array(adata_cc.uns[group_cc][name_cc].tolist())[:, ind]
    de_pd = pd.DataFrame.from_dict(de_dic).set_index("names").loc[adata_cc.var.index]
    de_pd.reset_index(inplace=True)

    result = []
    rna_list = rnade.index
    peaks = adata_cc.var.index

    peak_rna = adata_cc.var[["Gene Name1", "Gene Name2"]]

    for peak in range(len(peaks)):
        if (de_pd.iloc[peak, 1] <= pvalue_cutoff_cc) and (
            abs(de_pd.iloc[peak, 2]) >= lfc_cutoff_cc
        ):

            genes = list(peak_rna.iloc[peak])
            for gene in genes:
                if (
                    (gene in rna_list)
                    and (rnade.loc[gene]["pvalue"] <= pvalue_cutoff_rna)
                    and (abs(rnade.loc[gene]["log2FoldChange"]) >= lfc_cutoff_rna)
                ):
                    result.append(
                        [
                            peaks[peak],
                            de_pd.iloc[peak, 2],
                            de_pd.iloc[peak, 1],
                            gene,
                            rnade.loc[gene]["log2FoldChange"],
                            rnade.loc[gene]["pvalue"],
                        ]
                    )

    return pd.DataFrame(
        result,
        columns=[
            "Peak",
            name_cc + "_peak",
            "Pvalue_peak",
            "Gene",
            "Score_gene",
            "Pvalue_peak",
        ],
    )


def pair_peak_gene_sc_mu(
    mdata: MuData,
    adata_cc: str = "CC",
    adata: str = "RNA",
    peak_annotation: Optional[pd.DataFrame] = None,
    pvalue_cutoff_cc: float = 0.001,
    pvalue_cutoff_rna: float = 0.001,
    lfc_cutoff: float = 3,
    score_cutoff: float = 3,
    group_cc: str = "binomtest",
    group_adata: str = "rank_genes_groups",
    group_name: str = "RNA:cluster",
    save_name: str = "pair",
) -> MuData:

    """\
    Pair related peaks and genes. Designed for mudata object.
    Find out significant binding peaks for one cluster and then see whether the annotated genes are also significantly expressed.

    :param mdata:
        mdata for both CC and RNA.
    :param adata_cc:
        Name for CC data. Anndata is mdata[adata_cc].
    :param adata:
        Name for RNA data. Anndata is mdata[adata].
    :param peak_annotation:
        peak_annotation get from cc.pp.annotation and cc.pp.combine_annotation
    :param pvalue_cutoff_cc: Default is `0.001`.
        The cut off value for the pvalues of adata_cc.
    :param pvalue_cutoff_rna: Default is `0.001`.
        The cut off value for the pvalues of adata.
    :param score_cutoff: Default is `3`.
        The cut off value for the logfoldchange of adata_cc and the cut of score value for adata.
    :param group_cc: Default is `'binomtest'`.
        The name of target result in adata_cc.uns.
    :param group_adata: Default is `'rank_genes_groups'`.
        The name of target result in adata.uns.
    :param group_name: Default is `'rank_genes_groups'`.
        The name of the cluster in mdata.obs.


    :example:
    >>> import pycallingcards as cc
    >>> mdata = cc.datasets.mousecortex_data(data="Mudata")
    >>> cc.tl.pair_peak_gene_sc_mu(mdata, pvalue_cutoff_cc = 0.001, pvalue_cutoff_rna = 0.001, lfc_cutoff = 3, score_cutoff = 3)

    """

    if type(peak_annotation) != pd.DataFrame:
        peak_annotation = copy.copy(mdata[adata_cc].var)

    CC = adata_cc
    adata_cc = mdata[adata_cc]
    adata = mdata[adata]

    pvalue_cc = np.array(adata_cc.uns[group_cc]["pvalues"].tolist())
    name_cc = np.array(adata_cc.uns[group_cc]["names"].tolist())
    lfg_cc = np.array(adata_cc.uns[group_cc]["logfoldchanges"].tolist())
    score_adata = np.array(adata.uns[group_adata]["scores"].tolist())
    pvalue_adata = np.array(adata.uns[group_adata]["pvals"].tolist())
    name_adata = np.array(adata.uns[group_adata]["names"].tolist())
    peak_annotation["name"] = (
        peak_annotation["Chr"].astype(str)
        + "_"
        + peak_annotation["Start"].astype(str)
        + "_"
        + peak_annotation["End"].astype(str)
    )
    rnalists = list(adata.var.index)

    possible_group = list(mdata.obs[group_name].unique())
    possible_group.sort()

    result = []
    for cluster in range(pvalue_cc.shape[1]):
        for peak in range(pvalue_cc.shape[0]):
            if (
                pvalue_cc[peak, cluster] <= pvalue_cutoff_cc
                and abs(lfg_cc[peak, cluster]) >= lfc_cutoff
            ):
                rnalist = list(
                    peak_annotation[peak_annotation["name"] == name_cc[peak, cluster]][
                        ["Gene Name1", "Gene Name2"]
                    ].iloc[0]
                )
                for rna in list(set((rnalist))):
                    if rna in rnalists:
                        if (
                            pvalue_adata[:, cluster][name_adata[:, cluster] == rna]
                            <= pvalue_cutoff_rna
                            and abs(
                                score_adata[:, cluster][name_adata[:, cluster] == rna]
                            )
                            >= score_cutoff
                        ):

                            result.append(
                                [
                                    possible_group[cluster],
                                    name_cc[peak, cluster],
                                    lfg_cc[peak, cluster],
                                    pvalue_cc[peak, cluster],
                                    rna,
                                    score_adata[:, cluster][
                                        name_adata[:, cluster] == rna
                                    ][0],
                                    pvalue_adata[:, cluster][
                                        name_adata[:, cluster] == rna
                                    ][0],
                                ]
                            )
            else:
                continue

    mdata[CC].uns[save_name] = pd.DataFrame(
        result,
        columns=[
            "Cluster",
            "Peak",
            "Logfoldchanges",
            "Pvalue_peak",
            "Gene",
            "Score_gene",
            "Pvalue_gene",
        ],
    )

    return mdata
