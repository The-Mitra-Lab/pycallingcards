from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib import pyplot as plt
from scipy.sparse import lil_matrix
from tqdm import tqdm


def GWAS_adata_sc(
    adata: AnnData,
    number: int = 100,
    bindings: list = ["Chr", "Start", "End"],
    clusters: list = None,
    cluster_name: str = "cluster",
    figsize: Tuple[int, int] = (8, 40),
    cmap1: str = "BuPu",
    cmap2: str = None,
    font_scale: float = 1,
    pad: float = 0.01,
    rotation: int = 0,
    title: str = None,
    title_top: str = 0.97,
    title_fontsize: float = 5,
    save: bool = False,
):

    """\
    Plot GWAS results for the different cell types in single-cell calling cards data. It considers the relative number of insertions in each cell type.
    GWAS data is downloaded from `GWAS Catalog <https://www.ebi.ac.uk/gwas/docs/file-downloads>`__.


    :param adata:
        The anndata object of scCC data.
    :param number:
       The minimum total number for each SNP.
    :param bindings:
        The name for binding information.
    :param clusters:
        The cluster to consider.
        If None, it will use all the clusters in adata.obs[cluster_name]/
    :param cluster_name:
        The name of cluster in adata.obs.
    :param figsize:
        The size of the figure.
    :param cmap1:
        The colormap of the first heatmap which is the total count of each SNP.
    :param cmap2:
        The colormap of the second heatmap which is the relative number of insertions in each cell type.
    :param font_scale:
        The font_scale of the words on the plot (except fot title).
    :param pad:
        The distance of the color bar from the bottom of the heatmap.
    :param rotation:
        The angle of the bottom label of the second heatmap
    :param title:
        The title of the plot.
    :param title_top:
        Control the distance of the title from the top of the heatmap.
    :param title_fontsize: Default is title_fontsize.
        The fontsize of the title.
    :param save:
        Could be bool or str indicating the file name it would be saved as.
        If `True`, a default name will be given and the plot would be saved as a png file.


    :example:
    >>> import pycallingcards as cc
    >>> adata_cc = cc.datasets.mousecortex_data(data="CC")
    >>> adata_cc = cc.tl.liftover(adata_cc, bindings = ['Chr_liftover', 'Start_liftover', 'End_liftover'])
    >>> cc.pl.GWAS_adata_sc(adata_cc, bindings = ['Chr_liftover', 'Start_liftover', 'End_liftover'])
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
    total = total[total > number].reshape(final_result.shape[0], 1)

    for cluster in range(len(clusters)):
        final_result[:, cluster] = (
            final_result[:, cluster]
            / adata[adata.obs[cluster_name] == clusters[cluster]].shape[0]
        )

    final_result = final_result / final_result.sum(axis=1).reshape(
        final_result.shape[0], 1
    )
    final_result = pd.DataFrame(
        np.concatenate((total, final_result), axis=1),
        index=diseases,
        columns=["Total"] + clusters,
    )
    final_result = final_result.sort_values(by=list(clusters), ascending=False)

    sns.set(font_scale=font_scale)
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"width_ratios": [1, int(len(clusters))]},
    )
    ax1 = sns.heatmap(final_result[["Total"]], cbar=False, ax=ax1, cmap=cmap1)
    ax1.tick_params(labelright=False, labelleft=True, rotation=rotation)
    ax2 = sns.heatmap(final_result[clusters], cbar=False, ax=ax2, cmap=cmap2)
    ax2.tick_params(labelright=False, labelleft=False, rotation=rotation)

    plt.suptitle(title, fontsize=title_fontsize)

    fig.subplots_adjust(top=title_top)
    fig.colorbar(ax1.get_children()[0], orientation="horizontal", pad=pad)
    fig.colorbar(ax2.get_children()[0], orientation="horizontal", pad=pad)

    if save != False:
        if save == True:
            save = "GWAS_adata_Sc.png"
        fig.savefig(save, bbox_inches="tight")


def GWAS_adata_bulk(
    adata: AnnData,
    number: int = 100,
    bindings: list = ["Chr", "Start", "End"],
    figsize: Tuple[int, int] = (8, 40),
    cmap1: str = "BuPu",
    cmap2: str = None,
    font_scale: float = 1,
    pad: float = 0.01,
    rotation: int = 0,
    title: str = None,
    title_top: str = 0.97,
    title_fontsize: float = 5,
    save: bool = False,
):

    """\
    Plot GWAS results for different cell types in bulk calling cards data. It considers the relative number of insertions in each group.
    GWAS data is downloaded from `GWAS Catalog <https://www.ebi.ac.uk/gwas/docs/file-downloads>`__.


    :param adata:
        The anndata object of bulk CC data.
    :param number:
       The minimum total number for each SNP.
    :param bindings:
        The name for binding information.
    :param figsize:
        The size of the figure.
    :param cmap1:
        The colormap of the first heatmap which is the total count of each SNP.
    :param cmap2:
        The colormap of the second heatmap which is the relative number of insertions in each cell type.
    :param font_scale:
        The font_scale of the words on the plot (except fot title).
    :param pad:
        The distance of the color bar from the bottom of the heatmap.
    :param rotation:
        The angle of the bottom label of the second heatmap
    :param title:
        The title of the plot.
    :param title_top:
        Control the distance of the title from the top of the heatmap.
    :param title_fontsize:
        The fontsize of the title.
    :param save:
        Could be bool or str indicating the file name it would be saved as.
        If `True`, a default name would be given and the plot would be saved as a png file.


    :example:
    >>> import pycallingcards as cc
    >>> adata_cc = cc.datasets.mouse_brd4_data(data="CC")
    >>> adata_cc = cc.tl.liftover(adata_cc, bindings = ['Chr_liftover', 'Start_liftover', 'End_liftover'])
    >>> cc.pl.GWAS_adata_bulk(adata_cc, bindings = ['Chr_liftover', 'Start_liftover', 'End_liftover'])

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
    total = total[total > number].reshape(final_result.shape[0], 1)

    final_result[:, 0] = final_result[:, 0] / adata[0, :].X.sum()
    final_result[:, 1] = final_result[:, 1] / adata[1, :].X.sum()

    final_result = final_result / final_result.sum(axis=1).reshape(
        final_result.shape[0], 1
    )

    names = list(adata.obs.index)

    final_result = pd.DataFrame(
        np.concatenate((total, final_result), axis=1),
        index=diseases,
        columns=["Total"] + names,
    )
    final_result = final_result.sort_values(by=names, ascending=False)

    sns.set(font_scale=font_scale)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, sharey=True, gridspec_kw={"width_ratios": [1, 4]}
    )
    ax1 = sns.heatmap(
        final_result[["Total"]],
        cbar=False,
        ax=ax1,
        cmap=cmap1,
        vmax=int(
            final_result[["Total"]].mean() + np.sqrt(final_result[["Total"]].var())
        ),
    )
    ax1.tick_params(labelright=False, labelleft=True, rotation=rotation)
    ax2 = sns.heatmap(final_result[names], cbar=False, ax=ax2, cmap=cmap2)
    ax2.tick_params(labelright=False, labelleft=False, rotation=rotation)

    plt.suptitle(title, fontsize=title_fontsize)

    fig.subplots_adjust(top=title_top)
    fig.colorbar(ax1.get_children()[0], orientation="horizontal", pad=pad)
    fig.colorbar(ax2.get_children()[0], orientation="horizontal", pad=pad)

    if save != False:
        if save == True:
            save = "GWAS_adata_Sc.png"
        fig.savefig(save, bbox_inches="tight")
