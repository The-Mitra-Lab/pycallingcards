from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.axes import Axes
from mudata import MuData


def _myFuncsorting(e):
    try:
        return int(e.split("_")[0][3:])
    except:
        return int(ord(e.split("_")[0][3:]))


def dotplot_bulk(
    adata_cc: AnnData,
    rna: pd.DataFrame,
    selected_list: list,
    num_list: list,
    xticklabels: list = None,
    rate: float = 50,
    figsize: Tuple[int, int] = (12, 15),
    dotsize: float = 5,
    cmap: str = "Reds",
    title: str = "DE binding & RNA",
    topspace: float = 0.977,
    sort_by_chrom: bool = False,
    save: bool = False,
):

    """\
    Plot ranking of peaks.

    :param adata_cc:
        Anndata of peak.
    :param rna:
        pd.DataFrame of RNA expression.
    :param selected_list:
        A list of peak going to show.
    :param num_list:
        The distribution of samples in rna.
        eg. the first three columns for rna is female and the following two columns is male data, then num_list should be [3,2]
    :param xticklabels: Default is `None`.
        xticklabels for the column. If `None`, it would be the index of adata_cc.obs
    :param rate: Default is `50`.
        Rate to control the dot size.
    :param figsize: Default is (12, 15).
        The size of the figure.
    :param dotsize: Default is 5.
        The relative size of dots.
    :param cmap: Default is `'Reds'`.
        The colormap of the plot.
    :param title: Default is `'DE binding & RNA'`.
        The title of the plot.
    :param topspace: Default is 0.977.
        Parameter to control the title position.
    :param sort_by_chrom: Default is `False`.
        If `True`, it would sort by chr1, chr2, etc.
        sort_by_chrom can not be applied to yeast data.
    :param save: Default is `False`.
        Could be bool or str indicating the file name it would be saved.
        If `True`, a default name would be given and the plot would be saved as png.


    :example:
    >>> import pycallingcards as cc
    >>> adata_cc = cc.datasets.mouse_brd4_data(data = "CC")
    >>> rna = cc.datasets.mouse_brd4_data(data = "RNA")
    >>> cc.pl.dotplot_bulk(adata_cc,rna,
                   selected_list = ['chr1_72823300_72830641', 'chr1_174913218_174921560',
                    'chr4_68545354_68551370', 'chr5_13001870_13004057',
                    'chr5_13124523_13131816', 'chr5_13276480_13283561',
                    'chr5_16764617_16770523', 'chr5_17080322_17085124',
                    'chr7_55291506_55293906', 'chr7_56523379_56528437',
                    'chr8_102778665_102784309', 'chr10_57783900_57788071',
                    'chr11_46057069_46059464', 'chr12_56507583_56514677',
                    'chr14_88460574_88466755', 'chr14_88898126_88902522',
                    'chr15_11743635_11745457', 'chr15_11781285_11785784',
                    'chr15_11823522_11836910', 'chr19_59158212_59161670',
                    'chrY_1241882_1246464', 'chrY_1282449_1287505'] ,
                   num_list = [3,3],figsize = [12,8])
    """

    sns.set_theme()

    if sort_by_chrom:
        selected_list = list(selected_list)
        selected_list.sort(key=_myFuncsorting)

    length = rna.shape[1]
    num_cluster = adata_cc.shape[0]
    df = adata_cc.var[["Gene Name1", "Gene Name2"]]
    rna_list = list(rna.index)

    if xticklabels == None:
        xticklabels = list(adata_cc.obs.index)

    index0 = []
    index1 = []
    index2 = []
    result_cc = []

    for i in selected_list:

        gene1 = df.loc[i][0]
        gene2 = df.loc[i][1]
        if gene1 in rna_list and gene2 in rna_list:
            result_cc.append(
                adata_cc[:, i].X.T.toarray()[0].tolist()
                + list(rna.loc[gene1])
                + list(rna.loc[gene2])
            )
            index0.append(i)
            index1.append(gene1)
            index2.append(gene2)
        elif gene1 in rna_list and gene2 not in rna_list:
            result_cc.append(
                adata_cc[:, i].X.T.toarray()[0].tolist()
                + list(rna.loc[gene1])
                + [0] * length
            )
            index0.append(i)
            index1.append(gene1)
            index2.append(gene2)
        elif gene1 not in rna_list and gene2 in rna_list:
            result_cc.append(
                adata_cc[:, i].X.T.toarray()[0].tolist()
                + [0] * length
                + list(rna.loc[gene2])
            )
            index0.append(i)
            index1.append(gene1)
            index2.append(gene2)

    data = np.log2(np.array(result_cc) + 1)
    selected_length = data.shape[0]

    xticks = list(range(num_cluster))
    yticks = list(range(selected_length - 1, -1, -1))

    fig, ax = plt.subplots(1, 3, figsize=figsize)

    x = []
    y = []
    cs = []
    cs1 = []
    cs2 = []

    for cluster in range(num_cluster):
        cs = cs + list(data[:, cluster])
        x = x + [cluster] * selected_length
        y = y + list(range(selected_length - 1, -1, -1))
        cs1 = cs1 + list(
            np.mean(
                data[
                    :,
                    (num_cluster + sum(num_list[0:cluster])) : (
                        num_cluster + sum(num_list[0 : cluster + 1])
                    ),
                ],
                axis=1,
            )
        )
        cs2 = cs2 + list(
            np.mean(
                data[
                    :,
                    (num_cluster + sum(num_list) + sum(num_list[0:cluster])) : (
                        num_cluster + sum(num_list) + sum(num_list[0 : cluster + 1])
                    ),
                ],
                axis=1,
            )
        )

    cs = np.array(cs)
    cs = dotsize * cs / max(cs)

    cs1 = np.array(cs1)
    cs1 = dotsize * cs1 / max(cs1)

    cs2 = np.array(cs2)
    cs2 = dotsize * cs2 / max(cs2)

    ax[0].scatter(x, y, c=cs, s=rate * np.array(cs), cmap=cmap)
    ax[0].axis(xmin=-1, xmax=num_cluster)
    ax[0].set_xticks(xticks)
    ax[0].set_xticklabels(xticklabels)
    ax[0].set_yticks(yticks)
    ax[0].set_yticklabels(index0)

    ax[1].scatter(x, y, c=cs1, s=rate * np.array(cs1), cmap=cmap)
    ax[1].axis(xmin=-1, xmax=num_cluster)
    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels(xticklabels)
    ax[1].set_yticks(yticks)
    ax[1].set_yticklabels(index1)

    ax[2].scatter(x, y, c=cs2, s=rate * np.array(cs2), cmap=cmap)
    ax[2].axis(xmin=-1, xmax=num_cluster)
    ax[2].set_xticks(xticks)
    ax[2].set_xticklabels(xticklabels)
    ax[2].set_yticks(yticks)
    ax[2].set_yticklabels(index2)

    plt.tight_layout()
    plt.suptitle(title)
    fig.subplots_adjust(top=topspace)

    if save != False:
        if save == True:
            save = f"dotplot" + ".png"
        plt.savefig(save, bbox_inches="tight")

    mpl.rc_file_defaults()


def dotplot_sc(
    adata_cc: AnnData,
    adata: AnnData,
    result: pd.DataFrame,
    rate: float = 50,
    figsize: Tuple[int, int] = (10, 120),
    size: int = 1,
    cmap1: str = "Reds",
    cmap2: str = "BuPu",
    title="DE binding & RNA",
    topspace=0.977,
    save: bool = False,
):

    """\
    Plot ranking of peaks.
    :param adata_cc:
        Anndata of peak.
    :param adata:
        Anndata of RNA.
    :param result:
        pd.DataFrame of result gain from cc.tl.pair_peak_gene_sc with 'Peak' and 'Gene' columns.
    :param rate: Default is `50`.
        Rate to control the dot size.
    :param figsize: Default is (10, 120).
        The size of the figure.
    :param size: Default is 1.
        The size of relative size of text.
    :param cmap: Default is `'Reds'`.
        The colormap of the plot for bindings.
    :param cmap: Default is `'BuPu'`.
        The colormap of the plot for genes.
    :param title: Default is `'DE binding & RNA'`.
        The title of the plot.
    :param topspace: Default is 0.977.
        Parameter to control the title position.
    :param save: Default is `False`.
        Could be bool or str indicating the file name it would be saved.
        If `True`, a default name would be given and the plot would be saved as png.
    :example:
    >>> import pycallingcards as cc
    >>> import scanpy as sc
    >>> adata_cc = cc.datasets.mousecortex_data(data="CC")
    >>> adata = cc.datasets.mousecortex_data(data="RNA")
    >>> qbed_data = cc.datasets.mousecortex_data(data="qbed")
    >>> peak_data = cc.pp.callpeaks(qbed_data, method = "CCcaller", reference = "mm10",  maxbetween = 2000, pvalue_cutoff = 0.01,
                lam_win_size = 1000000,  pseudocounts = 1, record = True)
    >>> peak_annotation = cc.pp.annotation(peak_data, reference = "mm10")
    >>> peak_annotation = cc.pp.combine_annotation(peak_data,peak_annotation)
    >>> sc.tl.rank_genes_groups(adata,'cluster')
    >>> result = cc.tl.pair_peak_gene_sc(adata_cc,adata,peak_annotation)
    >>> cc.pl.dotplot_sc(adata_cc,adata,result)
    """

    sns.set_theme()

    genelist = list(result["Gene"])
    peaklist = list(result["Peak"])

    clusterlist = list(adata_cc.obs["cluster"].unique())
    clusterlist.sort()

    clusterandata = np.array(adata_cc.obs["cluster"])
    clusterehere = {}

    for i in range(len(clusterlist)):
        clusterehere[clusterlist[i]] = np.where(clusterandata == clusterlist[i])

    geneinfor_total = []
    peakinfor_total = []

    for num in range(len(genelist)):
        geneinfor = []
        peakinfor = []
        for cluster in range(len(clusterlist)):
            peakinfor.append(
                adata_cc[clusterehere[clusterlist[cluster]][0], peaklist[num]].X.mean()
            )
            geneinfor.append(
                float(
                    adata[clusterehere[clusterlist[cluster]][0], genelist[num]].X.mean()
                )
            )
        geneinfor_total.append(geneinfor)
        peakinfor_total.append(peakinfor)

    geneinfor_total = np.array(geneinfor_total)
    peakinfor_total = np.log2(np.array(peakinfor_total) + 1)

    geneinfor_total = geneinfor_total - geneinfor_total.min(axis=1).reshape(-1, 1)
    peakinfor_total = peakinfor_total - peakinfor_total.min(axis=1).reshape(-1, 1)

    geneinfor_total = geneinfor_total / geneinfor_total.max(axis=1).reshape(-1, 1)
    peakinfor_total = peakinfor_total / peakinfor_total.max(axis=1).reshape(-1, 1)

    x = list(range(len(clusterlist)))
    rate = 100

    total_num = len(genelist)
    fig, ax = plt.subplots(total_num, 1, figsize=figsize)

    fig.patch.set_visible(False)

    small = [] * len(clusterlist)
    for num in range(total_num):

        for spine in ["top", "right", "left", "bottom"]:
            ax[num].spines[spine].set_visible(False)

        ax[num].scatter(
            x,
            [1] * len(clusterlist),
            c=geneinfor_total[num],
            s=rate * geneinfor_total[num],
            facecolor="blue",
            cmap=cmap1,
        )
        ax[num].scatter(
            x,
            [0.5] * len(clusterlist),
            c=peakinfor_total[num],
            s=rate * peakinfor_total[num],
            facecolor="blue",
            cmap=cmap2,
        )
        ax[num].axis(ymin=0, ymax=1.5)
        ax[num].set_yticks([0.5, 1])
        ax[num].set_yticklabels([peaklist[num], genelist[num]], fontsize=10 * size)
        ax[num].set_xticks(x)
        ax[num].set_xticklabels(small, fontsize=10 * size)

    ax[num].set_xticklabels(clusterlist, rotation=90)
    plt.suptitle(title, size=15 * size)
    fig.subplots_adjust(top=topspace)

    if save != False:
        if save == True:
            save = f"dotplot" + ".png"
        plt.savefig(save, bbox_inches="tight")

    mpl.rc_file_defaults()


def dotplot_sc_mu(
    mdata: MuData,
    adata_cc: str = "CC",
    adata: str = "RNA",
    result: str = "pair",
    cluster_name: str = "RNA:cluster",
    rate: float = 50,
    figsize: Tuple[int, int] = (10, 120),
    size: int = 1,
    cmap1: str = "Reds",
    cmap2: str = "BuPu",
    title="DE binding & RNA",
    topspace=0.977,
    save: bool = False,
):

    """\
    Designed for mudata object.
    Plot ranking of peaks.

    :param mdata:
        mdata for both CC and RNA.
    :param adata_cc: Default is `'CC'`.
        Name for CC data. Anndata is mdata[adata_cc].
    :param adata: Default is `'RNA'`.
        Name for RNA data. Anndata is mdata[adata].
    :param result: Default is `'pair'`.
        pd.DataFrame of result gain from cc.tl.pair_peak_gene_sc with 'Peak' and 'Gene' columns.
    :param cluster_name: Default is `'RNA:cluster'`.
        The name of cluster in adata_cc and adata.
    :param rate: Default is `50`.
        Rate to control the dot size.
    :param figsize: Default is (10, 120).
        The size of the figure.
    :param size: Default is 1.
        The size of relative size of text.
    :param cmap: Default is `'Reds'`.
        The colormap of the plot for bindings.
    :param cmap: Default is `'BuPu'`.
        The colormap of the plot for genes.
    :param title: Default is `'DE binding & RNA'`.
        The title of the plot.
    :param topspace: Default is 0.977.
        Parameter to control the title position.
    :param save: Default is `False`.
        Could be bool or str indicating the file name it would be saved.
        If `True`, a default name would be given and the plot would be saved as png.


    :example:
    >>> import pycallingcards as cc
    >>> mdata = cc.datasets.mousecortex_data(data="Mudata")
    >>> cc.pl.dotplot_sc_mu(mdata)

    """

    sns.set_theme()

    adata_cc = mdata[adata_cc]
    adata = mdata[adata]
    result = adata_cc.uns[result]

    genelist = list(result["Gene"])
    peaklist = list(result["Peak"])

    clusterlist = list(mdata.obs[cluster_name].unique())
    clusterlist.sort()

    clusterandata = np.array(mdata.obs[cluster_name])
    clusterehere = {}

    for i in range(len(clusterlist)):
        clusterehere[clusterlist[i]] = np.where(clusterandata == clusterlist[i])

    geneinfor_total = []
    peakinfor_total = []

    for num in range(len(genelist)):
        geneinfor = []
        peakinfor = []
        for cluster in range(len(clusterlist)):
            peakinfor.append(
                adata_cc[clusterehere[clusterlist[cluster]][0], peaklist[num]].X.mean()
            )
            geneinfor.append(
                float(
                    adata[clusterehere[clusterlist[cluster]][0], genelist[num]].X.mean()
                )
            )
        geneinfor_total.append(geneinfor)
        peakinfor_total.append(peakinfor)

    geneinfor_total = np.array(geneinfor_total)
    peakinfor_total = np.log2(np.array(peakinfor_total) + 1)

    geneinfor_total = geneinfor_total - geneinfor_total.min(axis=1).reshape(-1, 1)
    peakinfor_total = peakinfor_total - peakinfor_total.min(axis=1).reshape(-1, 1)

    geneinfor_total = geneinfor_total / geneinfor_total.max(axis=1).reshape(-1, 1)
    peakinfor_total = peakinfor_total / peakinfor_total.max(axis=1).reshape(-1, 1)

    x = list(range(len(clusterlist)))
    rate = 100

    total_num = len(genelist)
    fig, ax = plt.subplots(total_num, 1, figsize=figsize)

    fig.patch.set_visible(False)

    small = [] * len(clusterlist)
    for num in range(total_num):

        for spine in ["top", "right", "left", "bottom"]:
            ax[num].spines[spine].set_visible(False)

        ax[num].scatter(
            x,
            [1] * len(clusterlist),
            c=geneinfor_total[num],
            s=rate * geneinfor_total[num],
            facecolor="blue",
            cmap=cmap1,
        )
        ax[num].scatter(
            x,
            [0.5] * len(clusterlist),
            c=peakinfor_total[num],
            s=rate * peakinfor_total[num],
            facecolor="blue",
            cmap=cmap2,
        )
        ax[num].axis(ymin=0, ymax=1.5)
        ax[num].set_yticks([0.5, 1])
        ax[num].set_yticklabels([peaklist[num], genelist[num]], fontsize=10 * size)
        ax[num].set_xticks(x)
        ax[num].set_xticklabels(small, fontsize=10 * size)

    ax[num].set_xticklabels(clusterlist, rotation=90)
    plt.suptitle(title, size=15 * size)
    fig.subplots_adjust(top=topspace)

    if save != False:
        if save == True:
            save = f"dotplot" + ".png"
        plt.savefig(save, bbox_inches="tight")

    mpl.rc_file_defaults()
