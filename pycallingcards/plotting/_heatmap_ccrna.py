from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib import pyplot as plt


def heatmap(
    adata_cc: AnnData,
    rna: Union[pd.DataFrame, None] = None,
    figsize: Tuple[int, int] = (28, 8),
    font_scale: float = 1,
    cmap: str = "BuPu",
    rnalabels: list = None,
    group: Union[None, str] = None,
    cclabels: list = None,
    log: bool = True,
    title: str = "Relative calling cards and RNA information",
    save: Union[bool, str] = False,
):

    """\
    Plot ranking of peaks.
    :param adata_cc:
        Annotated data matrix.
    :param ran: pd.DataFrame.
        pd.DataFrame for RNA data (genes*sample). Make sure the sample is in the same order as adata_cc.obs
    :param figsize:
        The size of the figure.
    :param font_scale:
        The scale of the font size.
    :param cmap:
        Color map of the plot.
    :param rnalabels:
        The labels of the RNA data to be displayed. Be sure the length of list match the number of samples in RNA file.
    :param group:
        The group information in anndata object if (sample*peak). It will read anndata.obs[group].
    :param cclabels:
        The labels of the CC data to be displayed. Be sure the length of list match the number of samples in CC file.
    :param log:
        Whether to log transform the gene expression or not.
    :param title:
        The title of the plot.
    :param save:
        Could be bool or str indicating the file name it would be saved.
        If `True`, a default name will be given and the plot would be saved as a png file.
    :example:
    >>> import pycallingcards as cc
    >>> adata_cc = cc.datasets.mouse_brd4_data(data = "CC")
    >>> rna = cc.datasets.mouse_brd4_data(data = "RNA")
    >>> cc.pl.heatmap(adata_cc,rna, rnalabels = ["Female1", "Female2", "Female3","Male1", "Male2","Male3"])
    """

    sns.set(rc={"figure.figsize": figsize})
    sns.set(font_scale=font_scale)

    if type(rna) == pd.DataFrame:

        print(
            "Please make sure that the samples in adata_cc and rna are in the same order."
        )

        rna = rna[~rna.index.duplicated(keep="first")]
        rna_list = list(rna.index)
        g = adata_cc.var[["Gene Name1", "Gene Name2"]]

        result_cc = []

        index_result = []

        length = rna.shape[1]

        if group == None:

            ccf = np.array((adata_cc.X.todense()))
            groupnumber = ccf.shape[0]
            groups = list(adata_cc.obs.index)

            for i in range(adata_cc.shape[1]):
                gene1 = g.iloc[i, 0]
                gene2 = g.iloc[i, 1]
                if gene1 in rna_list and gene2 in rna_list:
                    index_result.append([g.index[i], gene1, gene2])
                    result_cc.append(
                        list(ccf[:, i]) + list(rna.loc[gene1]) + list(rna.loc[gene2])
                    )
                elif gene1 in rna_list and gene2 not in rna_list:
                    index_result.append([g.index[i], gene1, gene2])
                    result_cc.append(
                        list(ccf[:, i]) + list(rna.loc[gene1]) + [0] * length
                    )
                elif gene1 not in rna_list and gene2 in rna_list:
                    index_result.append([g.index[i], gene1, gene2])
                    result_cc.append(
                        list(ccf[:, i]) + [0] * length + list(rna.loc[gene2])
                    )
                else:
                    index_result.append([g.index[i], gene1, gene2])
                    result_cc.append(list(ccf[:, i]) + [0] * (2 * length))

        else:

            groups = list(adata_cc.obs[group].unique())
            groupnumber = len(groups)
            gene_number = adata_cc.shape[1]
            ccf = np.array(
                adata_cc[(adata_cc.obs[[group]] == groups[0])[group]].X.sum(axis=0)
            )

            for grou in groups[1:]:
                ccf = np.concatenate(
                    (
                        ccf,
                        np.array(
                            adata_cc[(adata_cc.obs[[group]] == grou)[group]].X.sum(
                                axis=0
                            )
                        ),
                    ),
                    axis=0,
                )

            for i in range(gene_number):
                gene1 = g.iloc[i, 0]
                gene2 = g.iloc[i, 1]
                if gene1 in rna_list and gene2 in rna_list:
                    index_result.append([g.index[i], gene1, gene2])
                    result_cc.append(
                        list(ccf[:, i]) + list(rna.loc[gene1]) + list(rna.loc[gene2])
                    )
                elif gene1 in rna_list and gene2 not in rna_list:
                    index_result.append([g.index[i], gene1, gene2])
                    result_cc.append(
                        list(ccf[:, i]) + list(rna.loc[gene1]) + [0] * length
                    )
                elif gene1 not in rna_list and gene2 in rna_list:
                    index_result.append([g.index[i], gene1, gene2])
                    result_cc.append(
                        list(ccf[:, i]) + [0] * length + list(rna.loc[gene2])
                    )
                else:
                    index_result.append([g.index[i], gene1, gene2])
                    result_cc.append(list(ccf[:, i]) + [0] * (2 * length))

        if cclabels == None:
            cclabels = groups

        print(groups)
        print(["Ref gene1 " + s for s in rnalabels])
        print(["Ref gene2 " + s for s in rnalabels])

        if rnalabels != None:
            yticklabels = (
                cclabels
                + [""]
                + ["Ref gene1 " + s for s in rnalabels]
                + ["Ref gene2 " + s for s in rnalabels]
            )
        else:
            yticklabels = (
                cclabels + [""] + ["Ref gene1"] * length + ["Ref gene2"] * length
            )

        data = np.array(result_cc)
        size = data.shape[0]
        secondnum = groupnumber + length

        data[:, 0:groupnumber] = data[:, 0:groupnumber] / data[:, 0:groupnumber].sum(
            axis=1
        ).reshape((size, 1))
        data = data.T

        temp = data[0:groupnumber, :]
        rank = np.lexsort(temp[::-1])

        data = data[:, rank]
        data = data[:, data[groupnumber:, :].sum(axis=0) > 0]

        if log:
            data[groupnumber:, :] = np.log(data[groupnumber:, :] + 1)

        size = data.shape[1]
        data = np.concatenate(
            (
                data[0:groupnumber, :],
                np.zeros((1, size)),
                data[groupnumber:secondnum, :]
                / data[groupnumber:secondnum, :].max(axis=0).reshape((1, size)),
                data[secondnum:, :]
                / data[secondnum:, :].max(axis=0).reshape((1, size)),
            ),
            axis=0,
        )

        ax = sns.heatmap(data, cmap=cmap, xticklabels=False, yticklabels=yticklabels)

    elif rna == None:

        yticklabels = list(adata_cc.obs.index)

        data = np.log2(np.array(adata_cc.X.todense()) + 1)
        data = data / data.sum(axis=0)
        data = data[:, np.lexsort(data[::-1])]

        ax = sns.heatmap(data, cmap=cmap, xticklabels=False, yticklabels=yticklabels)

    ax.set_title(title, fontsize=16)

    if save != False:
        if save == True:
            save = f"heatmap" + ".png"
        plt.savefig(save, bbox_inches="tight")

    plt.show()

    import matplotlib as mpl

    mpl.rc_file_defaults()
