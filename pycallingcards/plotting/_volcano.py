from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib import pyplot as plt


def volcano_plot(
    adata_cc: AnnData,
    figsize: Tuple[int, int] = (10, 6),
    font_size: int = 1,
    pvalue_cutoff: float = 0.01,
    lfc_cutoff: float = 4,
    colorleft: str = "indianred",
    colorright: str = "lightseagreen",
    title: str = "Volcano plot",
    name: str = "fisher_exact",
    label: Union[None, str] = None,
    labelleft: Union[None, List[float]] = None,
    labelright: Union[None, List[float]] = None,
    pvalue_name: str = "pvalues",
    lfc_name: str = "logfoldchanges",
    save: Union[bool, str] = False,
):

    """\
    Plot the volcano plot comparing two sample/cluster.
    Please make sure that alternative equals to 'two-sided' for the differential binding analysis.


    :param figsize:
        The size of the figure.
    :param font_size:
        The font of the words on the plot.
    :param colorleft:
        The color of the dot for the left group.
    :param colorright:
        The color of the dot for the right group.
    :param pvalue_cutoff:
        The pvalue cutoff.
    :param lfc_cutoff:
        The log fold change cutoff.
    :param title:
        The title of the plot.
    :param label:
        The name of label. If None, it will use the top two index.
    :param labelleft:
        The exact place for left label. Default will automatically give it a place on left top of the plot.
    :param labelright:
        The exact place for left label. Default will automatically give it a place on right top of the plots.
    :param pvalue_name:
        The name The name stored in for pvalue/pvalue_adj in adata_cc.uns[name][pvalue_name].
    :param lfc_name:
        The name The name stored in for lfc/lfc_adj in adata_cc.uns[name][lfc_name].
    :param save:
        Could be bool or str indicating the file name it would be saved as.
        If `True`, a default name would be given and the plot would be saved as a png file.

    :example:
    >>> import pycallingcards as cc
    >>> adata_cc = cc.datasets.mouse_brd4_data(data = "CC")
    >>> cc.pl.volcano_plot(adata_cc,figsize = (6,10),labelright = (5,220),labelleft = (-9,220))
    """

    if label == None:
        label = list(adata_cc.obs.index)

    pva = -np.log10(np.array(adata_cc.uns[name][pvalue_name].tolist())[:, 1])
    fc = np.array(adata_cc.uns[name][lfc_name].tolist())[:, 1]

    sns.set_theme()
    figure, axis = plt.subplots(1, 1, figsize=figsize)
    plt.title(title, fontsize=25 * font_size)

    p_cutoff = -np.log10(pvalue_cutoff)

    lim = int(max(np.abs(fc)) + 2)
    maxy = max(pva)

    plt.plot(fc, pva, "gray", marker="o", linestyle="None")
    plt.plot(
        fc[(pva >= p_cutoff) & (fc <= (-1 * lfc_cutoff))],
        pva[(pva >= p_cutoff) & (fc <= (-1 * lfc_cutoff))],
        colorleft,
        marker="o",
        linestyle="None",
    )
    plt.plot(
        fc[(pva > p_cutoff) & (fc > lfc_cutoff)],
        pva[(pva > p_cutoff) & (fc > lfc_cutoff)],
        colorright,
        marker="o",
        linestyle="None",
    )

    if labelleft == None:
        plt.text(-lim * 0.85, 0.85 * maxy, label[0], fontsize=15 * font_size)
    else:
        plt.text(labelleft[0], labelleft[1], label[0], fontsize=15 * font_size)

    if labelright == None:
        plt.text(lim * 0.7, 0.85 * maxy, label[1], fontsize=15 * font_size)
    else:
        plt.text(labelright[0], labelright[1], label[1], fontsize=15 * font_size)

    plt.ylabel("-$Log_10$ P-value", fontsize=15 * font_size)
    plt.xlabel("$Log_2$ Fold Change in binding", fontsize=15 * font_size)
    plt.tick_params(axis="both", which="major", labelsize=13 * font_size)
    plt.xlim([-lim, lim])

    if save != False:
        if save == True:
            save = "Volcano_plot.png"
        plt.savefig(save, bbox_inches="tight")

    mpl.rc_file_defaults()
