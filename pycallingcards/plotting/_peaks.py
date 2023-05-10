from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.axes import Axes
from mudata import MuData

_draw_area_color = Optional[Literal["blue", "red", "green", "pruple"]]


def draw_area(
    chromosome: str,
    start: int,
    end: int,
    extend: int,
    peaks: pd.DataFrame,
    insertions: pd.DataFrame,
    reference: Union[str, pd.DataFrame],
    background: Union[None, pd.DataFrame] = None,
    adata: Optional[AnnData] = None,
    name: Optional[str] = None,
    key: Optional[str] = None,
    insertionkey: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 3),
    plotsize: list = [1, 1, 2],
    bins: Optional[int] = None,
    color: _draw_area_color = "red",
    color_cc: str = None,
    color_peak: str = None,
    color_genes: str = None,
    color_background: str = "lightgray",
    title: Optional[str] = None,
    example_length: int = 10000,
    peak_line: int = 1,
    font_size: int = 1,
    save: Union[bool, str] = False,
):

    """\
    Plot the specific area of the genome. This plot contains three sections.
    The top section is a plot of insertions plot: one dot is one insertion and the height is log(reads + 1).
    The middle section is the distribution plot of insertions.
    If backgound is the input, the colored one would be the experiment inerstions/distribution and the grey one would be the backgound one.
    If backgound is not the input and adata/name/key is provided, the colored one would be the inerstions/distribution for specific group and the grey one would be the whole data.
    The bottom section composes of reference genes and peaks.


    :param chromosome:
        The chromosome plotted.
    :param start:
        The start point of middle area. Usually, it's the start point of a peak.
    :param end:
        The end point of middle area. Usually, it's the end point of a peak.
    :param extend:
        The extend length (bp) to plot.
    :param peaks:
        pd.Dataframe of peaks
    :param insertions:
        pd.Datadrame of qbed
    :param reference:
        `'hg38'`, `'mm10'`, `'sacCer3'` or pd.DataFrame of the reference data.
    :param background:
        pd.DataFrame of qbed or None.
    :param adata:
        Input along with `name` and `key`.
        It will only show the insertions when the `key` of adata is `name`.
    :param name:
        Input along with `adata` and `key`.
        It will only show the insertions when the `key` of adata is `name`.
    :param key:
        Input along with `adata` and `name`.
        It will only show the insertions when the `key` of adata is `name`.
    :param insertionkey:
        Input along with `adata` and `name`.
        It will find the column `insertionkey` of the insertions file.
    :param figsize:
        The size of the figure.
    :param plotsize:
        The relative size of the dot plot, distribution plot and the peak plot.
    :param bins:
        The bins of histogram. It will automatically calculate if None.
    :param color:  `['blue','red','green','pruple']`.
        The color of the plot.
        If `color` is not a valid color, `color_cc`, `color_peak`, and `color_genes` should be utilized.
    :param color_cc:
        The color of qbed insertions. Used only when `color` is not a valid color.
    :param color_peak:
        The color of peaks. Used only when `color` is not a valid color.
    :param color_genes:
        The color of genes. Used only when `color` is not a valid color.
    :param color_background:
        The color of background.
    :param title:
        The title of the plot.
    :param example_length:
        The length of example.
    :param peak_line:
        The total number of peak lines.
    :param font_size:
        The relative font of the words on the plot.
    :param save:
        Could be bool or str indicating the file name It will be saved as.
        If `True`, a default name would be given and the plot would be saved as a png file.


    :example:
    >>> import pycallingcards as cc
    >>> qbed_data = cc.datasets.mousecortex_data(data="qbed")
    >>> peak_data = cc.pp.callpeaks(qbed_data, method = "CCcaller", reference = "mm10", record = True)
    >>> adata_cc = cc.datasets.mousecortex_data(data="CC")
    >>> cc.pl.draw_area("chr12",50102917,50124960,400000,peak_data,qbed_data,"mm10",adata_cc,"Neuron_Excit",'cluster',figsize = (30,6),peak_line = 4,color = "red")
    """

    if color == "blue":
        color_cc = "cyan"
        color_peak = "royalblue"
        color_genes = "skyblue"

    elif color == "red":
        color_cc = "tomato"
        color_peak = "red"
        color_genes = "mistyrose"

    elif color == "green":

        color_cc = "lightgreen"
        color_peak = "palegreen"
        color_genes = "springgreen"

    elif color == "purple":

        color_cc = "magenta"
        color_peak = "darkviolet"
        color_genes = "plum"

    peakschr = peaks[peaks.iloc[:, 0] == chromosome]
    insertionschr = insertions[insertions.iloc[:, 0] == chromosome]

    if type(adata) == AnnData:
        if name != None:
            if key == "Index":
                adata = adata[name, :]
            else:
                adata = adata[adata.obs[key] == name]

        if insertionkey == None:
            insertionschr = insertionschr[
                insertionschr["Barcodes"].isin(adata.obs.index)
            ]
        else:
            insertionschr = insertionschr[
                insertionschr[insertionkey].isin(adata.obs.index)
            ]

    if type(reference) == str:
        if reference == "hg38":
            refdata = pd.read_csv(
                "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.hg38.Sorted.bed",
                delimiter="\t",
                header=None,
            )
        elif reference == "mm10":
            refdata = pd.read_csv(
                "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.mm10.Sorted.bed",
                delimiter="\t",
                header=None,
            )
        elif reference == "sacCer3":
            refdata = pd.read_csv(
                "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.sacCer3.Sorted.bed",
                delimiter="\t",
                header=None,
            )

    elif type(reference) == pd.DataFrame:
        refdata = reference
    else:
        raise ValueError("Not valid reference.")

    refchr = refdata[refdata.iloc[:, 0] == chromosome]

    r1 = refchr[
        (refchr.iloc[:, 2] >= start - extend) & (refchr.iloc[:, 1] <= end + extend)
    ].to_numpy()
    d1 = insertionschr[
        (insertionschr.iloc[:, 2] >= start - extend)
        & (insertionschr.iloc[:, 1] <= end + extend)
    ]
    p1 = peakschr[
        (peakschr.iloc[:, 2] >= start - extend) & (peakschr.iloc[:, 1] <= end + extend)
    ].to_numpy()

    if bins == None:
        bins = int(
            min(plt.rcParams["figure.dpi"] * figsize[0], (end - start + 2 * extend)) / 4
        )

    bins = list(
        range(start - extend, end + extend, int((end - start + 2 * extend) / bins))
    )

    if type(background) == pd.DataFrame:

        backgroundchr = background[background.iloc[:, 0] == chromosome]
        b1 = backgroundchr[
            (backgroundchr.iloc[:, 1] >= start - extend)
            & (backgroundchr.iloc[:, 2] <= end + extend)
        ]

    elif name != None:

        backgroundchr = insertions[insertions.iloc[:, 0] == chromosome]
        b1 = backgroundchr[
            (backgroundchr.iloc[:, 1] >= start - extend)
            & (backgroundchr.iloc[:, 2] <= end + extend)
        ]

    if type(background) == pd.DataFrame:

        figure, axis = plt.subplots(
            5,
            1,
            figsize=figsize,
            gridspec_kw={
                "height_ratios": [
                    plotsize[0] / 2,
                    plotsize[0] / 2,
                    plotsize[1] / 2,
                    plotsize[1] / 2,
                    plotsize[2],
                ]
            },
        )

        axis[0].plot(
            list(d1.iloc[:, 1]),
            list(np.log(d1.iloc[:, 3] + 1)),
            color_cc,
            marker="o",
            linestyle="None",
            markersize=6,
        )
        # axis[0].axis("off")
        axis[0].set_xlim([start - extend, end + extend])
        axis[0].axes.get_xaxis().set_visible(False)
        axis[0].axes.get_yaxis().set_visible(False)
        axis[0].spines.top.set(visible=False)
        axis[0].spines.right.set(visible=False)
        axis[0].spines.left.set(visible=False)
        axis[0].text(
            1,
            0.01,
            "Experiment Insertions",
            ha="left",
            va="bottom",
            transform=axis[0].transAxes,
            size=13 * font_size,
        )

        axis[1].plot(
            list(b1.iloc[:, 1]),
            list(np.log(b1.iloc[:, 3] + 1)),
            color_background,
            marker="o",
            linestyle="None",
            markersize=6,
        )
        # axis[1].axis("off")
        axis[1].set_xlim([start - extend, end + extend])
        axis[1].axes.get_xaxis().set_visible(False)
        axis[1].axes.get_yaxis().set_visible(False)
        axis[1].spines.top.set(visible=False)
        axis[1].spines.right.set(visible=False)
        axis[1].spines.left.set(visible=False)
        axis[1].text(
            1,
            0.01,
            "Backgound Insertions",
            ha="left",
            va="bottom",
            transform=axis[1].transAxes,
            size=13 * font_size,
        )

        counts, binsqbed = np.histogram(np.array(d1.iloc[:, 1]), bins=bins)
        axis[2].hist(binsqbed[:-1], binsqbed, weights=counts, color=color_cc)
        axis[2].set_xlim([start - extend, end + extend])
        # axis[2].axis("off")
        axis[2].axes.get_xaxis().set_visible(False)
        axis[2].axes.get_yaxis().set_visible(False)
        axis[2].spines.top.set(visible=False)
        axis[2].spines.right.set(visible=False)
        axis[2].spines.left.set(visible=False)
        axis[2].text(
            1,
            0.01,
            "Experiment Density",
            ha="left",
            va="bottom",
            transform=axis[2].transAxes,
            size=13 * font_size,
        )

        counts, binsbg = np.histogram(np.array(b1.iloc[:, 1]), bins=bins)
        axis[3].hist(
            binsbg[:-1], binsbg, weights=np.log(counts + 1), color=color_background
        )
        axis[3].set_xlim([start - extend, end + extend])
        # axis[3].axis("off")
        axis[3].axes.get_xaxis().set_visible(False)
        axis[3].axes.get_yaxis().set_visible(False)
        axis[3].spines.top.set(visible=False)
        axis[3].spines.right.set(visible=False)
        axis[3].spines.left.set(visible=False)
        axis[3].text(
            1,
            0.01,
            "Background Density",
            ha="left",
            va="bottom",
            transform=axis[3].transAxes,
            size=13 * font_size,
        )

        pnumber = 0

        for i in range(len(p1)):

            axis[4].plot(
                [p1[i, 1], p1[i, 2]],
                [-1 * (pnumber % peak_line) + 0.15, -1 * (pnumber % peak_line) + 0.15],
                linewidth=10,
                c=color_peak,
            )
            axis[4].text(
                (p1[i, 2] + extend / 40),
                -1 * (pnumber % peak_line) + 0.15,
                "  " + p1[i, 0] + "_" + str(p1[i, 1]) + "_" + str(p1[i, 2]),
                fontsize=14 * font_size,
            )
            pnumber += 1

        for i in range(len(r1)):

            axis[4].plot(
                [max(r1[i, 1], start - extend), min(r1[i, 2], end + extend)],
                [1 + i, 1 + i],
                linewidth=5,
                c=color_genes,
            )
            axis[4].text(
                min(r1[i, 2] + extend / 40, end + extend),
                1 + i,
                " " + r1[i, 3] + ", " + r1[i, 4],
                fontsize=12 * font_size,
            )

            if r1[i, 5] == "-":
                axis[4].annotate(
                    "",
                    xytext=(min(r1[i, 2], end + extend), 1 + i),
                    xy=(max(r1[i, 1], start - extend), 1 + i),
                    xycoords="data",
                    va="center",
                    ha="center",
                    size=20,
                    arrowprops=dict(arrowstyle="simple", color=color_genes),
                )
            elif r1[i, 5] == "+":
                axis[4].annotate(
                    "",
                    xytext=(max(r1[i, 1], start - extend), 1 + i),
                    xy=(min(r1[i, 2], end + extend), 1 + i),
                    xycoords="data",
                    va="center",
                    ha="center",
                    size=20,
                    arrowprops=dict(arrowstyle="simple", color=color_genes),
                )

        axis[4].set_xlim([start - extend, end + extend])
        axis[4].axis("off")

        if example_length != None:
            axis[4].plot(
                [
                    end + extend - example_length - example_length / 5,
                    end + extend - example_length / 5,
                ],
                [-1 - peak_line, -1 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[4].plot(
                [
                    end + extend - example_length - example_length / 5,
                    end + extend - example_length - example_length / 5,
                ],
                [-1 - peak_line, -0.6 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[4].plot(
                [end + extend - example_length / 5, end + extend - example_length / 5],
                [-1 - peak_line, -0.6 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[4].text(
                end + extend,
                -1 - peak_line,
                str(example_length) + "bp",
                fontsize=12 * font_size,
            )

    elif name != None:

        figure, axis = plt.subplots(
            5,
            1,
            figsize=figsize,
            gridspec_kw={
                "height_ratios": [
                    plotsize[0] / 2,
                    plotsize[0] / 2,
                    plotsize[1] / 2,
                    plotsize[1] / 2,
                    plotsize[2],
                ]
            },
        )

        axis[0].plot(
            list(d1.iloc[:, 1]),
            list(np.log(d1.iloc[:, 3] + 1)),
            color_cc,
            marker="o",
            linestyle="None",
            markersize=6,
        )
        # axis[0].axis("off")
        axis[0].set_xlim([start - extend, end + extend])
        axis[0].axes.get_xaxis().set_visible(False)
        axis[0].axes.get_yaxis().set_visible(False)
        axis[0].spines.top.set(visible=False)
        axis[0].spines.right.set(visible=False)
        axis[0].spines.left.set(visible=False)
        axis[0].text(
            1,
            0.01,
            "Experiment Insertions",
            ha="left",
            va="bottom",
            transform=axis[0].transAxes,
            size=13 * font_size,
        )

        axis[1].plot(
            list(b1.iloc[:, 1]),
            list(np.log(b1.iloc[:, 3] + 1)),
            color_background,
            marker="o",
            linestyle="None",
            markersize=6,
        )
        # axis[1].axis("off")
        axis[1].set_xlim([start - extend, end + extend])
        axis[1].set_xlim([start - extend, end + extend])
        axis[1].axes.get_xaxis().set_visible(False)
        axis[1].axes.get_yaxis().set_visible(False)
        axis[1].spines.top.set(visible=False)
        axis[1].spines.right.set(visible=False)
        axis[1].spines.left.set(visible=False)
        axis[1].text(
            1,
            0.01,
            "Backgound Insertions",
            ha="left",
            va="bottom",
            transform=axis[1].transAxes,
            size=13 * font_size,
        )

        counts, binsqbed = np.histogram(np.array(d1.iloc[:, 1]), bins=bins)
        axis[2].hist(binsqbed[:-1], binsqbed, weights=counts, color=color_cc)
        axis[2].set_xlim([start - extend, end + extend])
        # axis[2].axis("off")
        axis[2].axes.get_xaxis().set_visible(False)
        axis[2].axes.get_yaxis().set_visible(False)
        axis[2].spines.top.set(visible=False)
        axis[2].spines.right.set(visible=False)
        axis[2].spines.left.set(visible=False)
        axis[2].text(
            1,
            0.01,
            "Experiment Density",
            ha="left",
            va="bottom",
            transform=axis[2].transAxes,
            size=13 * font_size,
        )

        counts, binsbg = np.histogram(np.array(b1.iloc[:, 1]), bins=bins)
        axis[3].hist(binsbg[:-1], binsbg, weights=counts, color=color_background)
        axis[3].set_xlim([start - extend, end + extend])
        # axis[3].axis("off")
        axis[3].axes.get_xaxis().set_visible(False)
        axis[3].axes.get_yaxis().set_visible(False)
        axis[3].spines.top.set(visible=False)
        axis[3].spines.right.set(visible=False)
        axis[3].spines.left.set(visible=False)
        axis[3].text(
            1,
            0.01,
            "Background Density",
            ha="left",
            va="bottom",
            transform=axis[3].transAxes,
            size=13 * font_size,
        )

        pnumber = 0

        for i in range(len(p1)):

            axis[4].plot(
                [p1[i, 1], p1[i, 2]],
                [-1 * (pnumber % peak_line) + 0.15, -1 * (pnumber % peak_line) + 0.15],
                linewidth=10,
                c=color_peak,
            )
            axis[4].text(
                (p1[i, 2] + extend / 40),
                -1 * (pnumber % peak_line) + 0.15,
                "  " + p1[i, 0] + "_" + str(p1[i, 1]) + "_" + str(p1[i, 2]),
                fontsize=14 * font_size,
            )
            pnumber += 1

        for i in range(len(r1)):

            axis[4].plot(
                [max(r1[i, 1], start - extend), min(r1[i, 2], end + extend)],
                [1 + i, 1 + i],
                linewidth=5,
                c=color_genes,
            )
            axis[4].text(
                min(r1[i, 2] + extend / 40, end + extend),
                1 + i,
                " " + r1[i, 3] + ", " + r1[i, 4],
                fontsize=12 * font_size,
            )

            if r1[i, 5] == "-":
                axis[4].annotate(
                    "",
                    xytext=(min(r1[i, 2], end + extend), 1 + i),
                    xy=(max(r1[i, 1], start - extend), 1 + i),
                    xycoords="data",
                    va="center",
                    ha="center",
                    size=20,
                    arrowprops=dict(arrowstyle="simple", color=color_genes),
                )
            elif r1[i, 5] == "+":
                axis[4].annotate(
                    "",
                    xytext=(max(r1[i, 1], start - extend), 1 + i),
                    xy=(min(r1[i, 2], end + extend), 1 + i),
                    xycoords="data",
                    va="center",
                    ha="center",
                    size=20,
                    arrowprops=dict(arrowstyle="simple", color=color_genes),
                )

        axis[4].set_xlim([start - extend, end + extend])
        axis[4].axis("off")

        if example_length != None:
            axis[4].plot(
                [
                    end + extend - example_length - example_length / 5,
                    end + extend - example_length / 5,
                ],
                [-1 - peak_line, -1 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[4].plot(
                [
                    end + extend - example_length - example_length / 5,
                    end + extend - example_length - example_length / 5,
                ],
                [-1 - peak_line, -0.6 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[4].plot(
                [end + extend - example_length / 5, end + extend - example_length / 5],
                [-1 - peak_line, -0.6 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[4].text(
                end + extend,
                -1 - peak_line,
                str(example_length) + "bp",
                fontsize=12 * font_size,
            )

    else:

        figure, axis = plt.subplots(
            3, 1, figsize=figsize, gridspec_kw={"height_ratios": plotsize}
        )

        axis[0].plot(
            list(d1.iloc[:, 1]),
            list(np.log(d1.iloc[:, 3] + 1)),
            color_cc,
            marker="o",
            linestyle="None",
            markersize=6,
        )
        # axis[0].axis("off")
        axis[0].set_xlim([start - extend, end + extend])
        axis[0].axes.get_xaxis().set_visible(False)
        axis[0].axes.get_yaxis().set_visible(False)
        axis[0].spines.top.set(visible=False)
        axis[0].spines.right.set(visible=False)
        axis[0].spines.left.set(visible=False)
        axis[0].text(
            1,
            0.01,
            "Experiment Insertions",
            ha="left",
            va="bottom",
            transform=axis[0].transAxes,
            size=13 * font_size,
        )

        counts, binsqbed = np.histogram(np.array(d1.iloc[:, 1]), bins=bins)
        axis[1].hist(binsqbed[:-1], binsqbed, weights=counts, color=color_cc)
        axis[1].set_xlim([start - extend, end + extend])
        # axis[1].axis("off")
        axis[1].axes.get_xaxis().set_visible(False)
        axis[1].axes.get_yaxis().set_visible(False)
        axis[1].spines.top.set(visible=False)
        axis[1].spines.right.set(visible=False)
        axis[1].spines.left.set(visible=False)
        axis[1].text(
            1,
            0.01,
            "Experiment Density",
            ha="left",
            va="bottom",
            transform=axis[1].transAxes,
            size=13 * font_size,
        )

        pnumber = 0

        for i in range(len(p1)):

            axis[2].plot(
                [p1[i, 1], p1[i, 2]],
                [-1 * (pnumber % peak_line) + 0.15, -1 * (pnumber % peak_line) + 0.15],
                linewidth=10,
                c=color_peak,
            )
            axis[2].text(
                (p1[i, 2] + extend / 40),
                -1 * (pnumber % peak_line) + 0.15,
                "  " + p1[i, 0] + "_" + str(p1[i, 1]) + "_" + str(p1[i, 2]),
                fontsize=14 * font_size,
            )
            pnumber += 1

        for i in range(len(r1)):

            axis[2].plot(
                [max(r1[i, 1], start - extend), min(r1[i, 2], end + extend)],
                [1 + i, 1 + i],
                linewidth=5,
                c=color_genes,
            )
            axis[2].text(
                min(r1[i, 2] + extend / 40, end + extend),
                1 + i,
                " " + r1[i, 3] + ", " + r1[i, 4],
                fontsize=12 * font_size,
            )

            if r1[i, 5] == "-":
                axis[2].annotate(
                    "",
                    xytext=(min(r1[i, 2], end + extend), 1 + i),
                    xy=(max(r1[i, 1], start - extend), 1 + i),
                    xycoords="data",
                    va="center",
                    ha="center",
                    size=20,
                    arrowprops=dict(arrowstyle="simple", color=color_genes),
                )
            elif r1[i, 5] == "+":
                axis[2].annotate(
                    "",
                    xytext=(max(r1[i, 1], start - extend), 1 + i),
                    xy=(min(r1[i, 2], end + extend), 1 + i),
                    xycoords="data",
                    va="center",
                    ha="center",
                    size=20,
                    arrowprops=dict(arrowstyle="simple", color=color_genes),
                )

        axis[2].set_xlim([start - extend, end + extend])
        axis[2].axis("off")

        if example_length != None:
            axis[2].plot(
                [
                    end + extend - example_length - example_length / 5,
                    end + extend - example_length / 5,
                ],
                [-1 - peak_line, -1 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[2].plot(
                [
                    end + extend - example_length - example_length / 5,
                    end + extend - example_length - example_length / 5,
                ],
                [-1 - peak_line, -0.6 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[2].plot(
                [end + extend - example_length / 5, end + extend - example_length / 5],
                [-1 - peak_line, -0.6 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[2].text(
                end + extend,
                -1 - peak_line,
                str(example_length) + "bp",
                fontsize=12 * font_size,
            )

    if title != None:
        figure.suptitle(title, fontsize=16 * font_size)

    if save != False:
        if save == True:
            save = (
                "draw_area_" + chromosome + "_" + str(start) + "_" + str(end) + ".png"
            )
        figure.savefig(save, bbox_inches="tight")


def _myFuncsorting(e):
    try:
        return int(e[3:])
    except:
        return int(ord(e[3:]))


def whole_peaks(
    peak_data: pd.DataFrame,
    reference="mm10",
    figsize: Tuple[int, int] = (100, 50),
    title: str = "Peaks over chromosomes",
    font_size: int = 10,
    linewidth: float = 4,
    color: str = "black",
    added: int = 10000,
    height_name: str = "Experiment Insertions",
    height_scale: float = 1.01,
    exact: bool = False,
    save: Union[bool, str] = False,
):

    """\
    Plot all the peaks in chromosomes.

    :param peak_data:
        Peak_data file from cc.pp.callpeaks.
    :param reference: `['mm10','hg38','sacCer3',None]`.
        The reference of the data.
    :param figsize:
        The size of the figure.
    :param title:
        The title of the plot.
    :param font_size:
        The font of the words on the plot.
    :param linewidth:
        The linewidth between words on the plot.
    :param color:
        The color of the plot, the same as color in plt.plot.
    :param added:
        Only valid when there is no reference provided. The max length(bp) added to the end of a chromosome shown.
    :param height_name:
        The height of each peak. If `None`, it will all be the same height.
    :param height_scale:
        The relative height for the chromosome section of the tallest peak.
    :param exact:
        Wheather to show the exact number of total bp at the end of chromosome.
    :param save:
        Could be bool or str indicating the file name It will be saved.
        If `True`, a default name will be given and the plot would be saved as a png file.

    :example:
    >>> import pycallingcards as cc
    >>> qbed_data = cc.datasets.mousecortex_data(data="qbed")
    >>> peak_data = cc.pp.callpeaks(qbed_data, method = "CCcaller", reference = "mm10", record = True)
    >>> cc.pl.whole_peaks(peak_data)
    """

    sortlist = list(peak_data["Chr"].unique())
    if reference != "sacCer3":
        sortlist.sort(key=_myFuncsorting)
    else:
        sortlist.sort()
    sortlist = np.array(sortlist)

    if reference == "mm10":

        ref = np.array(
            [
                195471971,
                182113224,
                160039680,
                156508116,
                151834684,
                149736546,
                145441459,
                129401213,
                124595110,
                130694993,
                122082543,
                120129022,
                120421639,
                124902244,
                104043685,
                98207768,
                94987271,
                90702639,
                61431566,
                171031299,
                91744698,
            ]
        )
        if exact:
            ref_d = ref
        else:
            ref_d = np.array(
                [
                    "195.5 M",
                    "182.1 M",
                    "160.0 M",
                    "156.5 M",
                    "151.8 M",
                    "149.7 M",
                    "145.4 M",
                    "129.4 M",
                    "124.6 M",
                    "130.7 M",
                    "122.1 M",
                    "120.1 M",
                    "120.4 M",
                    "124.9 M",
                    "104.0 M",
                    "98.2 M",
                    "95.0 M",
                    "90.7 M",
                    "61.4 M",
                    "171.0 M",
                    "91.7 M",
                ]
            )

        sortlist1 = np.array(
            [
                "chr1",
                "chr2",
                "chr3",
                "chr4",
                "chr5",
                "chr6",
                "chr7",
                "chr8",
                "chr9",
                "chr10",
                "chr11",
                "chr12",
                "chr13",
                "chr14",
                "chr15",
                "chr16",
                "chr17",
                "chr18",
                "chr19",
                "chrX",
                "chrY",
            ]
        )
        ref = ref[np.in1d(sortlist1, sortlist)]
        sortlist = sortlist1[np.in1d(sortlist1, sortlist)]

    elif reference == "hg38":
        ref = np.array(
            [
                248956422,
                242193529,
                198295559,
                190214555,
                181538259,
                170805979,
                159345973,
                145138636,
                138394717,
                133797422,
                135086622,
                133275309,
                114364328,
                107043718,
                101991189,
                90338345,
                83257441,
                80373285,
                58617616,
                64444167,
                46709983,
                50818468,
                156040895,
                57227415,
            ]
        )
        if exact:
            ref_d = ref

        else:
            ref_d = np.array(
                [
                    "249.0 M",
                    "242.2 M",
                    "198.3 M",
                    "190.2 M",
                    "181.5 M",
                    "170.8 M",
                    "159.3 M",
                    "145.1 M",
                    "138.4 M",
                    "133.8 M",
                    "135.1 M",
                    "133.3 M",
                    "114.4 M",
                    "107.0 M",
                    "102.0 M",
                    "90.3 M",
                    "83.3 M",
                    "80.4 M",
                    "58.6 M",
                    "64.4 M",
                    "46.7 M",
                    "50.8 M",
                    "156.0 M",
                    "57.2 M",
                ]
            )
        sortlist1 = np.array(
            [
                "chr1",
                "chr2",
                "chr3",
                "chr4",
                "chr5",
                "chr6",
                "chr7",
                "chr8",
                "chr9",
                "chr10",
                "chr11",
                "chr12",
                "chr13",
                "chr14",
                "chr15",
                "chr16",
                "chr17",
                "chr18",
                "chr19",
                "chr20",
                "chr21",
                "chr22",
                "chrX",
                "chrY",
            ]
        )
        ref = ref[np.in1d(sortlist1, sortlist)]
        sortlist = sortlist1[np.in1d(sortlist1, sortlist)]

    elif reference == "sacCer3":

        ref = np.array(
            [
                230218,
                813184,
                316620,
                1531933,
                576874,
                270161,
                1090940,
                562643,
                439888,
                745751,
                666816,
                1078177,
                924431,
                784333,
                1091291,
                948066,
            ]
        )
        if exact:
            ref_d = ref
        else:
            ref_d = np.array(
                [
                    "230.2 T",
                    "813.2 T",
                    "316.6 T",
                    "1.5 M",
                    "576.9 T",
                    "270.2 T",
                    "1.1 M",
                    "562.6 T",
                    "439.9 T",
                    "745.8 T",
                    "666.8 T",
                    "1.1 M",
                    "924.4 T",
                    "784.3 T",
                    "1.1 M",
                    "948.1 T ",
                ]
            )

        sortlist1 = np.array(
            [
                "chrI",
                "chrII",
                "chrIII",
                "chrIV",
                "chrV",
                "chrVI",
                "chrVII",
                "chrVIII",
                "chrIX",
                "chrX",
                "chrXI",
                "chrXII",
                "chrXIII",
                "chrXIV",
                "chrXV",
                "chrXVI",
            ]
        )
        ref = ref[np.in1d(sortlist1, sortlist)]
        sortlist = sortlist1[np.in1d(sortlist1, sortlist)]

    else:
        ref = []

        for chrom in range(len(sortlist)):
            end = list(peak_data[peak_data["Chr"] == sortlist[chrom]]["End"])
            ref.append(end[-1] + added)

    ref = np.array(ref)

    figure, axis = plt.subplots(len(sortlist), 1, figsize=figsize)
    figure.suptitle(title, fontsize=9 * font_size, y=0.90)

    if height_name != None:
        max_height = peak_data["Experiment Insertions"].max()
    else:
        max_height = 1

    for chrom in range(len(sortlist)):

        start = list(peak_data[peak_data["Chr"] == sortlist[chrom]]["Start"])
        end = list(peak_data[peak_data["Chr"] == sortlist[chrom]]["End"])
        if height_name != None:
            insertions = list(
                peak_data[peak_data["Chr"] == sortlist[chrom]][height_name]
            )
        else:
            insertions = [1] * len(start)

        axis[chrom].plot([0, ref[chrom]], [0, 0], color, linewidth=linewidth)
        axis[chrom].set_xlim([0, ref[chrom]])

        axis[chrom].set_ylim([0, height_scale * np.log(max_height + 1)])
        for peak in range(len(start)):
            axis[chrom].plot(
                [start[peak], start[peak]],
                [0, np.log(insertions[peak] + 1)],
                color,
                linewidth=linewidth,
            )
            axis[chrom].plot(
                [end[peak], end[peak]],
                [0, np.log(insertions[peak] + 1)],
                color,
                linewidth=linewidth,
            )
        axis[chrom].axis("off")

        if exact:
            axis[chrom].text(
                ref[chrom], 0, str(ref_d[chrom]) + "bp", fontsize=5 * font_size
            )
        else:
            axis[chrom].text(
                ref[chrom], 0, str(ref_d[chrom]) + "b", fontsize=5 * font_size
            )

        axis[chrom].text(
            0.001,
            0.9,
            sortlist[chrom],
            ha="left",
            va="top",
            transform=axis[chrom].transAxes,
            fontsize=5 * font_size,
        )

    if save != False:
        if save == True:
            save = "Peaks_over_chromosomes.png"
        figure.savefig(save, bbox_inches="tight")


def draw_area_mu(
    chromosome: str,
    start: int,
    end: int,
    extend: int,
    peaks: pd.DataFrame,
    insertions: pd.DataFrame,
    reference: Union[str, pd.DataFrame],
    background: Union[None, pd.DataFrame] = None,
    mdata: Optional[MuData] = None,
    adata_CC: str = "CC",
    name: Optional[str] = None,
    key: Optional[str] = None,
    insertionkey: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 3),
    plotsize: list = [1, 1, 2],
    bins: Optional[int] = None,
    color: _draw_area_color = "red",
    color_cc: str = None,
    color_peak: str = None,
    color_genes: str = None,
    color_background: str = "lightgray",
    title: Optional[str] = None,
    example_length: int = 10000,
    peak_line: int = 1,
    font_size: int = 1,
    save: Union[bool, str] = False,
):

    """\
    Plot the specific area of the genome (Designed for mudata object).
    Plot the specific area of the genome. This plot contains three sections.
    The top section is a plot of insertions plot: one dot is one insertion and the height is log(reads + 1).
    The middle section is the distribution plot of insertions.
    If backgound is the input, the colored one would be the experiment inerstions/distribution and the grey one would be the backgound one.
    If backgound is not the input and mdata/name/key is provided, the colored one would be the inerstions/distribution for specific group and the grey one would be the whole data.
    The third section composes of reference genes and peaks.
    :param chromosome:
        The chromosome plotted.
    :param start:
        The start point of middle area. Usually, it's the start point of a peak.
    :param end:
        The end point of middle area. Usually, it's the end point of a peak.
    :param extend:
        The extend length (bp) to plot.
    :param peaks:
        pd.Dataframe of peaks
    :param insertions:
        pd.Datadrame of qbed
    :param reference:
        `'hg38'`, `'mm10'`, `'sacCer3'` or pd.DataFrame of the reference data.
    :param background: Default is `None`.
        pd.DataFrame of qbed or None.
    :param mdata: Default is `None`.
        Input along with `name` and `key`.
        It will only show the insertions when the `key` of mdata is `name`.
    :param name: Default is `None`.
        Input along with `mdata` and `key`.
        It will only show the insertions when the `key` of mdata is `name`.
    :param key: Default is `None`.
        Input along with `mdata` and `name`.
        It will only show the insertions when the `key` of mdata is `name`.
    :param insertionkey: Default is `None`('Barcodes').
        Input along with `mdata` and `name`.
        It will find the column `insertionkey` of the insertions file.
    :param figsize: Default is (10, 3).
        The size of the figure.
    :param plotsize: Default is [1,1,2].
        The relative size of the dot plot, distribution plot and the peak plot.
    :param bins:  Default is `None`.
        The bins of histogram. I would automatically calculate if None.
    :param color:  `['blue','red','green','pruple']`. Default is `red`.
        The color of the plot.
        If `color` is not a valid color, `color_cc`, `color_peak`, and `color_genes` should be utilized.
    :param color_cc: Default is `None`.
        The color of qbed insertions. Used only when `color` is not a valid color.
    :param color_peak: Default is `None`.
        The color of peaks. Used only when `color` is not a valid color.
    :param color_genes: Default is `None`.
        The color of genes. Used only when `color` is not a valid color.
    :param color_background:
        The color of background.
    :param title: Default is `None`.
        The title of the plot.
    :param example_length:  Default is 10000.
        The length of example.
    :param peak_line: Default is 1.
        The total number of peak lines.
    :param font_size: Default is `10`.
        The relative font of the words on the plot.
    :param save: Default is `False`.
        Could be bool or str indicating the file name It will be saved.
        If `True`, a default name would be given and the plot would be saved as a png file.
    :example:
    >>> import pycallingcards as cc
    >>> mdata = cc.datasets.mousecortex_data(data="Mudata")
    >>> cc.pl.draw_area_mu("chr3",34638588,34656047,400000,peak_data,qbed_data,"mm10",mdata = mdata,
            name = 'Astrocyte',key ='RNA:cluster',figsize = (30,7),peak_line = 4,color = "blue", title = "chr3_34638588_34656047")
    """

    if color == "blue":
        color_cc = "cyan"
        color_peak = "royalblue"
        color_genes = "skyblue"

    elif color == "red":
        color_cc = "tomato"
        color_peak = "red"
        color_genes = "mistyrose"

    elif color == "green":

        color_cc = "lightgreen"
        color_peak = "palegreen"
        color_genes = "springgreen"

    elif color == "purple":

        color_cc = "magenta"
        color_peak = "darkviolet"
        color_genes = "plum"

    peakschr = peaks[peaks.iloc[:, 0] == chromosome]
    insertionschr = insertions[insertions.iloc[:, 0] == chromosome]

    if type(mdata) == MuData:
        if name != None:
            if key == "Index":
                mdata = mdata[adata_CC][name, :]
            else:
                mdata = mdata[adata_CC][mdata.obs[key] == name]

        if insertionkey == None:
            insertionschr = insertionschr[
                insertionschr["Barcodes"].isin(mdata.obs.index)
            ]
        else:
            insertionschr = insertionschr[
                insertionschr[insertionkey].isin(mdata.obs.index)
            ]

    if type(reference) == str:
        if reference == "hg38":
            refdata = pd.read_csv(
                "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.hg38.Sorted.bed",
                delimiter="\t",
                header=None,
            )
        elif reference == "mm10":
            refdata = pd.read_csv(
                "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.mm10.Sorted.bed",
                delimiter="\t",
                header=None,
            )
        elif reference == "sacCer3":
            refdata = pd.read_csv(
                "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.sacCer3.Sorted.bed",
                delimiter="\t",
                header=None,
            )

    elif type(reference) == pd.DataFrame:
        refdata = reference
    else:
        raise ValueError("Not valid reference.")

    refchr = refdata[refdata.iloc[:, 0] == chromosome]

    r1 = refchr[
        (refchr.iloc[:, 2] >= start - extend) & (refchr.iloc[:, 1] <= end + extend)
    ].to_numpy()
    d1 = insertionschr[
        (insertionschr.iloc[:, 2] >= start - extend)
        & (insertionschr.iloc[:, 1] <= end + extend)
    ]
    p1 = peakschr[
        (peakschr.iloc[:, 2] >= start - extend) & (peakschr.iloc[:, 1] <= end + extend)
    ].to_numpy()

    if bins == None:
        bins = int(
            min(plt.rcParams["figure.dpi"] * figsize[0], (end - start + 2 * extend)) / 4
        )

    bins = list(
        range(start - extend, end + extend, int((end - start + 2 * extend) / bins))
    )

    if type(background) == pd.DataFrame:

        backgroundchr = background[background.iloc[:, 0] == chromosome]
        b1 = backgroundchr[
            (backgroundchr.iloc[:, 1] >= start - extend)
            & (backgroundchr.iloc[:, 2] <= end + extend)
        ]

    elif name != None:

        backgroundchr = insertions[insertions.iloc[:, 0] == chromosome]
        b1 = backgroundchr[
            (backgroundchr.iloc[:, 1] >= start - extend)
            & (backgroundchr.iloc[:, 2] <= end + extend)
        ]

    if type(background) == pd.DataFrame:

        figure, axis = plt.subplots(
            5,
            1,
            figsize=figsize,
            gridspec_kw={
                "height_ratios": [
                    plotsize[0] / 2,
                    plotsize[0] / 2,
                    plotsize[1] / 2,
                    plotsize[1] / 2,
                    plotsize[2],
                ]
            },
        )

        axis[0].plot(
            list(d1.iloc[:, 1]),
            list(np.log(d1.iloc[:, 3] + 1)),
            color_cc,
            marker="o",
            linestyle="None",
            markersize=6,
        )
        # axis[0].axis("off")
        axis[0].set_xlim([start - extend, end + extend])
        axis[0].axes.get_xaxis().set_visible(False)
        axis[0].axes.get_yaxis().set_visible(False)
        axis[0].spines.top.set(visible=False)
        axis[0].spines.right.set(visible=False)
        axis[0].spines.left.set(visible=False)
        axis[0].text(
            1,
            0.01,
            "Experiment Insertions",
            ha="left",
            va="bottom",
            transform=axis[0].transAxes,
            size=13 * font_size,
        )

        axis[1].plot(
            list(b1.iloc[:, 1]),
            list(np.log(b1.iloc[:, 3] + 1)),
            color_background,
            marker="o",
            linestyle="None",
            markersize=6,
        )
        # axis[1].axis("off")
        axis[1].set_xlim([start - extend, end + extend])
        axis[1].axes.get_xaxis().set_visible(False)
        axis[1].axes.get_yaxis().set_visible(False)
        axis[1].spines.top.set(visible=False)
        axis[1].spines.right.set(visible=False)
        axis[1].spines.left.set(visible=False)
        axis[1].text(
            1,
            0.01,
            "Backgound Insertions",
            ha="left",
            va="bottom",
            transform=axis[1].transAxes,
            size=13 * font_size,
        )

        maxheight = max(
            max(list(np.log(d1.iloc[:, 3] + 1))), max(list(np.log(b1.iloc[:, 3] + 1)))
        )
        axis[0].set_ylim([0, maxheight + 1])
        axis[1].set_ylim([0, maxheight + 1])

        counts2, binsqbed = np.histogram(np.array(d1.iloc[:, 1]), bins=bins)
        axis[2].hist(binsqbed[:-1], binsqbed, weights=counts2, color=color_cc)
        axis[2].set_xlim([start - extend, end + extend])
        # axis[2].axis("off")
        axis[2].axes.get_xaxis().set_visible(False)
        axis[2].axes.get_yaxis().set_visible(False)
        axis[2].spines.top.set(visible=False)
        axis[2].spines.right.set(visible=False)
        axis[2].spines.left.set(visible=False)
        axis[2].text(
            1,
            0.01,
            "Experiment Density",
            ha="left",
            va="bottom",
            transform=axis[2].transAxes,
            size=13 * font_size,
        )

        counts3, binsbg = np.histogram(np.array(b1.iloc[:, 1]), bins=bins)
        axis[3].hist(
            binsbg[:-1], binsbg, weights=np.log(counts3 + 1), color=color_background
        )
        axis[3].set_xlim([start - extend, end + extend])
        # axis[3].axis("off")
        axis[3].axes.get_xaxis().set_visible(False)
        axis[3].axes.get_yaxis().set_visible(False)
        axis[3].spines.top.set(visible=False)
        axis[3].spines.right.set(visible=False)
        axis[3].spines.left.set(visible=False)
        axis[3].text(
            1,
            0.01,
            "Background Density",
            ha="left",
            va="bottom",
            transform=axis[3].transAxes,
            size=13 * font_size,
        )

        maxheight = max(counts2.max(), counts3.max())
        axis[2].set_ylim([0, maxheight + 1])
        axis[3].set_ylim([0, maxheight + 1])

        pnumber = 0

        for i in range(len(p1)):

            axis[4].plot(
                [p1[i, 1], p1[i, 2]],
                [-1 * (pnumber % peak_line) + 0.15, -1 * (pnumber % peak_line) + 0.15],
                linewidth=10,
                c=color_peak,
            )
            axis[4].text(
                (p1[i, 2] + extend / 40),
                -1 * (pnumber % peak_line) + 0.15,
                "  " + p1[i, 0] + "_" + str(p1[i, 1]) + "_" + str(p1[i, 2]),
                fontsize=14 * font_size,
            )
            pnumber += 1

        for i in range(len(r1)):

            axis[4].plot(
                [max(r1[i, 1], start - extend), min(r1[i, 2], end + extend)],
                [1 + i, 1 + i],
                linewidth=5,
                c=color_genes,
            )
            axis[4].text(
                min(r1[i, 2] + extend / 40, end + extend),
                1 + i,
                " " + r1[i, 3] + ", " + r1[i, 4],
                fontsize=12 * font_size,
            )

            if r1[i, 5] == "-":
                axis[4].annotate(
                    "",
                    xytext=(min(r1[i, 2], end + extend), 1 + i),
                    xy=(max(r1[i, 1], start - extend), 1 + i),
                    xycoords="data",
                    va="center",
                    ha="center",
                    size=20,
                    arrowprops=dict(arrowstyle="simple", color=color_genes),
                )
            elif r1[i, 5] == "+":
                axis[4].annotate(
                    "",
                    xytext=(max(r1[i, 1], start - extend), 1 + i),
                    xy=(min(r1[i, 2], end + extend), 1 + i),
                    xycoords="data",
                    va="center",
                    ha="center",
                    size=20,
                    arrowprops=dict(arrowstyle="simple", color=color_genes),
                )

        axis[4].set_xlim([start - extend, end + extend])
        axis[4].axis("off")

        if example_length != None:
            axis[4].plot(
                [
                    end + extend - example_length - example_length / 5,
                    end + extend - example_length / 5,
                ],
                [-1 - peak_line, -1 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[4].plot(
                [
                    end + extend - example_length - example_length / 5,
                    end + extend - example_length - example_length / 5,
                ],
                [-1 - peak_line, -0.6 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[4].plot(
                [end + extend - example_length / 5, end + extend - example_length / 5],
                [-1 - peak_line, -0.6 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[4].text(
                end + extend,
                -1 - peak_line,
                str(example_length) + "bp",
                fontsize=12 * font_size,
            )

    elif name != None:

        figure, axis = plt.subplots(
            5,
            1,
            figsize=figsize,
            gridspec_kw={
                "height_ratios": [
                    plotsize[0] / 2,
                    plotsize[0] / 2,
                    plotsize[1] / 2,
                    plotsize[1] / 2,
                    plotsize[2],
                ]
            },
        )

        axis[0].plot(
            list(d1.iloc[:, 1]),
            list(np.log(d1.iloc[:, 3] + 1)),
            color_cc,
            marker="o",
            linestyle="None",
            markersize=6,
        )
        # axis[0].axis("off")
        axis[0].set_xlim([start - extend, end + extend])
        axis[0].axes.get_xaxis().set_visible(False)
        axis[0].axes.get_yaxis().set_visible(False)
        axis[0].spines.top.set(visible=False)
        axis[0].spines.right.set(visible=False)
        axis[0].spines.left.set(visible=False)
        axis[0].text(
            1,
            0.01,
            "Experiment Insertions",
            ha="left",
            va="bottom",
            transform=axis[0].transAxes,
            size=13 * font_size,
        )

        axis[1].plot(
            list(b1.iloc[:, 1]),
            list(np.log(b1.iloc[:, 3] + 1)),
            color_background,
            marker="o",
            linestyle="None",
            markersize=6,
        )
        # axis[1].axis("off")
        axis[1].set_xlim([start - extend, end + extend])
        axis[1].set_xlim([start - extend, end + extend])
        axis[1].axes.get_xaxis().set_visible(False)
        axis[1].axes.get_yaxis().set_visible(False)
        axis[1].spines.top.set(visible=False)
        axis[1].spines.right.set(visible=False)
        axis[1].spines.left.set(visible=False)
        axis[1].text(
            1,
            0.01,
            "Backgound Insertions",
            ha="left",
            va="bottom",
            transform=axis[1].transAxes,
            size=13 * font_size,
        )

        counts, binsqbed = np.histogram(np.array(d1.iloc[:, 1]), bins=bins)
        axis[2].hist(binsqbed[:-1], binsqbed, weights=counts, color=color_cc)
        axis[2].set_xlim([start - extend, end + extend])
        # axis[2].axis("off")
        axis[2].axes.get_xaxis().set_visible(False)
        axis[2].axes.get_yaxis().set_visible(False)
        axis[2].spines.top.set(visible=False)
        axis[2].spines.right.set(visible=False)
        axis[2].spines.left.set(visible=False)
        axis[2].text(
            1,
            0.01,
            "Experiment Density",
            ha="left",
            va="bottom",
            transform=axis[2].transAxes,
            size=13 * font_size,
        )

        counts, binsbg = np.histogram(np.array(b1.iloc[:, 1]), bins=bins)
        axis[3].hist(binsbg[:-1], binsbg, weights=counts, color=color_background)
        axis[3].set_xlim([start - extend, end + extend])
        # axis[3].axis("off")
        axis[3].axes.get_xaxis().set_visible(False)
        axis[3].axes.get_yaxis().set_visible(False)
        axis[3].spines.top.set(visible=False)
        axis[3].spines.right.set(visible=False)
        axis[3].spines.left.set(visible=False)
        axis[3].text(
            1,
            0.01,
            "Background Density",
            ha="left",
            va="bottom",
            transform=axis[3].transAxes,
            size=13 * font_size,
        )

        pnumber = 0

        for i in range(len(p1)):

            axis[4].plot(
                [p1[i, 1], p1[i, 2]],
                [-1 * (pnumber % peak_line) + 0.15, -1 * (pnumber % peak_line) + 0.15],
                linewidth=10,
                c=color_peak,
            )
            axis[4].text(
                (p1[i, 2] + extend / 40),
                -1 * (pnumber % peak_line) + 0.15,
                "  " + p1[i, 0] + "_" + str(p1[i, 1]) + "_" + str(p1[i, 2]),
                fontsize=14 * font_size,
            )
            pnumber += 1

        for i in range(len(r1)):

            axis[4].plot(
                [max(r1[i, 1], start - extend), min(r1[i, 2], end + extend)],
                [1 + i, 1 + i],
                linewidth=5,
                c=color_genes,
            )
            axis[4].text(
                min(r1[i, 2] + extend / 40, end + extend),
                1 + i,
                " " + r1[i, 3] + ", " + r1[i, 4],
                fontsize=12 * font_size,
            )

            if r1[i, 5] == "-":
                axis[4].annotate(
                    "",
                    xytext=(min(r1[i, 2], end + extend), 1 + i),
                    xy=(max(r1[i, 1], start - extend), 1 + i),
                    xycoords="data",
                    va="center",
                    ha="center",
                    size=20,
                    arrowprops=dict(arrowstyle="simple", color=color_genes),
                )
            elif r1[i, 5] == "+":
                axis[4].annotate(
                    "",
                    xytext=(max(r1[i, 1], start - extend), 1 + i),
                    xy=(min(r1[i, 2], end + extend), 1 + i),
                    xycoords="data",
                    va="center",
                    ha="center",
                    size=20,
                    arrowprops=dict(arrowstyle="simple", color=color_genes),
                )

        axis[4].set_xlim([start - extend, end + extend])
        axis[4].axis("off")

        if example_length != None:
            axis[4].plot(
                [
                    end + extend - example_length - example_length / 5,
                    end + extend - example_length / 5,
                ],
                [-1 - peak_line, -1 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[4].plot(
                [
                    end + extend - example_length - example_length / 5,
                    end + extend - example_length - example_length / 5,
                ],
                [-1 - peak_line, -0.6 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[4].plot(
                [end + extend - example_length / 5, end + extend - example_length / 5],
                [-1 - peak_line, -0.6 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[4].text(
                end + extend,
                -1 - peak_line,
                str(example_length) + "bp",
                fontsize=12 * font_size,
            )

    else:

        figure, axis = plt.subplots(
            3, 1, figsize=figsize, gridspec_kw={"height_ratios": plotsize}
        )

        axis[0].plot(
            list(d1.iloc[:, 1]),
            list(np.log(d1.iloc[:, 3] + 1)),
            color_cc,
            marker="o",
            linestyle="None",
            markersize=6,
        )
        # axis[0].axis("off")
        axis[0].set_xlim([start - extend, end + extend])
        axis[0].axes.get_xaxis().set_visible(False)
        axis[0].axes.get_yaxis().set_visible(False)
        axis[0].spines.top.set(visible=False)
        axis[0].spines.right.set(visible=False)
        axis[0].spines.left.set(visible=False)
        axis[0].text(
            1,
            0.01,
            "Experiment Insertions",
            ha="left",
            va="bottom",
            transform=axis[0].transAxes,
            size=13 * font_size,
        )

        counts, binsqbed = np.histogram(np.array(d1.iloc[:, 1]), bins=bins)
        axis[1].hist(binsqbed[:-1], binsqbed, weights=counts, color=color_cc)
        axis[1].set_xlim([start - extend, end + extend])
        # axis[1].axis("off")
        axis[1].axes.get_xaxis().set_visible(False)
        axis[1].axes.get_yaxis().set_visible(False)
        axis[1].spines.top.set(visible=False)
        axis[1].spines.right.set(visible=False)
        axis[1].spines.left.set(visible=False)
        axis[1].text(
            1,
            0.01,
            "Experiment Density",
            ha="left",
            va="bottom",
            transform=axis[1].transAxes,
            size=13 * font_size,
        )

        pnumber = 0

        for i in range(len(p1)):

            axis[2].plot(
                [p1[i, 1], p1[i, 2]],
                [-1 * (pnumber % peak_line) + 0.15, -1 * (pnumber % peak_line) + 0.15],
                linewidth=10,
                c=color_peak,
            )
            axis[2].text(
                (p1[i, 2] + extend / 40),
                -1 * (pnumber % peak_line) + 0.15,
                "  " + p1[i, 0] + "_" + str(p1[i, 1]) + "_" + str(p1[i, 2]),
                fontsize=14 * font_size,
            )
            pnumber += 1

        for i in range(len(r1)):

            axis[2].plot(
                [max(r1[i, 1], start - extend), min(r1[i, 2], end + extend)],
                [1 + i, 1 + i],
                linewidth=5,
                c=color_genes,
            )
            axis[2].text(
                min(r1[i, 2] + extend / 40, end + extend),
                1 + i,
                " " + r1[i, 3] + ", " + r1[i, 4],
                fontsize=12 * font_size,
            )

            if r1[i, 5] == "-":
                axis[2].annotate(
                    "",
                    xytext=(min(r1[i, 2], end + extend), 1 + i),
                    xy=(max(r1[i, 1], start - extend), 1 + i),
                    xycoords="data",
                    va="center",
                    ha="center",
                    size=20,
                    arrowprops=dict(arrowstyle="simple", color=color_genes),
                )
            elif r1[i, 5] == "+":
                axis[2].annotate(
                    "",
                    xytext=(max(r1[i, 1], start - extend), 1 + i),
                    xy=(min(r1[i, 2], end + extend), 1 + i),
                    xycoords="data",
                    va="center",
                    ha="center",
                    size=20,
                    arrowprops=dict(arrowstyle="simple", color=color_genes),
                )

        axis[2].set_xlim([start - extend, end + extend])
        axis[2].axis("off")

        if example_length != None:
            axis[2].plot(
                [
                    end + extend - example_length - example_length / 5,
                    end + extend - example_length / 5,
                ],
                [-1 - peak_line, -1 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[2].plot(
                [
                    end + extend - example_length - example_length / 5,
                    end + extend - example_length - example_length / 5,
                ],
                [-1 - peak_line, -0.6 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[2].plot(
                [end + extend - example_length / 5, end + extend - example_length / 5],
                [-1 - peak_line, -0.6 - peak_line],
                linewidth=2,
                c="k",
            )
            axis[2].text(
                end + extend,
                -1 - peak_line,
                str(example_length) + "bp",
                fontsize=12 * font_size,
            )

    if title != None:
        figure.suptitle(title, fontsize=16 * font_size)

    if save != False:
        if save == True:
            save = (
                "draw_area_" + chromosome + "_" + str(start) + "_" + str(end) + ".png"
            )
        figure.savefig(save, bbox_inches="tight")
