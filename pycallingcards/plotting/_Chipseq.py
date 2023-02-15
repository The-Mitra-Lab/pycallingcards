from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
import pyBigWig
import tqdm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

_signaltype = Optional[Literal["mean", "median", "min", "max", "sum", "std"]]


def calculate_signal(
    peak_data: pd.DataFrame,
    chipseq_signal: str,
    before: int = 10000,
    after: int = 10000,
    signaltype: _signaltype = "mean",
    nbins: int = 100,
):

    """\
    Calculate the matrix of Chip-seq signal in each peak. This function uses  `pyBigWig <https://www.google.com/search?q=pyBigWig&rlz=1C5CHFA_enUS1018US1018&oq=pyBigWig&aqs=chrome.0.69i59j35i39j0i512j0i30l3j69i60l2.220j0j7&sourceid=chrome&ie=UTF-8>`__,
    please install it before using.

    :param peak_data:
        Peak_data file from cc.pp.callpeaks.
    :param chipseq_signal:
        Chipseq signal file should be bigWig/bigBed file.
    :param before:
        The length(bp) calculated before the middle point of the peak.
    :param after:
        The length(bp) calculated after the middle point of the peak.
    :param signaltype: `['mean', 'median', 'min', 'max', 'sum' ,'std']`.
        Define the type of statistic to be used over the bin size range.
    :param nbins:
        The number of bins calculated.


    :example:
    >>> import pycallingcards as cc
    >>> exp_ccf = cc.datasets.SP1_K562HCT116_data(data = 'HCT116_SP1_ccf')
    >>> bg_ccf = cc.datasets.SP1_K562HCT116_data(data = 'HCT116_brd4_ccf')
    >>> peak_data = cc.pp.callpeaks(exp_ccf, bg_ccf, method = 'ccf_tools', reference = 'hg38', pvalue_cutoffbg = 0.0001, window_size = 2000, step_size = 500,
              pvalue_cutoffTTAA = 0.0000001, lam_win_size = None)
    >>> # If "https://www.encodeproject.org/files/ENCFF587ZMX/@@download/ENCFF587ZMX.bigWig" could not work, please download it and read the local path.
    >>> mtx = cc.pl.calculate_signal(peak_data,"https://www.encodeproject.org/files/ENCFF587ZMX/@@download/ENCFF587ZMX.bigWig")



    """

    bw = pyBigWig.open(chipseq_signal)

    chrom = list(peak_data["Chr"])
    start = list(peak_data["Start"])
    end = list(peak_data["End"])

    signalmtx = np.zeros((len(peak_data), nbins))

    for i in tqdm.tqdm(range(len(peak_data))):
        mid = int((start[i] + end[i]) / 2)
        inta = mid - before
        intb = mid + after
        signalmtx[i, :] = np.array(
            bw.stats(chrom[i], inta, intb, type=signaltype, nBins=nbins)
        )

    return signalmtx


def signal_plot(
    signalmtx: np.array,
    fill_between: bool = True,
    alpha: float = 0.05,
    before: int = 10000,
    after: int = 10000,
    nbins: int = 100,
    figsize: Tuple[int, int] = (8, 6),
    fontsize: int = 10,
    color: str = "red",
    textbelow: float = 0,
    title: str = "Log2(FC) Chip-seq Signal",
    bottom: float = 0.05,
    save: bool = False,
):

    """\
    Plot the Chip-seq signal of peaks.

    :param signalmtx:
        The signal calculated from cc.pl.calculate_signal
    :param fill_between:
        Whether to fill with the range of signal values or not.
    :param alpha:
        The percentage of signals coverage in fill_between.
    :param before:
        The length(bp) calculated before the middle point of the peak.
    :param after:
        The length(bp) calculated after the middle point of the peak.
    :param nbins:
        The number of bins calculated.
    :param figsize:
        The size of the figure.
    :param font_size:
        The font of the words on the plot.
    :param color:
        The color of the plot, the same as color in plt.plot.
    :param textbelow:
        The distance of the bottom text and the plot.
    :param title:
        The title of the plot.
    :param bottom:
        The relative distance between the bottom words and the plot.
    :param save:
        Could be bool or str indicating the file name it would be saved.
        If `True`, a default name will be given and the plot would be saved as a png file.


    :example:
    >>> import pycallingcards as cc
    >>> exp_ccf = cc.datasets.SP1_K562HCT116_data(data = 'HCT116_SP1_ccf')
    >>> bg_ccf = cc.datasets.SP1_K562HCT116_data(data = 'HCT116_brd4_ccf')
    >>> peak_data = cc.pp.callpeaks(exp_ccf, bg_ccf, method = 'ccf_tools', reference = 'hg38', pvalue_cutoffbg = 0.0001, window_size = 2000, step_size = 500,
              pvalue_cutoffTTAA = 0.0000001, lam_win_size = None)
    >>> # If "https://www.encodeproject.org/files/ENCFF587ZMX/@@download/ENCFF587ZMX.bigWig" could not work, please download it and read the local path.
    >>> mtx = cc.pl.calculate_signal(peak_data,"https://www.encodeproject.org/files/ENCFF587ZMX/@@download/ENCFF587ZMX.bigWig")
    >>> cc.pl.signal_plot(mtx, alpha = 0.05)


    """

    mpl.rc_file_defaults()

    figure, axis = plt.subplots(figsize=figsize)

    axis.plot(
        np.array(range(0, after + before, int((after + before) / nbins))),
        np.log2(np.nanmean(signalmtx, axis=0) + 1),
        color=color,
    )

    if fill_between:
        if alpha < 0 or alpha > 1:
            raise ValueError("Invalid alpha.")
        peak_number = signalmtx.shape[0]
        axis.fill_between(
            np.array(range(0, after + before, int((after + before) / nbins))),
            np.log2(
                signalmtx[
                    np.argsort(signalmtx, axis=0)[int(peak_number * (1 - alpha / 2))],
                    range(0, 100),
                ]
                + 1
            ),
            np.log2(
                signalmtx[
                    np.argsort(signalmtx, axis=0)[int(peak_number * (alpha / 2))],
                    range(0, 100),
                ]
                + 1
            ),
            alpha=0.2,
            color=color,
        )

    axis.get_xaxis().set_visible(False)
    axis.text(
        0.05,
        bottom,
        str(before) + "bp",
        transform=axis.transAxes,
        fontsize=fontsize,
        ha="left",
        va="bottom",
    )
    axis.text(
        0.5,
        bottom,
        "Center",
        transform=axis.transAxes,
        fontsize=fontsize,
        ha="center",
        va="bottom",
    )
    axis.text(
        0.95,
        bottom,
        str(after) + "bp",
        transform=axis.transAxes,
        fontsize=fontsize,
        ha="right",
        va="bottom",
    )

    axis.set_title(title, fontsize=fontsize * 1.6)

    if save != False:
        if save == True:
            save = "Chipseq_signal_plot.png"
        figure.savefig(save, bbox_inches="tight")


def signal_heatmap(
    signalmtx: np.array,
    before: int = 10000,
    after: int = 10000,
    nbins: int = 100,
    figsize=(4, 15),
    fontsize: int = 10,
    colormap: str = "Reds",
    pad: float = 0.03,
    belowlength: float = 0,
    colormap_vmin: float = 0,
    colormap_vmax: float = 5,
    title: str = "Log2(FC) Chip-seq Signal Heatmap",
    save: bool = False,
):

    """\
    Plot the heatmap plot the Chip-seq signal of peaks.

    :param signalmtx:
        The signal calculated from cc.pl.calculate_signal
    :param before:
        The length(bp) calculated before the middle point of the peak.
    :param after:
        The length(bp) calculated after the middle point of the peak.
    :param nbins:
        The number of bins calculated.
    :param figsize:
        The size of the figure.
    :param font_size:
        The font of the words on the plot.
    :param color:
        The color of the plot, the same as Colormaps in Matplotlib.
    :param pad:
        Control the distance between plot and the colormap.
    :param belowlength:
        Control the distance between plot and the text below.
    :param colormap_vmin:
        vmin value of the colormap.
    :param colormap_vmax:
        vmax value of the colormap.
    :param title:
        The title of the plot.
    :param save:
        Could be bool or str indicating the file name it would be saved.
        If `True`, a default name will be given and the plot would be saved as a png file.


    :example:
    >>> import pycallingcards as cc
    >>> exp_ccf = cc.datasets.SP1_K562HCT116_data(data = 'HCT116_SP1_ccf')
    >>> bg_ccf = cc.datasets.SP1_K562HCT116_data(data = 'HCT116_brd4_ccf')
    >>> peak_data = cc.pp.callpeaks(exp_ccf, bg_ccf, method = 'ccf_tools', reference = 'hg38', pvalue_cutoffbg = 0.0001, window_size = 2000, step_size = 500,
              pvalue_cutoffTTAA = 0.0000001, lam_win_size = None)
    >>> # If "https://www.encodeproject.org/files/ENCFF587ZMX/@@download/ENCFF587ZMX.bigWig" could not work, please download it and read the local path.
    >>> mtx = cc.pl.calculate_signal(peak_data,"https://www.encodeproject.org/files/ENCFF587ZMX/@@download/ENCFF587ZMX.bigWig")
    >>> cc.pl.signal_heatmap(mtx, pad = 0.035)

    """

    figure, axis = plt.subplots(figsize=figsize)

    cf = axis.pcolormesh(
        np.log2(signalmtx[np.argsort(np.nanmean(signalmtx, axis=1)), :] + 1),
        cmap=colormap,
        vmin=colormap_vmin,
        vmax=colormap_vmax,
    )
    axis.get_yaxis().set_visible(False)
    axis.get_xaxis().set_visible(False)
    axis.text(
        0, -belowlength - 10 * fontsize, "-" + str(before) + "bp", fontsize=fontsize
    )
    axis.text(nbins * 0.41, -belowlength - 10 * fontsize, "Center", fontsize=fontsize)
    axis.text(
        nbins * 0.81, -belowlength - 10 * fontsize, str(after) + "bp", fontsize=fontsize
    )
    axis.set_title(title, fontsize=fontsize)

    cbar = figure.colorbar(cf, location="bottom", pad=pad)
    cbar.ax.tick_params(labelsize=fontsize * 1.3)

    if save != False:
        if save == True:
            save = "Chipseq_signal_plot_heatmap.png"
        figure.savefig(save, bbox_inches="tight")
