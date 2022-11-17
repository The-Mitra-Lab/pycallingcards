import numpy as np
import pandas as pd
from typing import Union, Optional, List, Sequence, Iterable, Literal, Tuple
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
import pyBigWig
import tqdm
import matplotlib as mpl

_signaltype = Optional[Literal['mean', 'median', 'min', 'max', 'sum' ,'std']]

def calculate_signal(
    peak_data: pd.DataFrame,
    chipseq_signal: str,
    before: int = 10000,
    after: int = 10000,
    signaltype: _signaltype = "mean",
    nbins : int = 100
):


    """\
    Calculate the matrix of Chip-seq signal in each peak. This function uses  `pyBigWig <https://www.google.com/search?q=pyBigWig&rlz=1C5CHFA_enUS1018US1018&oq=pyBigWig&aqs=chrome.0.69i59j35i39j0i512j0i30l3j69i60l2.220j0j7&sourceid=chrome&ie=UTF-8>`__,
    please install it before using.

    :param peak_data:
        Peak_data file from cc.pp.callpeaks.
    :param chipseq_signal: 
        Chipseq signal file should be bigWig/bigBed file.
    :param before: Default is `10000`.
        The length(bp) we need to calculate before the middle point of the peak. 
    :param after: Default is `10000`.
        The length(bp) we need to calculate after the middle point of the peak. 
    :param signaltype: `['mean', 'median', 'min', 'max', 'sum' ,'std']`. Default is `'mean'`.
        Define the type of statistic that should be used over the bin size range.
    :param nbins: Default is `100`.
        The number of bins calculated.
     

    :example:
    >>> import pycallingcards as cc
    >>> exp_ccf = cc.datasets.SP1_K562HCT116_data(data = "experience_ccf")
    >>> bg_ccf = cc.datasets.SP1_K562HCT116_data(data = "background_ccf")
    >>> peak_data = cc.pp.callpeaks(exp_ccf, bg_ccf, method = "MACS2", reference = "hg38", pvalue_cutoffbg = 0.0001, window_size = 2000, step_size = 500,
                  pvalue_cutoffTTAA = 0.0000001, lam_win_size = None)
    >>> # If "https://www.encodeproject.org/files/ENCFF205TXT/@@download/ENCFF205TXT.bigWig" could not work, please download it and read the local path.
    >>> mtx = cc.pl. calculate_signal(peak_data,"https://www.encodeproject.org/files/ENCFF205TXT/@@download/ENCFF205TXT.bigWig")


    """

    bw = pyBigWig.open(chipseq_signal)


    chrom = list(peak_data["Chr"])
    start = list(peak_data["Start"])
    end = list(peak_data["End"])

    signalmtx = np.zeros((len(peak_data),nbins))

    for i in tqdm.tqdm(range(len(peak_data))):
        mid = int((start[i] + end[i])/2)
        inta = mid - before
        intb = mid + after
        signalmtx[i,:] = np.array(bw.stats(chrom[i], inta, intb, type=signaltype, nBins = nbins))
    
    return signalmtx


def signal_plot(
    signalmtx: np.array,
    fill_between: bool = True,
    alpha: float = 0.05,
    before: int = 10000,
    after: int = 10000,
    nbins : int = 100,
    figsize: Tuple[int, int] = (8,6),
    fontsize: int = 10,
    color: str = "red",
    title: str = 'Log2(FC) Chip-seq Signal',
    save: bool= False
):

    """\
    Plot the Chip-seq signal of peaks.

    :param signalmtx:
        The signal calculated from cc.pl.calculate_signal
    :param fill_between: Default is `True`.
        Whether to fill with the range of signal values.
    :param alpha: Default is `0.05`.
        The percentage of signals coverage in fill_between.
    :param before: Default is `10000`.
        The length(bp) we need to calculate before the middle point of the peak. 
    :param after: Default is `10000`.
        The length(bp) we need to calculate after the middle point of the peak. 
    :param nbins: Default is `100`.
        The number of bins calculated.
    :param figsize: Default is `(8,6)`.
        The size of the figure. 
    :param font_size: Default is `10`.
        The font of the words on the plot.
    :param color:  Default is `'red'`.
        The color of the plot, the same as color in plt.plot.
    :param title: Default is `'Log2(FC) Chip-seq Signal'`.
        The title of the plot. 
    :param save: Default is `False`.
        Could be bool or str indicating the file name it would be saved.
        If `True`, a default name would be given and the plot would be saved as png.

     
    :example:
    >>> import pycallingcards as cc
    >>> exp_ccf = cc.datasets.SP1_K562HCT116_data(data = "experience_ccf")
    >>> bg_ccf = cc.datasets.SP1_K562HCT116_data(data = "background_ccf")
    >>> peak_data = cc.pp.callpeaks(exp_ccf, bg_ccf, method = "MACS2", reference = "hg38", pvalue_cutoffbg = 0.0001, window_size = 2000, step_size = 500,
                  pvalue_cutoffTTAA = 0.0000001, lam_win_size = None)
    >>> # If "https://www.encodeproject.org/files/ENCFF205TXT/@@download/ENCFF205TXT.bigWig" could not work, please download it and read the local path.
    >>> mtx = cc.pl. calculate_signal(peak_data,"https://www.encodeproject.org/files/ENCFF205TXT/@@download/ENCFF205TXT.bigWig")
    >>> cc.pl.signal_plot(mtx, alpha = 0.2)


    """

    mpl.rc_file_defaults()

    figure, axis = plt.subplots(figsize=figsize)

    axis.plot(np.array(range(0,after+before,int((after+before)/nbins))), np.log2(np.nanmean(signalmtx,axis = 0)+1), color = color)

    if fill_between:
        if alpha <0 or alpha >1:
            raise ValueError("Invalid alpha.")
        peak_number = signalmtx.shape[0]
        axis.fill_between(np.array(range(0,after+before,int((after+before)/nbins))), 
                        np.log2(signalmtx[np.argsort(signalmtx,axis = 0)[int(peak_number*(1-alpha/2))],range(0,100)]+1), 
                        np.log2(signalmtx[np.argsort(signalmtx,axis = 0)[int(peak_number*(alpha/2))],range(0,100)]+1), alpha=0.2, color = color)

    axis.get_xaxis().set_visible(False)
    axis.text(0 , 0, "-"+str(before)+"bp", fontsize=fontsize)
    axis.text(before*0.9 , 0, "Center", fontsize=fontsize)
    axis.text((after+before)*0.9 , 0, str(after)+"bp", fontsize=fontsize)
    axis.set_title(title, fontsize=fontsize*1.6)

    if save != False:
        if save == True:
            save = 'Chipseq_signal_plot.png'
        figure.savefig(save, bbox_inches='tight')


def signal_heatmap(
    signalmtx: np.array,
    before: int = 10000,
    after: int = 10000,
    nbins : int = 100,
    figsize = (4, 15),
    fontsize: int = 10,
    colormap: str = "Reds",
    pad: float= 0.03,
    belowlength: float = 0,
    title: str = 'Log2(FC) Chip-seq Signal Heatmap',
    save: bool= False
):


    """\
    Heatmap plot the Chip-seq signal of peaks.

    :param signalmtx:
        The signal calculated from cc.pl.calculate_signal
    :param before: Default is `10000`.
        The length(bp) we need to calculate before the middle point of the peak. 
    :param after: Default is `10000`.
        The length(bp) we need to calculate after the middle point of the peak. 
    :param nbins: Default is `100`.
        The number of bins calculated.
    :param figsize: Default is `(4,15)`.
        The size of the figure. 
    :param font_size: Default is `10`.
        The font of the words on the plot.
    :param color: Default is `'Reds'`.
        The color of the plot, the same as Colormaps in Matplotlib.
    :param color: Default is `0.03`.
        Control the distance between plot and the colormap.
    :param belowlength: Default is `0`.
        Control the distance between plot and the text below.
    :param title: Default is `'Log2(FC) Chip-seq Signal Heatmap'`.
        The title of the plot. 
    :param save: Default is `False`.
        Could be bool or str indicating the file name it would be saved.
        If `True`, a default name would be given and the plot would be saved as png.

     
    :example:
    >>> import pycallingcards as cc
    >>> exp_ccf = cc.datasets.SP1_K562HCT116_data(data = "experience_ccf")
    >>> bg_ccf = cc.datasets.SP1_K562HCT116_data(data = "background_ccf")
    >>> peak_data = cc.pp.callpeaks(exp_ccf, bg_ccf, method = "MACS2", reference = "hg38", pvalue_cutoffbg = 0.0001, window_size = 2000, step_size = 500,
                  pvalue_cutoffTTAA = 0.0000001, lam_win_size = None)
    >>> # If "https://www.encodeproject.org/files/ENCFF205TXT/@@download/ENCFF205TXT.bigWig" could not work, please download it and read the local path.
    >>> mtx = cc.pl. calculate_signal(peak_data,"https://www.encodeproject.org/files/ENCFF205TXT/@@download/ENCFF205TXT.bigWig")
    >>> cc.pl.signal_heatmap(mtx)

    """


    figure, axis = plt.subplots(figsize=figsize)
        
    cf = axis.pcolormesh(np.log2(signalmtx[np.argsort(np.nanmean(signalmtx,axis = 1)),:]+1),cmap = colormap)
    axis.get_yaxis().set_visible(False) 
    axis.get_xaxis().set_visible(False)
    axis.text(0 , -belowlength-10*fontsize, "-"+str(before)+"bp", fontsize=fontsize)
    axis.text(nbins*0.41 , -belowlength-10*fontsize, "Center", fontsize=fontsize)
    axis.text(nbins*0.81 , -belowlength-10*fontsize, str(after)+"bp", fontsize=fontsize)
    axis.set_title(title, fontsize=fontsize)

    cbar = figure.colorbar(cf, location='bottom', pad=pad)
    cbar.ax.tick_params(labelsize=fontsize*1.3) 

    if save != False:
        if save == True:
            save = 'Chipseq_signal_plot_heatmap.png'
        figure.savefig(save, bbox_inches='tight')