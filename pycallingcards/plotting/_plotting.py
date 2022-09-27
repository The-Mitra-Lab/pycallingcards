import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Union, Optional, List, Sequence, Iterable, Literal, Tuple
from matplotlib.axes import Axes
from matplotlib import rcParams, cm
from matplotlib import pyplot as pl

_draw_area_color = Optional[Literal['blue','red','green','pruple']]

def rank_peak_groups(
    adata_ccf: AnnData,
    groups: Union[str, Sequence[str]] = None,
    n_peaks: int = 10,
    peak_symbols: Optional[str] = None,
    key: Optional[str] = 'rank_peak_groups',
    fontsize: int = 8,
    ncols: int = 4,
    sharey: bool = True,
    show: Optional[bool] = True,
    save: Union[bool,str] = False
):

    """\
    Plot ranking of peaks.

    :param adata_ccf:
        Annotated data matrix.
    :param groups: Default is `None`.
        The groups used to show the peak ranking.
    :param n_peaks: Default is 10.
        Number of peaks that appear in the returned tables.
    :param peak_symbols: Default is `None`.
         Key for field in .var that stores peak symbols if you do not want to use .var_names.
    :param key: Default is `rank_peak_groups`.
        Key for the name of the cluster.
    :param fontsize: Default is 8.
        Fontsize for peak names.
    :param ncols: Default is 4.
        Number of panels shown per row.
    :param sharey: Default is `True`.
        Controls if the y-axis of each panels should be shared. By passing
        `sharey=False`, each panel has its own y-axis range.
    :param show: Default is `True`
        Controls if the plot shows or not. 
    :param save: Default is `False`.
        Could be bool or str indicating the file name it would be saved.
        If `True`, a default name would be given and the plot would be saved as png.


    :example:
    >>> import pycallingcards as cc
    >>> adata_ccf = cc.datasets.mousecortex_data(data="CCF")
    >>> cc.pl.rank_peaks_groups(adata_ccf)

    :See also: `tl.rank_peaks_groups`
    """

    n_panels_per_row = ncols
    if n_peaks < 1:
        raise NotImplementedError(
            "Specifying a negative number for n_peaks has not been implemented for "
            f"this plot. Received n_peaks={n_peaks}."
        )

    reference = str(adata_ccf.uns[key]['params']['reference'])
    group_names = adata_ccf.uns[key]['names'].dtype.names if groups is None else groups
    # one panel for each group
    # set up the figure
    n_panels_x = min(n_panels_per_row, len(group_names))
    n_panels_y = np.ceil(len(group_names) / n_panels_x).astype(int)

    from matplotlib import gridspec

    fig = pl.figure(
        figsize=(
            n_panels_x * rcParams['figure.figsize'][0],
            n_panels_y * rcParams['figure.figsize'][1],
        )
    )
    gs = gridspec.GridSpec(nrows=n_panels_y, ncols=n_panels_x, wspace=0.22, hspace=0.3)

    ax0 = None
    ymin = np.Inf
    ymax = -np.Inf
    for count, group_name in enumerate(group_names):
        peak_names = adata_ccf.uns[key]['names'][group_name][:n_peaks]
        pvalues = adata_ccf.uns[key]['pvalues'][group_name][:n_peaks]

        # Setting up axis, calculating y bounds
        if sharey:
            ymin = min(ymin, np.min(pvalues))
            ymax = max(ymax, np.max(pvalues))

            if ax0 is None:
                ax = fig.add_subplot(gs[count])
                ax0 = ax
            else:
                ax = fig.add_subplot(gs[count], sharey=ax0)
        else:
            ymin = np.min(pvalues)
            ymax = np.max(pvalues)
            ymax += 0.3 * (ymax - ymin)

            ax = fig.add_subplot(gs[count])
            ax.set_ylim(ymin, ymax)

        ax.set_xlim(-0.9, n_peaks - 0.1)

        # Mapping to peak_symbols
        if peak_symbols is not None:
            if adata_ccf.raw is not None and adata_ccf.uns[key]['params']['use_raw']:
                peak_names = adata_ccf.raw.var[peak_symbols][peak_names]
            else:
                peak_names = adata_ccf.var[peak_symbols][peak_names]

        # Making labels
        for ig, peak_name in enumerate(peak_names):
            ax.text(
                ig,
                pvalues[ig],
                peak_name,
                rotation='vertical',
                verticalalignment='bottom',
                horizontalalignment='center',
                fontsize=fontsize,
            )

        ax.set_title('{} vs. {}'.format(group_name, reference))
        if count >= n_panels_x * (n_panels_y - 1):
            ax.set_xlabel('ranking')

        # print the 'score' label only on the first panel per row.
        if count % n_panels_x == 0:
            ax.set_ylabel('pvalue')

    if sharey is True:
        ymax += 0.3 * (ymax - ymin)
        ax.set_ylim(ymin, ymax)

    if save != False:
        if save == True:
            save = f"rank_peak_groups_{adata_ccf.uns[key]['params']['groupby']}" + '.png'
        pl.savefig(save, bbox_inches='tight')
        
    if show:
        pl.show()

    pl.close() 



def draw_area(
    chromosome: str,
    start: int,
    end: int,
    extend: int,
    peaks: pd.DataFrame,
    htops: pd.DataFrame,
    reference: Union[str,pd.DataFrame],
    background: Union[None,pd.DataFrame] = None,
    adata: Optional[AnnData] = None,
    name: Optional[str] = None,
    key: Optional[str] = None ,
    htopkey: Optional[str] = None ,
    figsize: Tuple[int, int] = (10, 3),
    color: _draw_area_color = "blue",
    color_ccf: str = None,
    color_peak: str = None,
    color_genes: str = None,
    title: Optional[str] = None,
    example_length: int = 10000,
    peak_line: int = 1,
    save: Union[bool,str] = False):

    """\
    Plot the specific area of the genome.


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
    :param htops:
        pd.Datadrame of ccf
    :param reference:
        `hg38`, `mm10` or pd.DataFrame of the reference data.
    :param background: Default is `None`.
        pd.DataFrame of ccf or None. 
    :param adata: Default is `None`.
        Input along with `name` and `key`.
        It would only show the htops when the `key` of adata is `name`.
    :param name: Default is `None`.
        Input along with `adata` and `key`.
        It would only show the htops when the `key` of adata is `name`.
    :param key: Default is `None`.
        Input along with `adata` and `name`.
        It would only show the htops when the `key` of adata is `name`.
    :param htopkey: Default is `None`.
        Input along with `adata` and `name`.
        It would find the column `htopkey` of the htops file.
    :param figsize: Default is (10, 3).
        The size of the figure. 
    :param color:  [`blue`,`red`,`green`,`pruple`]. Default is `blue`.
        The color of the plot.
        If `color` is not a valid color, `color_ccf`, `color_peak`, `color_genes` should be utilized.
    :param color_ccf: Default is `None`.
        The color of ccf htops. Used only when `color` is not a valid color.
    :param color_peak: Default is `None`.
        The color of peaks. Used only when `color` is not a valid color.
    :param color_genes: Default is `None`.
        The color of genes. Used only when `color` is not a valid color.
    :param title: Default is `None`.
        The title of the plot. 
    :param example_length:  Default is 10000.
        The length of example.
    :param peak_line: Default is 1.
        The total number of peak lines. 
    :param save: Default is `False`. 
        Could be bool or str indicating the file name it would be saved.
        If `True`, a default name would be given and the plot would be saved as png.


    :Example:
    --------
    >>> import pycallingcards as cc
    >>> ccf_data = cc.datasets.mousecortex_data(data="ccf")
    >>> peak_data = cc.pp.callpeaks(ccf_data, method = "test", reference = "mm10", maxbetween = 2000,pvalue_cutoff = 0.01, lam_win_size = 1000000,  pseudocounts = 1, record = True)
    >>> adata_ccf = cc.datasets.mousecortex_data(data="CCF")
    >>> cc.pl.rank_peaks_groups(adata_ccf)
    >>> cc.pl.draw_area("chr12",50102917,50124960,400000,peak_data,ccf_data,"mm10",adata_ccf,"Neuron_Excit",'cluster',figsize = (30,6),peak_line = 4,color = "red")

    """

    import matplotlib.pyplot as pl

    if color == "blue":
        color_ccf = "cyan"
        color_peak = "royalblue"
        color_genes = "skyblue"

    elif color == "red":
        color_ccf = "tomato"
        color_peak = "red"
        color_genes = "mistyrose"

    elif color == "green":

        color_ccf = "lightgreen"
        color_peak = "palegreen"
        color_genes = "springgreen"

    elif color == "purple":

        color_ccf = "magenta"
        color_peak = "darkviolet"
        color_genes = "plum"


    peakschr = peaks[peaks.iloc[:,0] == chromosome]
    htopschr = htops[htops.iloc[:,0] == chromosome]
    

    if type(adata) == AnnData:
        if name != None:
            if key == "Index":
                adata = adata[name,:]
            else:
                adata = adata[adata.obs[key] == name]

                
        if htopkey == None:
            htopschr = htopschr[htopschr[5].isin(adata.obs.index)]
        else:
            htopschr = htopschr[htopschr[htopkey].isin(adata.obs.index)]

    if type(reference) == str:
        if reference == "hg38":
            refdata = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.hg38.Sorted.bed",delimiter="\t",header=None)
        elif reference == "mm10":
            refdata = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.mm10.Sorted.bed",delimiter="\t",header=None)
    elif type(reference) == pd.DataFrame:
        refdata = reference
    else:
        raise ValueError("Not valid reference.")


    refchr = refdata[refdata.iloc[:,0] == chromosome]

    r1 = refchr[(refchr.iloc[:,2]>=start-extend)  & (refchr.iloc[:,1]<= end + extend)].to_numpy()
    d1 = htopschr[(htopschr.iloc[:,1]>=start-extend)  & (htopschr.iloc[:,2]<= end + extend)]
    p1 = peakschr[(peakschr.iloc[:,1]>=start-extend)  & (peakschr.iloc[:,2]<= end + extend)].to_numpy()

    if type(background) == pd.DataFrame:
        backgroundchr = background[background.iloc[:,0] == chromosome]
        b1 = backgroundchr[(backgroundchr.iloc[:,1]>=start-extend)  & (backgroundchr.iloc[:,2]<= end + extend)]

    figure, axis = pl.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1,1]})

    if type(background) == pd.DataFrame:
        axis[0].plot(list(b1.iloc[:,1]), list(np.log(b1.iloc[:,3]+1)),"lightgray",marker = 'o',linestyle = 'None',markersize=6)
    
    axis[0].plot(list(d1.iloc[:,1]), list(np.log(d1.iloc[:,3]+1)),color_ccf,marker = 'o',linestyle = 'None',markersize=6)
    axis[0].axis('off')
    axis[0].set_xlim([start - extend, end + extend])


    pnumber = 0

    for i in range(len(p1)):

        axis[1].plot([p1[i,1],p1[i,2]], [ -1* (pnumber % peak_line) + 0.15,  -1* (pnumber % peak_line )+ 0.15], linewidth=10, c =color_peak)
        axis[1].text((p1[i,2]+extend/40),  -1*  (pnumber % peak_line )+ 0.15 , p1[i,0]+"_"+str(p1[i,1])+"_"+str(p1[i,2]),fontsize=14)
        pnumber += 1

    for i in range(len(r1)):

        axis[1].plot([max(r1[i,1],start - extend) ,min(r1[i,2],end + extend)], [1+i,1+i], linewidth=5, c = color_genes)
        axis[1].text(min(r1[i,2]+extend/40,end + extend), 1+i, r1[i,3]+", "+r1[i,4], fontsize=12)

        if r1[i,5] == "-":
            axis[1].annotate("",xytext=( min(r1[i,2],end + extend), 1+i), xy=( max(r1[i,1],start - extend), 1+i),  xycoords='data', va="center", ha="center",
                    size=20, arrowprops=dict(arrowstyle="simple", color =  color_genes))
        elif r1[i,5] == "+":
            axis[1].annotate("",xytext=(max(r1[i,1],start - extend), 1+i), xy=(min(r1[i,2],end + extend), 1+i),  xycoords='data', va="center", ha="center",
                    size=20, arrowprops=dict(arrowstyle="simple", color =  color_genes))

    axis[1].set_xlim([start - extend, end + extend])
    axis[1].axis('off')

    if title!= None:
        figure.suptitle(title, fontsize=16)

    if example_length != None:
        axis[1].plot([end+extend-example_length-example_length/5, end+extend-example_length/5], [-1,-1], linewidth=2, c = "k")
        axis[1].plot([end+extend-example_length-example_length/5, end+extend-example_length-example_length/5], [-1,-0.6], linewidth=2, c = "k")
        axis[1].plot([end+extend-example_length/5, end+extend-example_length/5], [-1,-0.6], linewidth=2, c = "k")
        axis[1].text(end+extend, -1, str(example_length)+"bp", fontsize=12)

    if save != False:
        if save == True:
            save = 'draw_area_' +chromosome + "_" + str(start) +  "_"  + str(end) + '.png'
        figure.savefig(save, bbox_inches='tight')

