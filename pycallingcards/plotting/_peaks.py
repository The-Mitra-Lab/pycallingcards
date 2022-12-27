import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Union, Optional, List, Sequence, Iterable, Literal, Tuple
from matplotlib.axes import Axes
from matplotlib import rcParams, cm
from matplotlib import pyplot as plt



_draw_area_color = Optional[Literal['blue','red','green','pruple']]


def draw_area(
    chromosome: str,
    start: int,
    end: int,
    extend: int,
    peaks: pd.DataFrame,
    insertions: pd.DataFrame,
    reference: Union[str,pd.DataFrame],
    background: Union[None,pd.DataFrame] = None,
    adata: Optional[AnnData] = None,
    name: Optional[str] = None,
    key: Optional[str] = None ,
    insertionkey: Optional[str] = None ,
    figsize: Tuple[int, int] = (10, 3),
    plotsize: list = [1,1,2],
    bins: Optional[int] = None,
    color: _draw_area_color = "red",
    color_ccf: str = None,
    color_peak: str = None,
    color_genes: str = None,
    title: Optional[str] = None,
    example_length: int = 10000,
    peak_line: int = 1,
    font_size: int = 1,
    save: Union[bool,str] = False):

    """\
    Plot the specific area of the genome. This plot contains three parts. 
    The first part is a plot of insertions plot: one dot is one insertion and the height is log(reads + 1).
    The second part is the distribution plot of insertions.
    If backgound is input, the colored one would be the experiment inerstions/distribution and the grey one would be the backgound one.
    If backgound is not input and adata/name/key are provided, the colored one would be the inerstions/distribution for specific group and the grey one would be the whole data.
    The third part composes of reference genes and peaks.


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
        pd.Datadrame of ccf
    :param reference:
        `'hg38'`, `'mm10'`, `'sacCer3'` or pd.DataFrame of the reference data.
    :param background: Default is `None`.
        pd.DataFrame of ccf or None. 
    :param adata: Default is `None`.
        Input along with `name` and `key`.
        It would only show the insertions when the `key` of adata is `name`.
    :param name: Default is `None`.
        Input along with `adata` and `key`.
        It would only show the insertions when the `key` of adata is `name`.
    :param key: Default is `None`.
        Input along with `adata` and `name`.
        It would only show the insertions when the `key` of adata is `name`.
    :param insertionkey: Default is `None`('Barcodes').
        Input along with `adata` and `name`.
        It would find the column `insertionkey` of the insertions file.
    :param figsize: Default is (10, 3).
        The size of the figure. 
    :param plotsize: Default is [1,1,2].
        The relateive size of the dot plot, distribution plot and the peak plot. 
    :param bins:  Default is `None`.
        The bins of histogram. I would automatically calculate if None.
    :param color:  `['blue','red','green','pruple']`. Default is `red`.
        The color of the plot.
        If `color` is not a valid color, `color_ccf`, `color_peak`, `color_genes` should be utilized.
    :param color_ccf: Default is `None`.
        The color of ccf insertions. Used only when `color` is not a valid color.
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
    :param font_size: Default is `10`.
        The relative font of the words on the plot.
    :param save: Default is `False`. 
        Could be bool or str indicating the file name it would be saved.
        If `True`, a default name would be given and the plot would be saved as png.


    :Example:
    --------
    >>> import pycallingcards as cc
    >>> ccf_data = cc.datasets.mousecortex_data(data="ccf")
    >>> peak_data = cc.pp.callpeaks(ccf_data, method = "test", reference = "mm10", record = True)
    >>> adata_ccf = cc.datasets.mousecortex_data(data="CCF")
    >>> cc.pl.rank_peaks_groups(adata_ccf)
    >>> cc.pl.draw_area("chr12",50102917,50124960,400000,peak_data,ccf_data,"mm10",adata_ccf,"Neuron_Excit",'cluster',figsize = (30,6),peak_line = 4,color = "red")

    """


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
    insertionschr = insertions[insertions.iloc[:,0] == chromosome]
    

    if type(adata) == AnnData:
        if name != None:
            if key == "Index":
                adata = adata[name,:]
            else:
                adata = adata[adata.obs[key] == name]

                
        if insertionkey == None:
            insertionschr = insertionschr[insertionschr["Barcodes"].isin(adata.obs.index)]
        else:
            insertionschr = insertionschr[insertionschr[insertionkey].isin(adata.obs.index)]

    if type(reference) == str:
        if reference == "hg38":
            refdata = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.hg38.Sorted.bed",delimiter="\t",header=None)
        elif reference == "mm10":
            refdata = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.mm10.Sorted.bed",delimiter="\t",header=None)
        elif reference == "sacCer3":
            refdata = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.sacCer3.Sorted.bed",delimiter="\t",header=None)
        

    elif type(reference) == pd.DataFrame:
        refdata = reference
    else:
        raise ValueError("Not valid reference.")


    refchr = refdata[refdata.iloc[:,0] == chromosome]

    r1 = refchr[(refchr.iloc[:,2]>=start-extend)  & (refchr.iloc[:,1]<= end + extend)].to_numpy()
    d1 = insertionschr[(insertionschr.iloc[:,1]>=start-extend)  & (insertionschr.iloc[:,2]<= end + extend)]
    p1 = peakschr[(peakschr.iloc[:,1]>=start-extend)  & (peakschr.iloc[:,2]<= end + extend)].to_numpy()
    
    if bins == None:
        bins = int(min(plt.rcParams['figure.dpi'] * figsize[0], (end - start + 2*extend))/4)
    
    bins = list(range(start - extend,end + extend,int((end - start + 2*extend)/bins)))
 

    

    if type(background) == pd.DataFrame:
        
        backgroundchr = background[background.iloc[:,0] == chromosome]
        b1 = backgroundchr[(backgroundchr.iloc[:,1]>=start-extend)  & (backgroundchr.iloc[:,2]<= end + extend)]
        
    elif name != None:
        
        backgroundchr = insertions[insertions.iloc[:,0] == chromosome]
        b1 = backgroundchr[(backgroundchr.iloc[:,1]>=start-extend)  & (backgroundchr.iloc[:,2]<= end + extend)]
        

    figure, axis = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': plotsize})
    

    if type(background) == pd.DataFrame:
        axis[0].plot(list(b1.iloc[:,1]), list(np.log(b1.iloc[:,3]+1)),"lightgray",marker = 'o',linestyle = 'None',markersize=6)
        counts, binsbg = np.histogram(np.array(b1.iloc[:,1]),bins=bins)
        axis[1].hist(binsbg[:-1], binsbg, weights=np.log(counts+1),color = "lightgray")
        
    elif name != None:
        
        axis[0].plot(list(b1.iloc[:,1]), list(np.log(b1.iloc[:,3]+1)),"lightgray",marker = 'o',linestyle = 'None',markersize=6)
        counts, binsbg = np.histogram(np.array(b1.iloc[:,1]),bins=bins)
        axis[1].hist(binsbg[:-1], binsbg, weights=counts,color = "lightgray")
        
    
    axis[0].plot(list(d1.iloc[:,1]), list(np.log(d1.iloc[:,3]+1)),color_ccf,marker = 'o',linestyle = 'None',markersize=6)
    axis[0].axis('off')
    axis[0].set_xlim([start - extend, end + extend])

    
    counts, binsccf = np.histogram(np.array(d1.iloc[:,1]),bins=bins)
    axis[1].hist(binsccf[:-1], binsccf, weights=counts,color = color_ccf)
    axis[1].set_xlim([start - extend, end + extend])
    axis[1].axis('off')

    

    pnumber = 0

    for i in range(len(p1)):

        axis[2].plot([p1[i,1],p1[i,2]], [ -1* (pnumber % peak_line) + 0.15,  -1* (pnumber % peak_line )+ 0.15], linewidth=10, c =color_peak)
        axis[2].text((p1[i,2]+extend/40),  -1*  (pnumber % peak_line )+ 0.15 , p1[i,0]+"_"+str(p1[i,1])+"_"+str(p1[i,2]),fontsize=14*font_size)
        pnumber += 1

    for i in range(len(r1)):

        axis[2].plot([max(r1[i,1],start - extend) ,min(r1[i,2],end + extend)], [1+i,1+i], linewidth=5, c = color_genes)
        axis[2].text(min(r1[i,2]+extend/40,end + extend), 1+i, r1[i,3]+", "+r1[i,4], fontsize=12*font_size)

        if r1[i,5] == "-":
            axis[2].annotate("",xytext=( min(r1[i,2],end + extend), 1+i), xy=( max(r1[i,1],start - extend), 1+i),  xycoords='data', va="center", ha="center",
                    size=20, arrowprops=dict(arrowstyle="simple", color =  color_genes))
        elif r1[i,5] == "+":
            axis[2].annotate("",xytext=(max(r1[i,1],start - extend), 1+i), xy=(min(r1[i,2],end + extend), 1+i),  xycoords='data', va="center", ha="center",
                    size=20, arrowprops=dict(arrowstyle="simple", color =  color_genes))

    axis[2].set_xlim([start - extend, end + extend])
    axis[2].axis('off')

    if title!= None:
        figure.suptitle(title, fontsize=16*font_size)

    if example_length != None:
        axis[2].plot([end+extend-example_length-example_length/5, end+extend-example_length/5], [-1-peak_line,-1-peak_line], linewidth=2, c = "k")
        axis[2].plot([end+extend-example_length-example_length/5, end+extend-example_length-example_length/5], [-1-peak_line,-0.6-peak_line], linewidth=2, c = "k")
        axis[2].plot([end+extend-example_length/5, end+extend-example_length/5], [-1-peak_line,-0.6-peak_line], linewidth=2, c = "k")
        axis[2].text(end+extend, -1-peak_line, str(example_length)+"bp", fontsize=12*font_size)

    if save != False:
        if save == True:
            save = 'draw_area_' +chromosome + "_" + str(start) +  "_"  + str(end) + '.png'
        figure.savefig(save, bbox_inches='tight')
        


def _myFuncsorting(e):
    try:
        return int(e[3:])
    except :
        return int(ord(e[3:]))


def whole_peaks(
    peak_data: pd.DataFrame,
    reference = "mm10",
    figsize: Tuple[int, int] = (100, 50),
    title: str = "Peaks over chromosomes",
    font_size: int = 10,
    linewidth: float = 4,
    color: str = "black",
    added: int = 10000,
    height_name: str = "Experiment Insertions",
    save: Union[bool,str] = False
):

    """\
    Plot all the peaks in chromosomes.

    :param peak_data:
        Peak_data file from cc.pp.callpeaks.
    :param reference: `['mm10','hg38',None]` Default is `'hg38'`.
        The reference of the data.
    :param figsize: Default is `(100, 40)`.
        The size of the figure. 
    :param title: Default is `'Peaks over chromosomes'`.
        The title of the plot. 
    :param font_size: Default is `10`.
        The font of the words on the plot.
    :param linewidth: Default is `4`.
        The linewidth of the words on the plot.
    :param color:  Default is `'black'`.
        The color of the plot, the same as color in plt.plot.
    :param added: Default is `10000`.
        Only valide when there is no reference provided. The max length(bp) added to the end of a chromosome shown.
    :param height_name: Default is 'Experiment Insertions'.
        The height of each peak. If `None`, it would all be the same height.
    :param save: Default is `False`.
        Could be bool or str indicating the file name it would be saved.
        If `True`, a default name would be given and the plot would be saved as png.


    :example:
    >>> import pycallingcards as cc
    >>> ccf_data = cc.datasets.mousecortex_data(data="ccf")
    >>> peak_data = cc.pp.callpeaks(ccf_data, method = "test", reference = "mm10", record = True)
    >>> cc.pl.whole_peaks(peak_data)

    """

    sortlist = list(peak_data["Chr"].unique())
    sortlist.sort(key=_myFuncsorting)
    sortlist = np.array(sortlist)

    
    
    if reference == "mm10":
        ref = np.array([195471971, 182113224, 160039680, 156508116, 151834684, 149736546, 145441459, 129401213, 124595110,  130694993, 
               122082543, 120129022, 120421639, 124902244, 104043685, 98207768, 94987271, 90702639, 61431566, 171031299, 91744698])
        sortlist1 = np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
                   'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chrX', 'chrY'])
        ref = ref[np.in1d(sortlist1, sortlist)]
        sortlist = sortlist1[np.in1d(sortlist1, sortlist)]
        
    elif reference == "hg38":
        ref = np.array([248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422,
              135086622, 133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285, 58617616, 64444167,
                46709983, 50818468, 156040895, 57227415])
        sortlist1 = np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
                    'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 
                    'chr21', 'chr22','chrX', 'chrY'])
        ref = ref[np.in1d(sortlist1, sortlist)]
        sortlist = sortlist1[np.in1d(sortlist1, sortlist)]
        
    else:
        ref = []

        
        
        for chrom in range(len(sortlist)):
            end = list(peak_data[peak_data["Chr"] == sortlist[chrom]]["End"])
            ref.append(end[-1] + added)
            
    ref = np.array(ref)
    
    figure, axis = plt.subplots(len(sortlist), 1, figsize=figsize)
    figure.suptitle(title, fontsize=9*font_size, y = 0.90)
    
        
    for chrom in range(len(sortlist)):
        
        start = list(peak_data[peak_data["Chr"] == sortlist[chrom]]["Start"])
        end = list(peak_data[peak_data["Chr"] == sortlist[chrom]]["End"])
        if height_name != None:
            insertions = list(peak_data[peak_data["Chr"] == sortlist[chrom]][height_name])
        else:
            insertions = [1] * len(start)

        axis[chrom].plot([0, ref[chrom]],[0,0],color, linewidth = linewidth)
        axis[chrom].set_xlim([0, ref[chrom]])
        axis[chrom].text(ref[chrom], 0, sortlist[chrom], fontsize=6*font_size)
        for peak in range(len(start)):
            axis[chrom].plot([start[peak] , start[peak]],[0,np.log(insertions[peak]+1)],color, linewidth = linewidth)
            axis[chrom].plot([end[peak], end[peak]],[0,np.log(insertions[peak]+1)],color, linewidth = linewidth)
        axis[chrom].axis('off')
        axis[chrom].text(20, -2, "0bp", fontsize=4*font_size)
        #axis[chrom].text(int(ref[chrom]/2) , -2, str(int(ref[chrom]/2))+"bp", fontsize=4*font_size)
        axis[chrom].text(ref[chrom] , -2, str(ref[chrom])+"bp", fontsize=4*font_size)

    if save != False:
        if save == True:
            save = 'Peaks_over_chromosomes.png'
        figure.savefig(save, bbox_inches='tight')

