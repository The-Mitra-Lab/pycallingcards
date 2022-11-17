import numpy as np
from anndata import AnnData
from typing import Union, Optional, List, Sequence, Iterable, Literal, Tuple
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns



def volcano_plot(
    adata_ccf: AnnData,
    figsize: Tuple[int, int] = (10, 6),
    font_size: int = 1,
    pvalue_cutoff: float = 0.01,
    lfc_cutoff: float = 4,
    colorleft: str = 'indianred',
    colorright: str = 'lightseagreen',
    title: str = "Volcano plot",
    labelleft: Union[None,List[float]] = None,
    labelright: Union[None,List[float]] = None,
    save: Union[bool,str] = False
    ):


    """\
    Volcano plot of for two sample/cluster comparing.
    Please make sure that alternative = 'two-sided' for the differential binding analysis.

    
    :param figsize: Default is `(4,15)`.
        The size of the figure. 
    :param font_size: Default is `10`.
        The font of the words on the plot.
    :param colorleft: Default is `'indianred'`.
        The color of the dot for the left group.
    :param colorright: Default is `'lightseagreen'`.
        The color of the dot for the right group.
    :param pvalue_cutoff: Default is `0.01`.
        The pvalue cutoff.
    :param lfc_cutoff: Default is `4`.
        The log fold change cutoff.
    :param title: Default is `'Log2(FC) Chip-seq Signal Heatmap'`.
        The title of the plot. 
    :param labelleft: Default is `None`.
        The exact place for left label. Default would automatically give it a place on left top of the plot.
    :param labelright: Default is `None`.
        The exact place for left label. Default would automatically give it a place on lrighr top of the plot.s
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


    

    label = list(adata_ccf.obs.index)
    pva = -np.log10(np.array(adata_ccf.uns['fisher_exact']['pvalues'].tolist())[:,1])
    fc = np.array(adata_ccf.uns['fisher_exact']['logfoldchanges'].tolist())[:,1]

    sns.set_theme()
    figure, axis = plt.subplots(1, 1, figsize=figsize)
    plt.title(title, fontsize=25*font_size)

    p_cutoff = -np.log10(pvalue_cutoff)

    lim = int(max(np.abs(fc))+2)
    maxy = max(pva)

    plt.plot(fc,pva,'gray', marker = "o",linestyle = 'None')
    plt.plot(fc[(pva>p_cutoff) & (fc<(-1*lfc_cutoff))],pva[(pva>p_cutoff) & (fc<(-1*lfc_cutoff))],colorleft, marker = "o",linestyle = 'None')
    plt.plot(fc[(pva>p_cutoff) & (fc>lfc_cutoff)],pva[(pva>p_cutoff) & (fc>lfc_cutoff)],colorright, marker = "o",linestyle = 'None')

    if labelleft == None:
        plt.text(-lim*0.85,0.85*maxy,label[0], fontsize=15*font_size)
    else:
        plt.text(labelleft[0],labelleft[1],label[0], fontsize=15*font_size)

    if labelright == None:
        plt.text(lim*0.7,0.85*maxy,label[1], fontsize=15*font_size)
    else:
        plt.text(labelright[0],labelright[1], label[1], fontsize=15*font_size)

    plt.ylabel('-$Log_10$ P-value',fontsize=15*font_size)
    plt.xlabel('$Log_2$ Fold Change in binding', fontsize=15*font_size)
    plt.tick_params(axis='both', which='major', labelsize=13*font_size)
    plt.xlim([-lim, lim])
    
    if save != False:
        if save == True:
            save = 'Volcano_plot.png'
        plt.savefig(save, bbox_inches='tight')

    mpl.rc_file_defaults()
        


