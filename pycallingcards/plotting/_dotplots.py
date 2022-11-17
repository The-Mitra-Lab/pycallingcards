import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Union, Optional, List, Sequence, Iterable, Literal, Tuple
from matplotlib.axes import Axes
from matplotlib import rcParams, cm
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib as mpl



def dotplot_bulk(
    adata_ccf: AnnData,
    rna: pd.DataFrame,
    selected_list: list,
    num_list: list,
    xticklabels: list = None,
    rate: float = 50,
    figsize:  Tuple[int, int] = (12, 15),
    dotsize: float = 5,
    cmap : str= "Reds",
    title = "DE binding & RNA",
    save: bool = False
):


    """\
    Plot ranking of peaks.

    :param adata_ccf:
        Anndata of peak.
    :param rna: 
        pd.DataFrame of RNA expression.
    :param selected_list:
        A list of peak going to show.
    :param num_list:
        The distribution of samples in rna. 
        eg. the first three columns for rna is female and the following two columns is male data, then num_list should be [3,2]
    :param xticklabels: Default is `None`.
        xticklabels for the column. If `None`, it would be the index of adata_ccf.obs
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
    :param save: Default is `False`.
        Could be bool or str indicating the file name it would be saved.
        If `True`, a default name would be given and the plot would be saved as png.


    :example:
    >>> import pycallingcards as cc
    >>> cc.pl.dotplot(adata_ccf,rna,selected_list = list(result["Peak"]), num_list = [3,3])
    """


    sns.set_theme()

    length = rna.shape[1]
    num_cluster = adata_ccf.shape[0]
    df = adata_ccf.var[["Gene Name1","Gene Name2"]]
    rna_list = list(rna.index)

    if xticklabels == None:
        xticklabels = list(adata_ccf.obs.index)


    index0 = []
    index1 = []
    index2 = []
    result_ccf = []

    for i in selected_list:

        gene1 = df.loc[i][0]
        gene2 = df.loc[i][1]
        if gene1 in rna_list and gene2 in rna_list:
            result_ccf.append(adata_ccf[:,i].X.T.toarray()[0].tolist() + list(rna.loc[gene1]) + list(rna.loc[gene2]))
            index0.append(i)
            index1.append(gene1)
            index2.append(gene2)
        elif gene1 in rna_list and gene2 not in rna_list:
            result_ccf.append(adata_ccf[:,i].X.T.toarray()[0].tolist() + list(rna.loc[gene1]) + [0]*length)
            index0.append(i)
            index1.append(gene1)
            index2.append(gene2)
        elif gene1 not in rna_list and gene2 in rna_list:
            result_ccf.append(adata_ccf[:,i].X.T.toarray()[0].tolist() + [0]*length + list(rna.loc[gene2]))
            index0.append(i)
            index1.append(gene1)
            index2.append(gene2)



    data = np.log2(np.array(result_ccf)+1)
    selected_length = data.shape[0]


    xticks = list(range(num_cluster))
    yticks = list(range(selected_length-1,-1,-1))


    fig, ax = plt.subplots(1, 3, figsize=figsize)

    x = []
    y = []
    cs = []
    cs1 = []
    cs2 = []

    for cluster in range(num_cluster):
        cs = cs + list(data[:,cluster])
        x = x + [cluster]*selected_length 
        y = y + list(range(selected_length -1,-1,-1))
        cs1 = cs1 + list(np.mean(data[:,(num_cluster+sum(num_list[0:cluster])):(num_cluster+sum(num_list[0:cluster+1]))],axis = 1))
        cs2 = cs2 + list(np.mean(data[:,(num_cluster+sum(num_list)+sum(num_list[0:cluster])):(num_cluster+sum(num_list)+sum(num_list[0:cluster+1]))],axis = 1))
        
    
    cs = np.array(cs)
    cs = dotsize*cs/max(cs)
    
    cs1 = np.array(cs1)
    cs1 = dotsize*cs1/max(cs1)
    
    cs2 = np.array(cs2)
    cs2 = dotsize*cs2/max(cs2)


    ax[0].scatter(x, y, c= cs, s= rate*np.array(cs), cmap =  cmap)
    ax[0].axis(xmin=-1,xmax=num_cluster)
    ax[0].set_xticks(xticks)
    ax[0].set_xticklabels(xticklabels)
    ax[0].set_yticks(yticks)
    ax[0].set_yticklabels(index0)

    ax[1].scatter(x, y, c= cs1, s= rate*np.array(cs1), cmap =  cmap)
    ax[1].axis(xmin=-1,xmax=num_cluster)
    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels(xticklabels)
    ax[1].set_yticks(yticks)
    ax[1].set_yticklabels(index1)


    ax[2].scatter(x, y, c= cs2, s= rate*np.array(cs2), cmap =  cmap)
    ax[2].axis(xmin=-1,xmax=num_cluster)
    ax[2].set_xticks(xticks)
    ax[2].set_xticklabels(xticklabels)
    ax[2].set_yticks(yticks)
    ax[2].set_yticklabels(index2)


    plt.tight_layout()
    plt.suptitle(title)
    fig.subplots_adjust(top=0.96)


    if save != False:
        if save == True:
            save = f"dotplot" + '.png'
        plt.savefig(save, bbox_inches='tight')

    
    mpl.rc_file_defaults()
    



def dotplot_sc(
    adata_ccf: AnnData,
    adata: AnnData,
    result: pd.DataFrame,
    rate: float = 50,
    figsize: Tuple[int, int] = (10, 120),
    size: int = 1,
    cmap: str = "Reds",
    title = "DE binding & RNA",
    save: bool = False
):


    """\
    Plot ranking of peaks.

    :param adata_ccf:
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
        The colormap of the plot. 
    :param title: Default is `'DE binding & RNA'`.
        The title of the plot. 
    :param save: Default is `False`.
        Could be bool or str indicating the file name it would be saved.
        If `True`, a default name would be given and the plot would be saved as png.


    :example:
    >>> import pycallingcards as cc
    >>> cc.pl.dotplot_sc(adata_ccf,adata,result)
    
    """


    sns.set_theme()

    genelist = list(result['Gene'])
    peaklist = list(result["Peak"])

    clusterlist = list(adata_ccf.obs["cluster"].unique())
    clusterlist.sort()

    clusterandata = np.array(adata_ccf.obs["cluster"])
    clusterehere = {}

    for i in range(len(clusterlist)):  
        clusterehere[clusterlist[i]] = np.where(clusterandata == clusterlist[i] )

    geneinfor_total = []
    peakinfor_total = []

    for num in range(len(genelist)):
        geneinfor = []
        peakinfor = []
        for cluster in range(len(clusterlist)):  
            peakinfor.append(adata_ccf[clusterehere[clusterlist[cluster]][0],peaklist[num]].X.mean())
            geneinfor.append(float(adata[clusterehere[clusterlist[cluster]][0],genelist[num]].X.mean()))
        geneinfor_total.append(geneinfor)
        peakinfor_total.append(peakinfor)
        
    geneinfor_total = np.array(geneinfor_total)
    peakinfor_total = np.log2(np.array(peakinfor_total)+1)

    geneinfor_total = geneinfor_total-geneinfor_total.min(axis = 1).reshape(-1,1)
    peakinfor_total = peakinfor_total-peakinfor_total.min(axis = 1).reshape(-1,1)

    geneinfor_total = geneinfor_total/geneinfor_total.max(axis = 1).reshape(-1,1)
    peakinfor_total = peakinfor_total/peakinfor_total.max(axis = 1).reshape(-1,1)


    x = list(range(len(clusterlist)))
    rate = 100
    
    total_num = len(genelist)
    fig, ax = plt.subplots(total_num , 1, figsize=figsize)

    fig.patch.set_visible(False)

    small =[]*len(clusterlist)
    for num in range(total_num):
        
        for spine in ['top', 'right', 'left','bottom']:
            ax[num].spines[spine].set_visible(False)
            

        ax[num].scatter(x , [1]*len(clusterlist),c=geneinfor_total[num], s=rate*geneinfor_total[num] , facecolor='blue',cmap = cmap)
        ax[num].scatter(x , [0.5]*len(clusterlist),c=peakinfor_total[num], s=rate*peakinfor_total[num], facecolor='blue',cmap = cmap)
        ax[num].axis(ymin=0,ymax=1.5)
        ax[num].set_yticks([0.5,1])
        ax[num].set_yticklabels([peaklist[num],genelist[num]], fontsize=10*size)
        ax[num].set_xticks(x)
        ax[num].set_xticklabels(small, fontsize=10*size)
        

    ax[num].set_xticklabels(clusterlist,rotation=90)
    plt.suptitle(title,size = 15*size)
    fig.subplots_adjust(top=0.977)


    if save != False:
        if save == True:
            save = f"dotplot" + '.png'
        plt.savefig(save, bbox_inches='tight')


    mpl.rc_file_defaults()
    

