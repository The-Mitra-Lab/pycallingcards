import numpy as np
from anndata import AnnData
from typing import Union, Optional, List, Sequence, Iterable, Literal, Tuple
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def heatmap(
    adata_ccf: AnnData,
    rna: Union[pd.DataFrame, None] = None,
    figsize: Tuple[int, int] = (28, 8),
    font_scale: float = 1,
    cmap: str ='BuPu',
    rnalabels: list = None,
    log: bool = True,
    title: str = "Relative calling cards and RNA information",
    save: Union[bool,str] = False
):

    """\
    Plot ranking of peaks. 

    :param adata_ccf:
        Annotated data matrix.
    :param ran: pd.DataFrame.
        pd.DataFrame for RNA data (genes*sample). Make sure the sample is in the same order of adata_ccf.obs
    :param figsize: Default is (28, 8). 
        The size of the figure.
    :param font_scale: Default is 1.
        The scale of the font size.
    :param cmap: Default is `'BuPu'`.
        Color map of the plot.
    :param rnalabels: Default is `None`.
        The labels of the rna data. Be sure the length of list match the number of sample in rna file.
    :param log: Default is `True`.
        Whether to log transform the gene expression or not.
   :param title: Default is `'Relative calling cards and RNA information'`.
        The title of the plot. 
    :param save: Default is `False`.
        Could be bool or str indicating the file name it would be saved.
        If `True`, a default name would be given and the plot would be saved as png.

    """

    sns.set(rc={'figure.figsize':figsize})
    sns.set(font_scale=font_scale)

    if type(rna) == pd.DataFrame:

        print("Please make sure that the samples in adata_ccf and rna are in the same order.")

        rna_list = list(rna.index)
        g = adata_ccf.var[['Gene Name1','Gene Name2']]

        result_ccf = []
        ccf = np.array((adata_ccf.X.todense()))
        index_result = []
        
        length = rna.shape[1]
        groupnumber = ccf.shape[0]

        for i in range(adata_ccf.shape[1]):
            gene1 = g.iloc[i,0]
            gene2 = g.iloc[i,1]
            if gene1 in rna_list and gene2 in rna_list:
                index_result.append([g.index[i],gene1,gene2])
                result_ccf.append(list(ccf[:,i]) + list(rna.loc[gene1]) + list(rna.loc[gene2]))
            elif gene1 in rna_list and gene2 not in rna_list:
                index_result.append([g.index[i],gene1,gene2])
                result_ccf.append(list(ccf[:,i]) + list(rna.loc[gene1]) + [0]*length)
            elif gene1 not in rna_list and gene2 in rna_list:
                index_result.append([g.index[i],gene1,gene2])
                result_ccf.append(list(ccf[:,i]) + [0]*length + list(rna.loc[gene2]))
            else:
                index_result.append([g.index[i],gene1,gene2])
                result_ccf.append(list(ccf[:,i]) + [0]*(2*length) )


        
        if rnalabels != None:
            yticklabels = list(adata_ccf.obs.index) + [""] + ["Ref gene1 " + s for s in rnalabels] + ["Ref gene2 " + s for s in rnalabels]
        else:
            yticklabels = list(adata_ccf.obs.index) + [""] + ["Ref gene1"]*length + ["Ref gene2"]*length 


        data = np.array(result_ccf)

        size = data.shape[0]
        secondnum = groupnumber + length 
        
        data[:,0:groupnumber] = data[:,0:groupnumber]/data[:,0:groupnumber].sum(axis = 1).reshape((size,1))
        data = data.T
        

        temp = data[0:groupnumber,:]
        rank = np.lexsort(temp[::-1])
        
        data = data[:,rank]
        data = data[:,data[groupnumber:,:].sum(axis = 0)>0]

        if log:
            data[groupnumber:,:] = np.log(data[groupnumber:,:] +1)
            
        size = data.shape[1]
        data = np.concatenate((data[0:groupnumber,:], np.zeros((1,size)),
                                data[groupnumber:secondnum,:]/data[groupnumber:secondnum,:].max(axis = 0).reshape((1,size)), 
                                data[secondnum:,:]/data[secondnum:,:].max(axis = 0).reshape((1,size))), axis = 0)

        
        ax = sns.heatmap(data, cmap=cmap, xticklabels=False, yticklabels= yticklabels)

    elif rna == None:

        yticklabels = list(adata_ccf.obs.index)

        data = np.log2(np.array(adata_ccf.X.todense())+1)
        data = data/data.sum(axis = 0)
        data = data[:,np.lexsort(data[::-1])]

        ax = sns.heatmap(data, cmap=cmap, xticklabels=False, yticklabels= yticklabels)


    ax.set_title(title, fontsize=16)

    if save != False:
        if save == True:
            save = f"heatmap" + '.png'
        plt.savefig(save, bbox_inches='tight')


    plt.show()

    import matplotlib as mpl
    mpl.rc_file_defaults()
