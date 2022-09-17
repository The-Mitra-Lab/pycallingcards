import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
import anndata as ad
from scipy.sparse import csr_matrix
from anndata import AnnData



def _myFuncsorting(e):
    try:
        return int(e.split('_')[0][3:])
    except :
        return int(ord(e.split('_')[0][3:]))

def makeAnndata(
    insertions: pd.DataFrame, 
    peaks: pd.DataFrame, 
    barcodes: pd.DataFrame,
    length: int = 3
    ) -> AnnData:

    """\
    Make cell by peak anndata for calling cards.

    :param insertions:
        pd.DataFrame with first three columns: chromosome, start, end, reads number, diretion, barcodes. 
        Chromosome, start, end, barcodes are actually needed.
    :param peaks:
        pd.DataFrame with first three columns: chromosome, start, end. Other information may go after these.
    :param barcodes:
        pd.DataFrame with all barcodes information.


    :Returns:
        Annotated data matrix, where observations/cells are named by their barcode and 
        variables/peaks by Chr_Start_End. Stores the following information.
        
        | **anndata.AnnData.X** -  The data matrix is stored
        | **anndata.AnnData.obs_names** -  Cell names
        | **anndata.AnnData.var_names** -  Peak names
        | **anndata.AnnData.var['peak_ids']** -  Peak information fron the origin file
        | **anndata.AnnData.var['feature_types']** -  Feature types

   
    :Example:
    >>> import pycallingcards as cc
    >>> ccf_data = cc.datasets.mousecortex_ccf()
    >>> peak_data = cc.pp.callpeaks(ccf_data, method = "test", reference = "mm10",  maxbetween = 2000,pvalue_cutoff = 0.01, lam_win_size = 1000000,  pseudocounts = 1, record = True)
    >>> barcodes = cc.datasets.mousecortex_barcodes()
    >>> adata_ccf = cc.pp.makeAnndata(ccf_data, peak_data, barcodes)

    """
    
    barcodesorigin = barcodes.copy()
    barcodes = list(barcodes.iloc[:,0])
    barcodes_dict = {}
    for i in range(len(barcodes)):
        barcodes_dict[barcodes[i]] = i

    # peaks to order
    
    peaks1 = peaks.copy()
    peaks1["name"] = peaks1.iloc[:,[0,1,2]].T.astype(str).agg('_'.join)
 
    peak_name_unique = list()
    
    for number in list(peaks1["name"]):
        if number not in peak_name_unique:
            peak_name_unique.append(number)
    peak_name_unique.sort(key=_myFuncsorting)

    peak_name_dict = {}
    for i in range(len(peak_name_unique)):
        peak_name_dict[peak_name_unique[i]] = i
        
  
    # create an empty matrix to store peaks * cell
    cellpeaks = lil_matrix((len(peak_name_unique),len(barcodes)), dtype=np.float32)
    
    
    #pairing
    chrolist = list(peaks1.iloc[:,0].unique())

    peaksf = peaks1.iloc[:,[0,1,2,-1,-1]].to_numpy()

    for i in range(len(peaksf)):
        peaksf[i,4] = peak_name_dict[peaksf[i,3]]
    

    insertions1 = insertions.copy()
    
    insertions1[6] = insertions1[5].apply(lambda x: barcodes_dict[x])
    ccff = insertions1.iloc[:,[0,1,5,6]].to_numpy()

    
    for chro in chrolist:

        peaksfchr = peaksf[peaksf[:,0] == chro]
        ccffchr = ccff[ccff[:,0] == chro]

        for i in range(len(peaksfchr)):

            ptemp = ccffchr[(ccffchr[:,1]>= (peaksfchr[i,1] - length))  & (ccffchr[:,1]<= (peaksfchr[i,2] ))][:,3]

            for j in ptemp:
                cellpeaks[peaksfchr[i,4],j] += 1

    peaks1 = peaks1.set_index(keys = "name")
    peaks1 = peaks1.loc[peak_name_unique]

    
    adata = ad.AnnData(csr_matrix(cellpeaks).T,obs= barcodesorigin.set_index(keys =barcodesorigin.columns[0]),var = peaks1)
    
    return adata
        
 