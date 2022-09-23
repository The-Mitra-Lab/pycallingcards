import numpy as np
import pandas as pd
from typing import Union, Optional, List, Sequence, Iterable, Mapping, Literal, Tuple

_reference2 = Optional[Literal["hg38","mm10"]]
_method_annotation = Optional[Literal["homer","bedtools"]]


def combine_annotation(
    peak_data: pd.DataFrame, 
    peak_annotation: pd.DataFrame, 
    ) ->  pd.DataFrame:

    """\
    Combine peak information and annotation information

    :param peak_data:
        pd.DataFrame with first three columns as: chromosome, start, end.
        The folloing columns indecate the extra information of the peak.
    :param peak_annotation:
        pd.DataFrame with first three columns as: chromosome, start, end.
        The folloing columns indecate the annotation of the peak.


    :Returns:
        pd.DataFrame with first three columns as: chromosome, start, end and the following columns from peak_data and peak_annotation.

    :Notes: The first three columns for peak_data and peak_annotation should be exactly the same.

    :Example:
    >>> import pycallingcards as cc
    >>> ccf_data = cc.datasets.mousecortex_data(data="ccf")
    >>> peak_data = cc.pp.callpeaks(ccf_data, method = "test", reference = "mm10",  maxbetween = 2000,pvalue_cutoff = 0.01,
                  lam_win_size = 1000000,  pseudocounts = 1, record = True)
    >>> peak_annotation = cc.pp.annotation(peak_data, reference = "mm10", method = "bedtools")
    >>> peak_annotation = cc.pp.combine_annotation(peak_data,peak_annotation)
   
    """
    
    peak =  peak_data.iloc[:,[0,1,2]]

        
    annotation =  peak_annotation.iloc[:,[0,1,2]]

        
    if ((peak == annotation).all()).all():
        return pd.concat([peak_data, peak_annotation.iloc[:,3:]],axis = 1)
    else:
        print("The peaks for peak data and anotation data are not the same")
        
def annotation(
    peaks_frame: pd.DataFrame = None, 
    peaks_path: str = None, 
    reference: _reference2  = "hg38", 
    method: _method_annotation = "homer", 
    save_peak: str = None, 
    save_annotation: str = None, 
    bedtools_path: str = None
    ) -> pd.DataFrame:
    
    """\
    Combine the peak information and the annotation information

    :param peaks_frame:
        pd.DataFrame with first three columns as: chromosome, start, end.
        Will not be used if peak_path is pervided.
    :param peaks_path:
        The path to the peak data. 
        External program would be used in this function so peak_path is perferred to peaks_frame.
    :param reference:
        reference of the annoatation data. Default is `'hg38'`.
        Currently, `'hg38'` and `'mm10'` are provided only.
    :param method:
        Methods for annotation. 
        Two different methods are provided. See `'bedtools' <https://bedtools.readthedocs.io/en/latest/index.html>`__ [Quinlan10]_ and
        `'homer' <http://homer.ucsd.edu/homer/>`__ [Heinz10]_ .
        Default is `'homer'`.
    :param save_peak:
        if peaks_path is not provided, peaks_frame would be saved. This is the path and name peaks_frame would be saved.
        Default is `'None'` and it would not be saved.
    :param save_annotation:
        The path and name of the annotation results would be save.
        Default is `'None'` and it would not be saved.
    :param bedtools_path:
        The bedtools path if method = `'bedtools'`.
        Default is `'None'` and it uses the default path for bedtools.

    :Returns:
        pd.DataFrame with first three columns as: chromosome, start, end and the following columns are the peak_annotation.

        | **Chr** - The chromosome of the peak.
        | **Start** - The start point of the peak.
        | **End** - The end point of the peak.
        | **Nearest Refseq1/ Nearest Refseq** - The Refseq of the closest gene.
        | **Nearest Refseq2** - The name of the second closest gene.
        | **Gene Name1/ Gene Name** - The name of the closest gene.
        | **Gene Name2** -  The name of the second closest gene.

    :Notes:
        Method `'bedtools'` would output two cloest genes and  `'homer'` would output one cloest genes.

    :Example:
    >>> import pycallingcards as cc
    >>> ccf_data = cc.datasets.mousecortex_data(data="ccf")
    >>> peak_data = cc.pp.callpeaks(ccf_data, method = "test", reference = "mm10",  maxbetween = 2000,pvalue_cutoff = 0.01, lam_win_size = 1000000,  pseudocounts = 1, record = True)
    >>> peak_annotation = cc.pp.annotation(peak_data, reference = "mm10", method = "bedtools")
    
    """

    
    save_peakinitial = save_peak 

    if method == "homer":
        
        import os
        print("In homer method, we would use homer under your default path.")
        
        save_annotationinitial = save_annotation
        if save_annotation ==  None:
            import random
            save_annotation = "save_annotation_"+str(random.randint(1,1000))+".gos"
            
        if save_peak ==  None:
            import random
            save_peak = "temp_peaks_"+str(random.randint(1,1000))+".gos"

        if  peaks_path != None:
            
            print("We would use peaks in the peaks_path file and temporary anotation file would be saved to save_annotation.")
            
            pd.read_csv(peaks_path,sep = "\t", header = None)[[0,1,2]].to_csv(save_peak, sep = "\t",
                                                                  header = None, index = None)
            
            #if save_annotation ==  None:
            #    import random
            #    save_annotation = "save_annotation_"+str(random.randint(1,1000))+".gos"
                
            cmd = "annotatePeaks.pl " + save_peak + " " + reference + " > " + save_annotation
            
        elif type(peaks_frame) == pd.DataFrame :
            print("Temporary peak would be saved to save_peak and temporary anotation file would be saved to save_annotation.")
            
            if save_peak ==  None:
                import random
                save_peak = "temp_peaks_"+str(random.randint(1,10000))+".gos"
                
                
            peaks_frame.iloc[:,[0,1,2]].to_csv(save_peak, sep = "\t", index = None, header = None)
            
            if save_annotation ==  None:
                import random
                save_annotation = "save_annotation_"+str(random.randint(1,1000))+".gos"

            cmd = "annotatePeaks.pl " + save_peak + " " + reference + " > " + save_annotation
            
        else:
             print("Please input a valid peak.")
            
        os.system(cmd)
        annotated_peaks = pd.read_table(save_annotation, index_col = 0).sort_values(["Chr","Start"])
        
        if save_annotationinitial ==  None:
            os.remove(save_annotation)
            
        if save_peakinitial == None and peaks_path==None:
            os.remove(save_peak)
            
        return annotated_peaks[["Chr","Start","End","Nearest Refseq","Gene Name"]]
    
    elif method == "bedtools":
        
        import pybedtools
        print("In the bedtools method, we would use bedtools in the default path. Set bedtools path by 'bedtools_path' if needed.")
        
        if bedtools_path != None:
            pybedtools.helpers.set_bedtools_path(path=bedtools_path)
            
            
        if  peaks_path != None:
            peaks_bed = pybedtools.BedTool(peaks_path)
            
            
        elif type(peaks_frame) == pd.DataFrame :
            
            
            if save_peak ==  None:
                
                import random
                save_peak = "temp_peaks_"+str(random.randint(1,1000))+".bed"
                
            peaks_frame.to_csv(save_peak, sep = "\t", index = None, header = None)
            peaks_bed = pybedtools.BedTool(save_peak)
            
        
        else :
            print("Please input a valid peak.")
            
            
        if reference == "hg38":

            import os 
            from appdirs import user_cache_dir
            PYCALLINGCARDS_CACHE_DIR = user_cache_dir("pycallingcards")

            if not os.path.exists(PYCALLINGCARDS_CACHE_DIR):
                os.makedirs(PYCALLINGCARDS_CACHE_DIR)
            
            filename = os.path.join(PYCALLINGCARDS_CACHE_DIR, "refGene.hg38.Sorted.bed")
            
            if  os.path.exists(filename) == False:
                from urllib import request
                URL = "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.hg38.Sorted.bed"
                response = request.urlretrieve( URL, filename)

            refGene_filename = pybedtools.BedTool(filename)

        elif reference == "mm10":

            import os 
            from appdirs import user_cache_dir
            PYCALLINGCARDS_CACHE_DIR = user_cache_dir("pycallingcards")

            if not os.path.exists(PYCALLINGCARDS_CACHE_DIR):
                os.makedirs(PYCALLINGCARDS_CACHE_DIR)
            
            filename = os.path.join(PYCALLINGCARDS_CACHE_DIR, "refGene.mm10.Sorted.bed")
            
            if  os.path.exists(filename) == False:
                from urllib import request
                URL = "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.mm10.Sorted.bed"
                response = request.urlretrieve( URL, filename)

            refGene_filename = pybedtools.BedTool(filename)

        temp_annotated_peaks = peaks_bed.closest(refGene_filename,D="ref",t="first",k=2)
        
        temp_annotated_peaks = pd.read_table(temp_annotated_peaks.fn, header = None).iloc[: ,[0,1,2,-4,-3 ]]
        temp_annotated_peaks = temp_annotated_peaks
        temp_annotated_peaks.columns = ["Chr","Start","End","Nearest Refseq","Gene Name"]
        temp_annotated_peaks1 = temp_annotated_peaks.iloc[::2].reset_index()
        temp_annotated_peaks1 = temp_annotated_peaks1[["Chr","Start","End","Nearest Refseq",
                                                       "Gene Name"]].rename(columns={"Nearest Refseq": "Nearest Refseq1", 
                                                                                     "Gene Name": "Gene Name1"})
        temp_annotated_peaks2 = temp_annotated_peaks.iloc[1::2].reset_index()
        temp_annotated_peaks2 = temp_annotated_peaks2[["Nearest Refseq",
                                                       "Gene Name"]].rename(columns={"Nearest Refseq": "Nearest Refseq2", 
                                                                                     "Gene Name": "Gene Name2"})

        finalresult = pd.concat([temp_annotated_peaks1, temp_annotated_peaks2],axis = 1)
        
        if save_annotation != None:
            finalresult.to_csv(save_annotation,index = None, sep = "\t")
            
        if save_peakinitial ==  None and peaks_path==None:
            import os
            os.remove(save_peak)
                
                
        return pd.concat([temp_annotated_peaks1, temp_annotated_peaks2],axis = 1)
