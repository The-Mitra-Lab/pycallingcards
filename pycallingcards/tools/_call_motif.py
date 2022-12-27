import numpy as np
import pandas as pd
from typing import  Optional, Literal
import os 

_reference2 = Optional[Literal["hg38","mm10","sacCer3"]]


def call_motif(
    peaks_path: str = None, 
    peaks_frame: pd.DataFrame = None, 
    save_name: str = None, 
    reference: _reference2  = "hg38", 
    save_homer: str = None, 
    size: int = 1000,
    homer_path: str = None,
    motif_length: int = None,
    num_cores: int = 3
    ):

    """\
    Call motif by `HOMER <http://homer.ucsd.edu/homer/>`__ [Heinz10]_.
    Please make sure HOMER is installed along with the genome data.

    :param peaks_path: Default is `None`.
        pd.DataFrame with the path to the peak data. 
        If this is provided, it would not consider peaks_frame and save_name.
    :param peaks_frame: Default is `None`.
        pd.DataFrame with the first three columns as chromosome, start and end.
    :param save_name: Default is `None`.
        The name of saved peak file. Only used when peaks_frame is provided and peaks_path is not provided.
    :param reference:  Default is `'hg38'`.
        reference of the annoatation data.
        Currently, `'hg38'`, `'mm10'`, `'sacCer3'` are provided only.
        Make sure the installed HOMER 
        Eg for mm10: `perl [path]/homer/.//configureHomer.pl -install mm10`
    :param save_homer:  Default is `None`.
        The path and name of the annotation results would be saved.
        If 'None' it would be saved to "Homerresult/peaks_name"
    :param size:  Default is `800`.
        The size of the region for motif finding.
        This is one of the most important parameters and also a source of confusion for many.  
        If you wish to find motifs using your peaks using their exact sizes, use the option "-size given").  
        However, for Transcription Factor peaks, most of the motifs are found +/- 50-75 bp from the peak center, making it better to use a fixed size rather than depend on your peak size.
    :param homer_path: Default is `None`.
        Default uses the default path for Homer.
    :param motif_length: Default is `None`.
        Default uses the default Motif length for HOMER <http://homer.ucsd.edu/homer/ngs/peakMotifs.html>.
        Specifies the length of motifs to be found. 
    :param num_cores: Default is `3`.
        Number of CPUs to use.
    
    """
        
        
    if type(peaks_path) == str:

        print("Peak data " + peaks_path + " would be used here.")
        name = peaks_path

    elif peaks_path == None:

        if type(peaks_frame) == pd.DataFrame:

            if save_name == None: 
                print("There is no save_name, it will save to temp_Homer_trial.bed and then delete.")
                name = "temp_Homer_trial.bed"
                
            elif type(save_name) == str:
                if save_name[-4:] != ".bed":
                    save_name = save_name + ".bed"
                name = save_name
            peaks_frame.to_csv(name,sep ="\t",header = None, index = None)

        else:
            raise ValueError('Please input correct form of peaks_frame')
    
    else:
        raise ValueError('Please input correct form of peaks_path')
    
    if save_homer == None:
        if save_name == None:
            save_homer = "Homerresult" 
        else:
            save_homer = "Homerresult_" + save_name

    if homer_path == None:
        cmd = "perl " + "/findMotifsGenome.pl " + name + " " + reference + " " + save_homer + " -nomotif -size " + str(size) + " "
    else:
        cmd = "perl " + homer_path + "/findMotifsGenome.pl " + name + " " + reference + " " + save_homer + " -nomotif -size " + str(size) + " "
        

    if motif_length != None:
        if type(motif_length) == int:
            cmd = cmd + "-len " + str(motif_length)
    
    if num_cores != None:
        if type(num_cores) == int:
            cmd = cmd + "-p " + str(num_cores)
           
    os.system(cmd)

    if peaks_path == None and save_name == None:
        os.system("rm temp_Homer_trial.bed")
    
    print("Finished!")



