import os
from typing import Literal, Optional

import numpy as np
import pandas as pd

_reference2 = Optional[Literal["hg38", "mm10", "sacCer3"]]


def call_motif(
    peaks_path: Optional[str] = None,
    peaks_frame: Optional[pd.DataFrame] = None,
    save_name: Optional[str] = None,
    reference: _reference2 = "hg38",
    save_homer: Optional[str] = None,
    size: int = 1000,
    homer_path: Optional[str] = None,
    motif_length: Optional[int] = None,
    num_cores: int = 3,
    denovo: bool = False,
):

    """\
    Call motif by `HOMER <http://homer.ucsd.edu/homer/ngs/peakMotifs.html>`__ and :cite:`heinz2010simple`.
    Please make sure HOMER is installed along with the genome data.

    :param peaks_path:
        pd.DataFrame with the path to the peak data.
        If this is provided, it will not consider peaks_frame and save_name.
    :param peaks_frame:
        pd.DataFrame with the first three columns as chromosome, start and end.
    :param save_name:
        The name of a saved peak file. Only used when peaks_frame is provided and peaks_path is not provided.
    :param reference:
        reference of the annoatation data.
        Currently, only `'hg38'`, `'mm10'`, `'sacCer3'` are provided.
        Make sure the genome in HOMER is installed.
        Eg for mm10: `perl [path]/homer/.//configureHomer.pl -install mm10`
    :param save_homer:
        Where path and name of the annotation results will be saved.
        If 'None' it will be saved to "Homerresult/peaks_name"
    :param size:
        The size of the region for motif finding.
        This is one of the most important parameters and also a source of confusion for many.
        If you wish to find motifs using your peaks using their exact sizes, use the option "-size given").
        However, for Transcription Factor peaks, most of the motifs are found +/- 50-75 bp from the peak center, making it better to use a fixed size rather than depending on your peak size.
    :param homer_path:
        The default uses the default path for Homer.
    :param motif_length:
        The default uses the default motif length for `HOMER <http://homer.ucsd.edu/homer/ngs/peakMotifs.html>`__.
        Specifies the length of motifs to be found.
    :param num_cores:
        Number of CPUs to use.
    :param deno:
        Whether to call denovo modif or not.

    :Examples:
    >>> import pycallingcards as cc
    >>> HCT116_SP1 = cc.datasets.SP1_K562HCT116_data(data="HCT116_SP1_qbed")
    >>> HCT116_brd4 = cc.datasets.SP1_K562HCT116_data(data="HCT116_brd4_qbed")
    >>> peak_data_HCT116 = cc.pp.callpeaks(HCT116_SP1, HCT116_brd4, method = "cc_tools", reference = "hg38",  window_size = 2000, step_size = 500,
            pvalue_cutoffTTAA = 0.001, pvalue_cutoffbg = 0.1, lam_win_size = None,  pseudocounts = 0.1, record = True, save = "peak_HCT116_test.bed")
    >>> cc.tl.call_motif("peak_HCT116_test.bed",reference ="hg38",save_homer = "Homer/peak_HCT116_test", homer_path = "/ref/rmlab/software/homer/bin")


    """

    if type(peaks_path) == str:

        print("Peak data " + peaks_path + " is used here.")
        name = peaks_path

    elif peaks_path == None:

        if type(peaks_frame) == pd.DataFrame:

            if save_name == None:
                print(
                    "There is no save_name, it will save to temp_Homer_trial.bed and then delete."
                )
                name = "temp_Homer_trial.bed"

            elif type(save_name) == str:
                if save_name[-4:] != ".bed":
                    save_name = save_name + ".bed"
                name = save_name

            peaks_frame["ID"] = peaks_frame.index
            peaks_frame[["Chr", "Start", "End", "ID"]].to_csv(
                name, sep="\t", header=None, index=None
            )

        else:
            raise ValueError("Please input correct form of peaks_frame")

    else:
        raise ValueError("Please input correct form of peaks_path")

    if save_homer == None:
        if save_name == None:
            save_homer = "Homerresult"
        else:
            save_homer = "Homerresult_" + save_name

    if homer_path == None:
        cmd = (
            "perl "
            + "/findMotifsGenome.pl "
            + name
            + " "
            + reference
            + " "
            + save_homer
            + " -size "
            + str(size)
            + " "
        )
    else:
        cmd = (
            "perl "
            + homer_path
            + "/findMotifsGenome.pl "
            + name
            + " "
            + reference
            + " "
            + save_homer
            + " -size "
            + str(size)
            + " "
        )

    if motif_length != None:
        if type(motif_length) == int:
            cmd = cmd + "-len " + str(motif_length)

    if num_cores != None:
        if type(num_cores) == int:
            cmd = cmd + "-p " + str(num_cores)

    if denovo == False:
        cmd = cmd + " -nomotif"

    os.system(cmd)

    if peaks_path == None and save_name == None:
        os.system("rm temp_Homer_trial.bed")

    print("Finished!")


def compare_motif(
    motif_path1: str,
    motif_path2: str,
    qvalue_cutoff: float = 0.05,
    pvalue_cutoff: float = 0.05,
) -> pd.DataFrame:

    """\
    Compare the motifs from the motif results of two groups from `HOMER <http://homer.ucsd.edu/homer/ngs/peakMotifs.html>`__ and :cite:`heinz2010simple`.
    Here, we will find the motif in group1 but not in group2.

    :param peaks_path1:
        The path of motif result for the first dataset.
    :param peaks_path2:
        The path of motif result for the second datast.
    :param qvalue_cutoff:
        The cutoff for q-value (Benjamini).
    :param pvalue_cutoff:
        The cutoff for p-value.

    """

    p1 = pd.read_csv(motif_path1 + "/knownResults.txt", sep="\t")
    p2 = pd.read_csv(motif_path2 + "/knownResults.txt", sep="\t")
    p1_list = list(
        p1[
            (p1["q-value (Benjamini)"] < qvalue_cutoff)
            & (p1["P-value"] < pvalue_cutoff)
        ]["Motif Name"]
    )
    p2_list = list(
        p2[
            (p2["q-value (Benjamini)"] < qvalue_cutoff)
            & (p2["P-value"] < pvalue_cutoff)
        ]["Motif Name"]
    )

    return p1[p1["Motif Name"].isin(list(set(p1_list) - set(p2_list)))]
