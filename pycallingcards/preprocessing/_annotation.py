from typing import Literal, Optional

import numpy as np
import pandas as pd

_reference2 = Optional[Literal["hg38", "mm10", "sacCer3"]]


def combine_annotation(
    peak_data: pd.DataFrame,
    peak_annotation: pd.DataFrame,
) -> pd.DataFrame:

    """\
    Combine peak information and annotation information.

    :param peak_data:
        pd.DataFrame with the first three columns as chromosome, start and end.
        Following columns indicate extra information of the peak.
    :param peak_annotation:
        pd.DataFrame with first three columns as chromosome, start and end.
        Folloing columns indicate the annotation of the peak.


    :Returns:
        pd.DataFrame with the first three columns as chromosome, start and end. Following columns are from peak_data and peak_annotation.

    :Notes: The first three columns for peak_data and peak_annotation should be exactly the same.

    :Example:
    >>> import pycallingcards as cc
    >>> qbed_data = cc.datasets.mousecortex_data(data="qbed")
    >>> peak_data = cc.pp.callpeaks(qbed_data, method = "test", reference = "mm10", record = True)
    >>> peak_annotation = cc.pp.annotation(peak_data, reference = "mm10")
    >>> peak_annotation = cc.pp.combine_annotation(peak_data,peak_annotation)

    """

    peak = peak_data.iloc[:, [0, 1, 2]]

    annotation = peak_annotation.iloc[:, [0, 1, 2]]

    if ((peak == annotation).all()).all():
        return pd.concat([peak_data, peak_annotation.iloc[:, 3:]], axis=1)
    else:
        print("The peaks for peak data and anotation data are not the same")


def annotation(
    peaks_frame: pd.DataFrame = None,
    peaks_path: str = None,
    reference: _reference2 = "hg38",
    save_annotation: str = None,
    bedtools_path: str = None,
    refGene: pd.DataFrame = None,
) -> pd.DataFrame:

    """\
    Annotate the peak data using `bedtools <https://bedtools.readthedocs.io/en/latest/index.html>`__ :cite:`quinlan2010bedtools`.

    :param peaks_frame:
        pd.DataFrame with the first three columns as chromosome, start and end.
        Will not be used if peak_path is provided.
    :param peaks_path:
        The path to the peak data.
        An external program is used in this function so peak_path is preferred over peaks_frame.
    :param reference:  Default is `'hg38'`.
        Reference of the annotation data.
        Currently, only `'hg38'`, `'mm10'`, `'sacCer3'` are provided.
    :param save_annotation:
        The path and name of the annotation results would be saved.
    :param bedtools_path:
        Default uses the default path for bedtools.
    :param refGene:
        Default is None. If None, it would use the saved refgenome according to the reference provided. Else, if a Dataframe is input, it will use the refgenome provided.
        Please note that the DataFrame should follow the same format as this "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.hg38.Sorted.bed" which contains 6 columns in total (Chrom, Start, End, Refseq, Name, Direction)

    :Returns:
        pd.DataFrame with the first three columns as chromosome, start and end. Following the columns is the peak_annotation.

        | **Chr** - The chromosome of the peak.
        | **Start** - The start point of the peak.
        | **End** - The end point of the peak.
        | **Nearest Refseq1** - The Refseq of the closest gene.
        | **Nearest Refseq2** - The name of the second closest gene.
        | **Gene Name1** - The name of the closest gene.
        | **Gene Name2** -  The name of the second closest gene.

    :Example:
    >>> import pycallingcards as cc
    >>> qbed_data = cc.datasets.mousecortex_data(data="qbed")
    >>> peak_data = cc.pp.callpeaks(qbed_data, method = "CCcaller", reference = "mm10", record = True)
    >>> peak_annotation = cc.pp.annotation(peak_data, reference = "mm10")

    """

    import pybedtools

    print(
        "In the bedtools method, we would use bedtools in the default path. Set bedtools path by 'bedtools_path' if needed."
    )

    if bedtools_path != None:
        pybedtools.helpers.set_bedtools_path(path=bedtools_path)

    if peaks_path != None:
        peaks_bed = pybedtools.BedTool(peaks_path)
    elif type(peaks_frame) == pd.DataFrame:
        peaks_bed = pybedtools.BedTool.from_dataframe(peaks_frame)
    else:
        print("Please input a valid peak.")

    if isinstance(refGene, type(None)):

        if reference == "hg38":

            import os

            from appdirs import user_cache_dir

            PYCALLINGCARDS_CACHE_DIR = user_cache_dir("pycallingcards")

            if not os.path.exists(PYCALLINGCARDS_CACHE_DIR):
                os.makedirs(PYCALLINGCARDS_CACHE_DIR)

            filename = os.path.join(PYCALLINGCARDS_CACHE_DIR, "refGene.hg38.Sorted.bed")

            if os.path.exists(filename) == False:
                from urllib import request

                URL = "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.hg38.Sorted.bed"
                response = request.urlretrieve(URL, filename)

            refGene_filename = pybedtools.BedTool(filename)

        elif reference == "mm10":

            import os

            from appdirs import user_cache_dir

            PYCALLINGCARDS_CACHE_DIR = user_cache_dir("pycallingcards")

            if not os.path.exists(PYCALLINGCARDS_CACHE_DIR):
                os.makedirs(PYCALLINGCARDS_CACHE_DIR)

            filename = os.path.join(PYCALLINGCARDS_CACHE_DIR, "refGene.mm10.Sorted.bed")

            if os.path.exists(filename) == False:
                from urllib import request

                URL = "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.mm10.Sorted.bed"
                response = request.urlretrieve(URL, filename)

            refGene_filename = pybedtools.BedTool(filename)

        elif reference == "sacCer3":

            import os

            from appdirs import user_cache_dir

            PYCALLINGCARDS_CACHE_DIR = user_cache_dir("pycallingcards")

            if not os.path.exists(PYCALLINGCARDS_CACHE_DIR):
                os.makedirs(PYCALLINGCARDS_CACHE_DIR)

            filename = os.path.join(
                PYCALLINGCARDS_CACHE_DIR, "refGene.sacCer3.Sorted.bed"
            )

            if os.path.exists(filename) == False:
                from urllib import request

                URL = "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.sacCer3.Sorted.bed"
                response = request.urlretrieve(URL, filename)

            refGene_filename = pybedtools.BedTool(filename)

    else:
        refGene_filename = pybedtools.BedTool(refGene)

    temp_annotated_peaks = peaks_bed.closest(
        refGene_filename, D="ref", t="first", k=2, d=True
    )

    temp_annotated_peaks = pd.read_table(temp_annotated_peaks.fn, header=None).iloc[
        :, [0, 1, 2, -4, -3, -2, -1]
    ]
    temp_annotated_peaks = temp_annotated_peaks
    temp_annotated_peaks.columns = [
        "Chr",
        "Start",
        "End",
        "Nearest Refseq",
        "Gene Name",
        "Direction",
        "Distance",
    ]
    temp_annotated_peaks1 = temp_annotated_peaks.iloc[::2].reset_index()
    temp_annotated_peaks1 = temp_annotated_peaks1[
        [
            "Chr",
            "Start",
            "End",
            "Nearest Refseq",
            "Gene Name",
            "Direction",
            "Distance",
        ]
    ].rename(
        columns={
            "Nearest Refseq": "Nearest Refseq1",
            "Gene Name": "Gene Name1",
            "Direction": "Direction1",
            "Distance": "Distance1",
        }
    )
    temp_annotated_peaks2 = temp_annotated_peaks.iloc[1::2].reset_index()
    temp_annotated_peaks2 = temp_annotated_peaks2[
        ["Nearest Refseq", "Gene Name", "Direction", "Distance"]
    ].rename(
        columns={
            "Nearest Refseq": "Nearest Refseq2",
            "Gene Name": "Gene Name2",
            "Direction": "Direction2",
            "Distance": "Distance2",
        }
    )

    finalresult = pd.concat([temp_annotated_peaks1, temp_annotated_peaks2], axis=1)

    if save_annotation != None:
        finalresult.to_csv(save_annotation, index=None, sep="\t")

    return pd.concat([temp_annotated_peaks1, temp_annotated_peaks2], axis=1)
