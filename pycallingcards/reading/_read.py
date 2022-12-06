import pandas as pd
from typing import List

def read_ccf(
    filename: str
    ) -> pd.DataFrame:

    """\
    Read .ccf file.

    :param filename: 
        Path to the ccf file.

    :Returns:
        pd.DataFrame for the ccf file.

    :example:
    >>> import pycallingcards as cc
    >>> ccf = cc.rd.read_ccf("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/SP1_P10.txt")

       
    """

    return pd.read_csv(filename, sep = "\t",  header = None, names =  ["Chr", "Start", "End", "Reads", "Direction", "Barcodes"])

def combine_ccf(
    filelist: list, 
    name: List = ["Chr", "Start"]
    ) -> pd.DataFrame:

    """\
    Read .ccf file.

    :param filelist: 
        List containing all the pd.DataFrames need to be combined.
    :param name: 
        Name of the first two colums. Default is ["Chr", "Start"].

    :Returns:
        pd.DataFrame after combined.

    :example:
    >>> import pycallingcards as cc
    >>> P10 = cc.rd.read_ccf("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/SP1_P10.txt")
    >>> P28 = cc.rd.read_ccf("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/SP1_P28.txt")
    >>> ccf = cc.rd.combine_ccf([P10,P28])
    """

    return pd.concat(filelist).sort_values(by=name).reset_index(drop = True)

def read_rna(
    filename: str
):

    """\
    Read RNA file which column index is sample names and row index is gene names.

    :param filename: 
        Path to RNA file.

    :Returns:
        pd.DataFrame for the RNA file.

    :example:
    >>> import pycallingcards as cc
    >>> rna = cc.rd.read_rna("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/dmso_MF.csv")

    """

    return pd.read_csv("dmso.csv", index_col = 0)