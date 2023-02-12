from functools import reduce

import numpy as np
from pybedtools import BedTool

# numpy log10 float max
FLOAT_MAX = np.finfo(np.float64).max
LOG10_FLOAT_MAX = np.log10(FLOAT_MAX)


# Fast method for getting number of lines in a file
# For BED files, much faster than calling len() on file
# From https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
def file_len(fname):
    """Fast method for getting number of lines in a file. For BED files, much faster than calling len() on a BedTool object. From https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python

    Parameters
    ----------
    fname: str
        Input (text) filename

    Returns
    -------
    length: int
        Length of fname
    """

    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def getChromosomesInDF(df):
    """Helper function to get a list of unique chromsomes in a ``pandas`` DataFrame.

    Parameters
    ----------
    df: ``pandas`` DataFrame
        Input genomic data (e.g BED, qBED, CCF) as a DataFrame

    Returns
    -------
    chroms: list
        List of chromosomes
    """

    return reduce(lambda l, x: l if x in l else l + [x], df["chrom"], [])


def isSortedBEDObject(bed_object):
    """Tests whether a BedTool object is sorted.

    Parameters
    ----------
    bed_object: BedTool object
        Input data as a BedTool object

    Returns
    -------
    is_sorted: bool
    """

    # Convert BedTool object to pandas DataFrame
    df = bed_object.to_dataframe()
    # First, check that chrom is in sorted order
    if df["chrom"].is_monotonic:
        # If so, check that the start coordinates are in order
        chroms = getChromosomesInDF(df)
        for c in chroms:
            if not df[df["chrom"] == c]["start"].is_monotonic:
                return False
        return True
    return False


def isSortedBEDFile(bed_file_path):
    """Wrapper function to feed filepaths isSortedBEDObject.

    Parameters
    ----------
    bed_file_path: str
        Path to BED/qBED/CCF data file

    Returns
    -------
    is_sorted: bool
    """

    return isSortedBEDObject(BedTool(bed_file_path))
