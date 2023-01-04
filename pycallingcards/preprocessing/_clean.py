import pandas as pd


def clean_qbed(
    qbed: pd.DataFrame,
    chrom: list = [
        "chr1",
        "chr10",
        "chr11",
        "chr12",
        "chr13",
        "chr14",
        "chr15",
        "chr16",
        "chr17",
        "chr18",
        "chr19",
        "chr2",
        "chr20",
        "chr21",
        "chr22",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr8",
        "chr9",
        "chrX",
        "chrY",
    ],
) -> pd.DataFrame:

    """\
    Clean qbed with some extra chromosomes. Only preserve insertions in chrom.

    :param qbed:
        qbed file.
    :param chrom:
        list of valid chromosomes.


    :Returns:
        pd.DataFrame for the cleaned qbed file.

    :example:
    >>> import pycallingcards as cc
    >>> qbed = cc.rd.read_qbed("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/SP1_P10.txt")
    >>> qbed = cc.pp.clean_qbed(qbed)

    """

    return qbed[qbed["Chr"].isin(chrom)]
