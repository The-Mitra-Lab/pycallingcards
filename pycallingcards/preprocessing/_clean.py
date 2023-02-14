import pandas as pd
from anndata import AnnData


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
        "chrI",
        "chrII",
        "chrIII",
        "chrIV",
        "chrV",
        "chrVI",
        "chrVII",
        "chrVIII",
        "chrIX",
        "chrX",
        "chrXI",
        "chrXII",
        "chrXIII",
        "chrXIV",
        "chrXV",
        "chrXVI",
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


def filter_adata_sc(
    adata_cc: AnnData,
    adata: AnnData,
) -> AnnData:

    """\
    The function would make sure the adata_cc only keeps the cells from adata.

    :param adata_cc:
        Anndata object for scCC data
    :param adata:
        Anndata object for scRNA-seq data.


    :Returns:
        adata_cc object after filtering.

    :example:
    >>> import pycallingcards as cc
    >>> cc_data = cc.datasets.mousecortex_data(data="qbed")
    >>> peak_data = cc.pp.callpeaks(cc_data, method = "test", reference = "mm10",  record = True)
    >>> barcodes = cc.datasets.mousecortex_data(data="barcodes")
    >>> adata_cc = cc.pp.makeAnndata(cc_data, peak_data, barcodes)
    >>> adata = cc.datasets.mousecortex_data(data="RNA")
    >>> adata_cc = cc.pp.filter_adata_sc(adata_cc,adata)


    """

    adata_cc = adata_cc[adata.obs.index]

    if adata_cc.shape[0] != adata.shape[0]:
        raise ValueError("Please check your obs index again")

    if (adata_cc.obs.index != adata_cc.obs.index).sum() > 0:
        raise ValueError("Please check your obs index again")

    return adata_cc
