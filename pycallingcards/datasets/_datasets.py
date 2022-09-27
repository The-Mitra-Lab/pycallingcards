import os

import pandas as pd
import scanpy as sc
import anndata as ad
from appdirs import user_cache_dir

PYCALLINGCARDS_CACHE_DIR = user_cache_dir("pycallingcards")
if not os.path.exists(PYCALLINGCARDS_CACHE_DIR):
    os.makedirs(PYCALLINGCARDS_CACHE_DIR)

def mousecortex_ccf() -> pd.DataFrame:

    """\
    Mouse cortex single-cell calling cards htops data.

    :reference: `Self-Reporting Transposons Enable Simultaneous Readout of Gene Expression and Transcription Factor Binding in Single Cells` `[Moudgil20] <https://doi.org/10.1016/j.cell.2020.06.037>`__,
    """

    ccf_data = pd.read_csv("https://github.com/JuanruMaryGuo/CCFtools/releases/download/data/Mouse-Cortex.ccf",
                        sep = "\t",  header = None)
    return ccf_data

def mousecortex_barcodes() -> pd.DataFrame:

    """\
    Mouse cortex single-cell calling cards barcodes data.

    :reference: `Self-Reporting Transposons Enable Simultaneous Readout of Gene Expression and Transcription Factor Binding in Single Cells` `[Moudgil20] <https://doi.org/10.1016/j.cell.2020.06.037>`__,
    """

    barcodes = pd.read_csv("https://github.com/JuanruMaryGuo/CCFtools/releases/download/data/Mouse-Cortex-scCC_barcodes.csv",
                        sep = ",")
    return barcodes

def mousecortex_RNA() -> ad.AnnData:

    """\
    Mouse cortex single-cell calling cards RNA adata.

    :reference: `Self-Reporting Transposons Enable Simultaneous Readout of Gene Expression and Transcription Factor Binding in Single Cells` `[Moudgil20] <https://doi.org/10.1016/j.cell.2020.06.037>`__,
    """

    filename = os.path.join(PYCALLINGCARDS_CACHE_DIR, "Mouse-Cortex_RNA.h5ad")
    url = "https://github.com/JuanruMaryGuo/CCFtools/releases/download/data/Mouse-Cortex_RNA.h5ad"

    adata = sc.read(filename, backup_url=url)
    return adata


def mousecortex_CCF() -> ad.AnnData:

    """\
    Mouse cortex single-cell calling cards CCF adata.

    :reference: `Self-Reporting Transposons Enable Simultaneous Readout of Gene Expression and Transcription Factor Binding in Single Cells` `[Moudgil20] <https://doi.org/10.1016/j.cell.2020.06.037>`__,
    """

    filename = os.path.join(PYCALLINGCARDS_CACHE_DIR, "Mouse-Cortex_CCF.h5ad")
    url = "https://github.com/JuanruMaryGuo/CCFtools/releases/download/data/Mouse-Cortex_CCF.h5ad"

    adata = sc.read(filename, backup_url=url)
    return adata