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
    Mouse cortex single-cell calling cards htops data [Moudgil20]_.

    """

    ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/Mouse-Cortex.ccf",
                        sep = "\t",  header = None)
    return ccf_data

def mousecortex_barcodes() -> pd.DataFrame:

    """\
    Mouse cortex single-cell calling cards barcodes data [Moudgil20]_.

    """

    barcodes = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/Mouse-Cortex-scCC_barcodes.csv",
                        sep = ",")
    return barcodes

def mousecortex_RNA() -> ad.AnnData:

    """\
    Mouse cortex single-cell calling cards RNA adata [Moudgil20]_.

    """

    filename = os.path.join(PYCALLINGCARDS_CACHE_DIR, "Mouse-Cortex_RNA.h5ad")
    url = "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/Mouse-Cortex_RNA.h5ad"

    adata = sc.read(filename, backup_url=url)
    return adata


def mousecortex_CCF() -> ad.AnnData:

    """\
    Mouse cortex single-cell calling cards CCF adata [Moudgil20]_.
    
    """

    filename = os.path.join(PYCALLINGCARDS_CACHE_DIR, "Mouse-Cortex_CCF.h5ad")
    url = "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/Mouse-Cortex_CCF.h5ad"

    adata = sc.read(filename, backup_url=url)
    return adata