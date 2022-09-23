import os

import pandas as pd
import scanpy as sc
import anndata as ad
from appdirs import user_cache_dir
from typing import Union,Optional,Literal

PYCALLINGCARDS_CACHE_DIR = user_cache_dir("pycallingcards")
if not os.path.exists(PYCALLINGCARDS_CACHE_DIR):
    os.makedirs(PYCALLINGCARDS_CACHE_DIR)

_mouse_cortex = Optional[Literal["ccf","barcodes","RNA","CCF"]]

def mousecortex_data(
    data = _mouse_cortex
) -> Union[pd.DataFrame,ad.AnnData]:


    """\
    Mouse cortex single-cell calling cards  data [Moudgil20]_.

    :param data:
        Should be among `["ccf","barcodes","RNA","CCF"]`. No default.
        `ccf` reads the htop ccf file.
        `barcodes` reads the barcodes file.
        `RNA` reads the RNA anndata.
        `CCF` reads the CCF anndata.

    """

    if data == "ccf":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/Mouse-Cortex.ccf",
                            sep = "\t",  header = None)
        return ccf_data

    elif data == "barcodes":
        barcodes = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/Mouse-Cortex-scCC_barcodes.csv",
                        sep = ",")
        return barcodes

    elif data == "RNA":
        filename = os.path.join(PYCALLINGCARDS_CACHE_DIR, "Mouse-Cortex_RNA.h5ad")
        url = "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/Mouse-Cortex_RNA.h5ad"
        adata = sc.read(filename, backup_url=url)
        return adata

    elif data == "CCF":
        filename = os.path.join(PYCALLINGCARDS_CACHE_DIR, "Mouse-Cortex_CCF.h5ad")
        url = "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/Mouse-Cortex_CCF.h5ad"
        adata = sc.read(filename, backup_url=url)
        return adata

    else:
        avail_data = ["ccf","barcodes","RNA","CCF"]
        raise ValueError(f'data must be one of {avail_data}.')
            

_SP1 = Optional[Literal["experience_ccf","background_ccf","barcodes"]]

def SP1_K562HCT116_data(
    data = _SP1
) -> Union[pd.DataFrame,ad.AnnData]:


    """\
    Transcription factor SP1 is added to both K562 and HCT-116 cell lines seperately. 
    Wild type single-cell calling cards is also done in mixed K562 and HCT-116 cell lines [Moudgil20]_.

    :param data:
        Should be among `["experience_ccf","background_ccf","barcodes"]`. No default.
        `experience_ccf` reads the htop ccf file for the combined data of K562 and HCT-116 cell lines.
        `background_ccf` reads the htop ccf file for the data of mixed K562 and HCT-116 cell lines.
        `barcodes` reads the barcode file for the combined data of K562 and HCT-116 cell lines.


    """

    if data == "experience_ccf":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/K562HCT116_SP1.ccf",
                            sep = "\t",  header = None)
        return ccf_data
        
    if data == "background_ccf":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/K562HCT116_wild.ccf",
                            sep = "\t",  header = None)
        return ccf_data

    elif data == "barcodes":
        barcodes = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/K562HCT116_barcodes.csv",
                        sep = ",")
        return barcodes

    else:
        avail_data = ["experience_ccf","background_ccf","barcodes"]
        raise ValueError(f'data must be one of {avail_data}.')
            


def mouse_brd4_data(
    data = _SP1
) -> Union[pd.DataFrame,ad.AnnData]:


    """\
    Bulk Brd4 bindings for both male and female mouse [Kfoury21]_.

    :param data:
        Should be among `["Female_Brd4","Male_Brd4","Female_WT","Male_WT"]`. No default.
        `Female_Brd4` reads the htop ccf file for the female mouse Brd4 binding bulk data.
        `Male_Brd4` reads the htop ccf file for the male mouse Brd4 binding bulk data.
        `Female_WT` reads the htop ccf file for the female mouse wild type binding bulk data.
        `Male_WT` reads the htop ccf file for the male mouse wild type  binding bulk data.


    """

    if data == "Female_Brd4":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSE156678_F6_Brd4_pBC.ccf.txt",
                            sep = "\t",  header = None)
        return ccf_data
        
    if data == "Male_Brd4":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSE156678_M6_Brd4_pBC.ccf.txt",
                            sep = "\t",  header = None)
        return ccf_data

    if data == "Female_WT":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSE156678_F6_WT_pBC.ccf.txt",
                            sep = "\t",  header = None)
        return ccf_data
        
    if data == "Male_WT":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSE156678_M6_WT_pBC.ccf.txt",
                            sep = "\t",  header = None)
        return ccf_data

    else:
        avail_data = ["Female_Brd4","Male_Brd4","Female_WT","Male_WT"]
        raise ValueError(f'data must be one of {avail_data}.')
            
