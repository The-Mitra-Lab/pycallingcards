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

    :param data: `['ccf','barcodes','RNA','CCF']`. 
        `ccf` reads the ccf file.
        `barcodes` reads the barcodes file.
        | `RNA` reads the RNA anndata.
        | `CCF` reads the CCF anndata.

    """

    if data == "ccf":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/Mouse-Cortex.ccf",
                            sep = "\t",  header = None , names = ["Chr", "Start", "End", "Reads", "Direction", "Barcodes"])
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
            

_SP1 = Optional[Literal['HCT116_SP1_ccf','K562_SP1_ccf','HCT116_brd4_ccf','K562_brd4_ccf','barcodes']]

def SP1_K562HCT116_data(
    data = _SP1
) -> Union[pd.DataFrame,ad.AnnData]:


    """\
    Transcription factor SP1 is added to both K562 and HCT-116 cell lines seperately. 
    Wild-type single-cell calling cards is data also recorded in mixed K562 and HCT-116 cell lines [Moudgil20]_.

    :param data: `['HCT116_SP1_ccf','K562_SP1_ccf','HCT116_brd4_ccf','K562_brd4_ccf','barcodes']`.
        `experience_ccf` reads the ccf file for the combined data for K562 and HCT-116 cell lines.
        | `background_ccf` reads the ccf file for the data for mixed K562 and HCT-116 cell lines.
        | `barcodes` reads the barcode file for the combined data for K562 and HCT-116 cell lines.


    """

    if data == "HCT116_brd4_ccf":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSM4471646_HCT-116_HyPBase_scCC.ccf.txt",
                            sep = "\t",  header = None , names =  ["Chr", "Start", "End", "Reads", "Direction", "Barcodes"])
        return ccf_data

    elif data == "K562_brd4_ccf":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSM4471646_K562_HyPBase_scCC.ccf.txt",
                            sep = "\t",  header = None , names =  ["Chr", "Start", "End", "Reads", "Direction", "Barcodes"])
        return ccf_data
        
    elif data == "HCT116_SP1_ccf":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSM4471648_HCT-116_SP1-HyPBase_scCC.ccf.txt",
                            sep = "\t",  header = None , names =  ["Chr", "Start", "End", "Reads", "Direction", "Barcodes"])
        return ccf_data

    elif data == "K562_SP1_ccf":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSM4471650_K562_SP1-HyPBase_scCC.ccf.txt",
                            sep = "\t",  header = None , names =  ["Chr", "Start", "End", "Reads", "Direction", "Barcodes"])
        return ccf_data

    elif data == "barcodes":
        barcodes = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/K562HCT116_barcodes.csv",
                        sep = "\t")
        return barcodes

    else:
        avail_data = ['HCT116_SP1_ccf','K562_SP1_ccf','HCT116_brd4_ccf','K562_brd4_ccf','barcodes']
        raise ValueError(f'data must be one of {avail_data}.')
            
_mouse_brd4 = Optional[Literal['Female_Brd4','Male_Brd4','RNA']]

def mouse_brd4_data(
    data = _mouse_brd4
) -> Union[pd.DataFrame,ad.AnnData]:


    """\
    Bulk Brd4 bindings for both male and female mice [Kfoury21]_.

    :param data: `['Female_Brd4','Male_Brd4','RNA']`. 
        | `Female_Brd4` reads the ccf file for the bulk Brd4 binding data for female mouse .
        | `Male_Brd4` reads the ccf file for the bulk Brd4 binding data for male mouse.


    """

    if data == "Female_Brd4":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSE156678_F6_Brd4_pBC.ccf.txt",
                            sep = "\t",  header = None, names =  ["Chr", "Start", "End", "Reads", "Direction", "Barcodes"]) 
        return ccf_data
        
    if data == "Male_Brd4":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSE156678_M6_Brd4_pBC.ccf.txt",
                            sep = "\t",  header = None, names = ["Chr", "Start", "End", "Reads", "Direction", "Barcodes"])
        return ccf_data

    if data == "RNA":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/dmso_MF.csv", index_col = 0)
        return ccf_data


    else:
        avail_data = ['Female_Brd4','Male_Brd4','RNA']
        raise ValueError(f'data must be one of {avail_data}.')
            
_Cre = Optional[Literal["SP1_P10","SP1_P28","background"]]

def SP1_Cre_data(
    data = _Cre
) -> Union[pd.DataFrame,ad.AnnData]:


    """\
    Bulk SP1 bindings for both P10 and P28 cohort in Cre-driver mouse lines [Cammack19]_.

    :param data: `['SP1_P10','SP1_P28','background']`.
        `SP1_P10` reads the ccf file for P10 cohert.
        | `SP1_P28` reads the ccf file for P28 cohert.
        | `background` reads the ccf file for backgound.


    """

    if data == "SP1_P10":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/SP1_P10.txt",
                            sep = "\t",  header = None, names = ["Chr", "Start", "End", "Reads", "Direction", "Barcodes"])
        return ccf_data
        
    if data == "SP1_P28":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/SP1_P28.txt",
                            sep = "\t",  header = None, names =  ["Chr", "Start", "End", "Reads", "Direction", "Barcodes"])
        return ccf_data

    if data == "background":
        ccf_data = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/SP1_bg.txt",
                            sep = "\t",  header = None, names =  ["Chr", "Start", "End", "Reads", "Direction", "Barcodes"])
        return ccf_data
        
    else:
        avail_data = ["SP1_P10","SP1_P28","background"]
        raise ValueError(f'data must be one of {avail_data}.')
            
