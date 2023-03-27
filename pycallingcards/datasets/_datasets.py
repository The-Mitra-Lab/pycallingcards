import os
from typing import Literal, Optional, Union

import anndata as ad
import mudata as md
import pandas as pd
import scanpy as sc
from appdirs import user_cache_dir

PYCALLINGCARDS_CACHE_DIR = user_cache_dir("pycallingcards")
if not os.path.exists(PYCALLINGCARDS_CACHE_DIR):
    os.makedirs(PYCALLINGCARDS_CACHE_DIR)


def mousecortex_data(
    data: Optional[Literal["qbed", "barcodes", "RNA", "CC", "Mudata"]]
) -> Union[pd.DataFrame, ad.AnnData]:

    """\
    Mouse cortex single-cell calling cards  data :cite:`moudgil2020self`.

    :param data: `['qbed','barcodes','RNA','CC']`.
        `qbed` reads the qbed file.
        `barcodes` reads the barcodes file.
        | `RNA` reads the RNA anndata.
        | `CC` reads the CC anndata.
        | `Mudata` reads the mudata for both RNA and CC.

    :example:
    >>> import pycallingcards as cc
    >>> data = cc.datasets.mousecortex_data(data='qbed')

    """

    if data == "qbed":
        qbed_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/Mouse-Cortex.ccf",
            sep="\t",
            header=None,
            names=["Chr", "Start", "End", "Reads", "Direction", "Barcodes"],
        )
        return qbed_data

    elif data == "barcodes":
        barcodes = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/Mouse-Cortex-scCC_barcodes.csv",
            sep=",",
        )
        return barcodes

    elif data == "RNA":
        filename = os.path.join(PYCALLINGCARDS_CACHE_DIR, "Mouse-Cortex_RNA.h5ad")
        url = "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/Mouse-Cortex_RNA.h5ad"
        adata = sc.read(filename, backup_url=url)
        return adata

    elif data == "CC":
        filename = os.path.join(PYCALLINGCARDS_CACHE_DIR, "Mouse-Cortex_CCF.h5ad")
        url = "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/Mouse-Cortex_CCF.h5ad"
        adata = sc.read(filename, backup_url=url)
        return adata

    elif data == "Mudata":
        import wget

        url = "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/Mouse-Cortex.h5mu"
        filename = os.path.join(PYCALLINGCARDS_CACHE_DIR, "Mouse-Cortex.h5mu")
        wget.download(url, filename)
        mudata = md.read_h5mu(filename)

        return mudata

    else:
        avail_data = ["qbed", "barcodes", "RNA", "CC", "Mudata"]
        raise ValueError(f"data must be one of {avail_data}.")


_SP1 = Optional[
    Literal[
        "HCT116_SP1_qbed",
        "K562_SP1_qbed",
        "HCT116_brd4_qbed",
        "K562_brd4_qbed",
        "barcodes",
    ]
]


def SP1_K562HCT116_data(data: _SP1) -> Union[pd.DataFrame, ad.AnnData]:

    """\
    Transcription factor SP1 is added to both K562 and HCT-116 cell lines seperately.
    Wild-type single-cell calling cards is data also recorded in mixed K562 and HCT-116 cell lines :cite:`moudgil2020self`.

    :param data: `['HCT116_SP1_qbed','K562_SP1_qbed','HCT116_brd4_qbed','K562_brd4_qbed','barcodes']`.
        `experience_qbed` reads the qbed file for the combined data for K562 and HCT-116 cell lines.
        | `background_qbed` reads the qbed file for the data for mixed K562 and HCT-116 cell lines.
        | `barcodes` reads the barcode file for the combined data for K562 and HCT-116 cell lines.

    :example:
    >>> import pycallingcards as cc
    >>> data = cc.datasets.SP1_K562HCT116_data(data='HCT116_SP1_qbed')
    """

    if data == "HCT116_brd4_qbed":
        qbed_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSM4471646_HCT-116_HyPBase_scCC.ccf.txt",
            sep="\t",
            header=None,
            names=["Chr", "Start", "End", "Reads", "Direction", "Barcodes"],
        )
        return qbed_data

    elif data == "K562_brd4_qbed":
        qbed_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSM4471646_K562_HyPBase_scCC.ccf.txt",
            sep="\t",
            header=None,
            names=["Chr", "Start", "End", "Reads", "Direction", "Barcodes"],
        )
        return qbed_data

    elif data == "HCT116_SP1_qbed":
        qbed_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSM4471648_HCT-116_SP1-HyPBase_scCC.ccf.txt",
            sep="\t",
            header=None,
            names=["Chr", "Start", "End", "Reads", "Direction", "Barcodes"],
        )
        return qbed_data

    elif data == "K562_SP1_qbed":
        qbed_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSM4471650_K562_SP1-HyPBase_scCC.ccf.txt",
            sep="\t",
            header=None,
            names=["Chr", "Start", "End", "Reads", "Direction", "Barcodes"],
        )
        return qbed_data

    elif data == "barcodes":
        barcodes = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/K562HCT116_barcodes.csv",
            sep="\t",
        )
        return barcodes

    else:
        avail_data = [
            "HCT116_SP1_qbed",
            "K562_SP1_qbed",
            "HCT116_brd4_qbed",
            "K562_brd4_qbed",
            "barcodes",
        ]
        raise ValueError(f"data must be one of {avail_data}.")


_mouse_brd4 = Optional[
    Literal[
        "Female_Brd4",
        "Male_Brd4",
        "RNA",
        "CC",
        "Female_Brd4_rep1",
        "Female_Brd4_rep2",
        "Male_Brd4_rep1",
        "Male_Brd4_rep2",
    ]
]


def mouse_brd4_data(data: _mouse_brd4) -> Union[pd.DataFrame, ad.AnnData]:

    """\
    Bulk Brd4 bindings for both male and female mice :cite:`kfoury2021brd4`.

    :param data: `['Female_Brd4','Male_Brd4','RNA']`.
        | `Female_Brd4` reads the qbed file for the bulk Brd4 binding data for female mouse .
        | `Male_Brd4` reads the qbed file for the bulk Brd4 binding data for male mouse.
        | `RNA` reads the normalized RNA data.
        | `CC` reads the anndata object.

    :example:
    >>> import pycallingcards as cc
    >>> data = cc.datasets.mouse_brd4_data(data='Female_Brd4')
    """

    if data == "Female_Brd4":
        qbed_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSE156678_F_Brd4.qbed",
            sep="\t",
            header=None,
            names=["Chr", "Start", "End", "Reads", "Direction", "Barcodes"],
        )
        return qbed_data

    elif data == "Male_Brd4":
        qbed_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSE156678_M_Brd4.qbed",
            sep="\t",
            header=None,
            names=["Chr", "Start", "End", "Reads", "Direction", "Barcodes"],
        )
        return qbed_data

    elif data == "RNA":
        qbed_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/dmso_MF.csv",
            index_col=0,
        )
        return qbed_data

    elif data == "CC":
        filename = os.path.join(PYCALLINGCARDS_CACHE_DIR, "Brd4_bindings_bulk.h5ad")
        url = "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/Brd4_bindings_bulk.h5ad"
        adata = sc.read(filename, backup_url=url)
        return adata

    elif data == "Female_Brd4_rep1":
        qbed_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSE156678_F6_Brd4_mBC.qbed",
            sep="\t",
            header=None,
            names=["Chr", "Start", "End", "Reads", "Direction", "Barcodes"],
        )
        return qbed_data

    elif data == "Female_Brd4_rep2":
        qbed_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSE156678_F6_Brd4_pBC.qbed",
            sep="\t",
            header=None,
            names=["Chr", "Start", "End", "Reads", "Direction", "Barcodes"],
        )
        return qbed_data

    elif data == "Male_Brd4_rep1":
        qbed_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSE156678_M6_Brd4_mBC.qbed",
            sep="\t",
            header=None,
            names=["Chr", "Start", "End", "Reads", "Direction", "Barcodes"],
        )
        return qbed_data

    elif data == "Male_Brd4_rep2":
        qbed_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/GSE156678_M6_Brd4_pBC.qbed",
            sep="\t",
            header=None,
            names=["Chr", "Start", "End", "Reads", "Direction", "Barcodes"],
        )
        return qbed_data

    else:
        avail_data = [
            "Female_Brd4",
            "Male_Brd4",
            "RNA",
            "CC",
            "Female_Brd4_rep1",
            "Female_Brd4_rep2",
            "Male_Brd4_rep1",
            "Male_Brd4_rep2",
        ]
        raise ValueError(f"data must be one of {avail_data}.")


_Cre = Optional[Literal["SP1_P10", "SP1_P28", "background"]]


def SP1_Cre_data(data: _Cre) -> Union[pd.DataFrame, ad.AnnData]:

    """\
    Bulk SP1 bindings for both P10 and P28 cohort in Cre-driver mouse lines :cite:`cammack2020viral`.

    :param data: `['SP1_P10','SP1_P28','background']`.
        `SP1_P10` reads the qbed file for P10 cohert.
        | `SP1_P28` reads the qbed file for P28 cohert.
        | `background` reads the qbed file for backgound.

    :example:
    >>> import pycallingcards as cc
    >>> data = cc.datasets.SP1_Cre_data(data='SP1_P10')
    """

    if data == "SP1_P10":
        qbed_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/SP1_P10.txt",
            sep="\t",
            header=None,
            names=["Chr", "Start", "End", "Reads", "Direction", "Barcodes"],
        )
        return qbed_data

    elif data == "SP1_P28":
        qbed_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/SP1_P28.txt",
            sep="\t",
            header=None,
            names=["Chr", "Start", "End", "Reads", "Direction", "Barcodes"],
        )
        return qbed_data

    elif data == "background":
        qbed_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/SP1_bg.txt",
            sep="\t",
            header=None,
            names=["Chr", "Start", "End", "Reads", "Direction", "Barcodes"],
        )
        return qbed_data

    else:
        avail_data = ["SP1_P10", "SP1_P28", "background"]
        raise ValueError(f"data must be one of {avail_data}.")
