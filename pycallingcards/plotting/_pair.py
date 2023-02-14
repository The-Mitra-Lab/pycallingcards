from typing import Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import numpy as np
import scanpy as sc
from mudata import MuData

_method = Optional[
    Literal["avginsertions", "logAvginsertions", "suminsertions", "logSuminsertions"]
]


def _adata_insertions(
    mdata: MuData,
    name: str,
    adata_cc: str = "CC",
    adata: str = "RNA",
    groupby: str = "cluster",
    method: _method = "logAvginsertions",
    peak: str = "all",
):

    possible_group = list(mdata.obs[adata + ":" + groupby].unique())
    my_dictionary = {}

    for groupname in possible_group:

        if peak == "all":
            tempadata = mdata[adata_cc][
                (mdata.obs[[adata + ":" + groupby]] == groupname)[
                    adata + ":" + groupby
                ],
                :,
            ].X
        else:
            tempadata = mdata[adata_cc][
                (mdata.obs[[adata + ":" + groupby]] == groupname)[
                    adata + ":" + groupby
                ],
                peak,
            ].X

        if method == "avginsertions":
            my_dictionary[groupname] = tempadata.nnz / tempadata.shape[0]
        elif method == "logAvginsertions":
            my_dictionary[groupname] = np.log2((tempadata.nnz / tempadata.shape[0]) + 1)
        elif method == "suminsertions":
            my_dictionary[groupname] = tempadata.nnz
        elif method == "logSuminsertions":
            my_dictionary[groupname] = np.log2(tempadata.nnz + 1)
        else:

            avail_data = [
                "avginsertions",
                "logAvginsertions",
                "suminsertions",
                "logSuminsertions",
            ]
            raise ValueError(f"data must be one of {avail_data}.")

    mdata[adata].obs[name] = mdata[adata].obs[groupby]
    mdata[adata].obs = mdata[adata].obs.replace({name: my_dictionary})
    mdata[adata].obs[name] = mdata[adata].obs[name].astype(float)

    return adata


def plot_matched(
    mdata: MuData,
    peak: str,
    gene: str,
    name: str = None,
    adata_cc: str = "CC",
    adata: str = "RNA",
    groupby: str = "cluster",
    method: _method = "logAvginsertions",
):

    """\
    Plot matched peak,gene and cluster pair (Designed for mudata object).

    :param mdata:
        mdata for both CC and RNA.
    :param peak:
        Peak name.
    :param gene:
        Gene name.
    :param name:
        The name for peak displyed.
    :param adata_cc:
        Name for CC data. Anndata is mdata[adata_cc].
    :param adata: Default is `'RNA'`.
        Name for RNA data. Anndata is mdata[adata].
    :param groupby:
        The name all the cells are grouped by.
    :param method: `["avginsertions","logAvginsertions","suminsertions","logSuminsertions"]`.
        Method to calculate the insertions.



    :example:
    >>> import pycallingcards as cc
    >>> mdata = cc.datasets.mousecortex_data(data="Mudata")
    >>> cc.pl.plot_matched(mdata,'chr4_22969921_22973019','Pou3f2')

    """

    mpl.rc_file_defaults()

    if name == None:
        name = peak

    _adata_insertions(
        mdata,
        name,
        method=method,
        adata_cc=adata_cc,
        adata=adata,
        groupby=groupby,
        peak=peak,
    )
    sc.pl.umap(mdata[adata], color=[peak, gene, "cluster"])
    mdata[adata].obs = mdata[adata].obs.drop(columns=[name])
