import numpy as np
from anndata import AnnData
from typing import Union, Optional, List, Sequence, Iterable, Mapping, Literal, Tuple
import scanpy as sc

def filter_peaks(
    data: AnnData,
    min_counts: Optional[int] = None,
    min_cells: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_cells: Optional[int] = None,
    inplace: bool = True,
    copy: bool = False,
) -> Union[AnnData, None, Tuple[np.ndarray, np.ndarray]]:

    """\

    Filter peaks based on number of cells or counts.
    Keep peaks that have at least `min_counts` counts or are expressed in at
    least `min_cells` cells or have at most `max_counts` counts or are expressed
    in at most `max_cells` cells.
    Only provide one of the optional parameters `min_counts`, `min_cells`,
    `max_counts`, `max_cells` per call.


    :param data:
        An annotated data matrix of shape `n_obs` * `n_vars`. Rows correspond
        to cells and columns to peaks.
    :param min_counts:
        Minimum number of counts required for a peak to pass filtering.
    :param min_cells:
        Minimum number of cells expressed required for a peak to pass filtering.
    :param max_counts:
        Maximum number of counts required for a peak to pass filtering.
    :param max_cells:
        Maximum number of cells expressed required for a peak to pass filtering.
    :param inplace:
        Perform computation inplace or return result.
        
    :Returns:
    
    Depending on `inplace`, returns the following arrays or directly subsets
    and annotates the data matrix.

    | **peak_subset** - Boolean index mask that does filtering. `True` means that the
        peak is kept. `False` means the peak is removed.
    | **number_per_peak** - Depending on what was tresholded (`counts` or `cells`), the array stores
        `n_counts` or `n_cells` per peak.   
        
    :Example:
    >>> import pycallingcards as cc
    >>> adata_ccf = cc.datasets.mousecortex_CCF()
    >>> cc.pp.filter_peaks(adata_ccf, min_counts=1)
 
    """


    return  sc.pp.filter_genes(data = data, min_counts=min_counts, min_cells=min_cells, max_counts=max_counts, 
                                max_cells=max_cells, inplace=inplace, copy=copy)


