import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Union, Optional, List, Sequence, Iterable, Literal, Tuple
from matplotlib.axes import Axes
from matplotlib import rcParams, cm
from matplotlib import pyplot as plt




def rank_peak_groups(
    adata_ccf: AnnData,
    groups: Union[str, Sequence[str]] = None,
    n_peaks: int = 10,
    peak_symbols: Optional[str] = None,
    key: Optional[str] = 'rank_peak_groups',
    fontsize: int = 8,
    ncols: int = 4,
    sharey: bool = True,
    show: Optional[bool] = True,
    save: Union[bool,str] = False
):

    """\
    Plot ranking of peaks.

    :param adata_ccf:
        Annotated data matrix.
    :param groups: Default is `None`.
        The groups used to show the peak ranking.
    :param n_peaks: Default is 10.
        Number of peaks that appear in the returned tables.
    :param peak_symbols: Default is `None`.
         Key for field in .var that stores peak symbols if you do not want to use .var_names.
    :param key: Default is `rank_peak_groups`.
        Key for the name of the cluster.
    :param fontsize: Default is 8.
        Fontsize for peak names.
    :param ncols: Default is 4.
        Number of panels shown per row.
    :param sharey: Default is `True`.
        Controls if the y-axis of each panels should be shared. By passing
        `sharey=False`, each panel has its own y-axis range.
    :param show: Default is `True`
        Controls if the plot shows or not. 
    :param save: Default is `False`.
        Could be bool or str indicating the file name it would be saved.
        If `True`, a default name would be given and the plot would be saved as png.


    :example:
    >>> import pycallingcards as cc
    >>> adata_ccf = cc.datasets.mousecortex_data(data="CCF")
    >>> cc.pl.rank_peaks_groups(adata_ccf)

    :See also: `tl.rank_peaks_groups`
    """

    import matplotlib as mpl
    mpl.rc_file_defaults()

    n_panels_per_row = ncols
    if n_peaks < 1:
        raise NotImplementedError(
            "Specifying a negative number for n_peaks has not been implemented for "
            f"this plot. Received n_peaks={n_peaks}."
        )

    reference = str(adata_ccf.uns[key]['params']['reference'])
    group_names = adata_ccf.uns[key]['names'].dtype.names if groups is None else groups
    # one panel for each group
    # set up the figure
    n_panels_x = min(n_panels_per_row, len(group_names))
    n_panels_y = np.ceil(len(group_names) / n_panels_x).astype(int)

    from matplotlib import gridspec

    fig = plt.figure(
        figsize=(
            n_panels_x * rcParams['figure.figsize'][0],
            n_panels_y * rcParams['figure.figsize'][1],
        )
    )
    gs = gridspec.GridSpec(nrows=n_panels_y, ncols=n_panels_x, wspace=0.22, hspace=0.3)

    ax0 = None
    ymin = np.Inf
    ymax = -np.Inf
    for count, group_name in enumerate(group_names):
        peak_names = adata_ccf.uns[key]['names'][group_name][:n_peaks]
        pvalues = adata_ccf.uns[key]['pvalues'][group_name][:n_peaks]

        # Setting up axis, calculating y bounds
        if sharey:
            ymin = min(ymin, np.min(pvalues))
            ymax = max(ymax, np.max(pvalues))

            if ax0 is None:
                ax = fig.add_subplot(gs[count])
                ax0 = ax
            else:
                ax = fig.add_subplot(gs[count], sharey=ax0)
        else:
            ymin = np.min(pvalues)
            ymax = np.max(pvalues)
            ymax += 0.3 * (ymax - ymin)

            ax = fig.add_subplot(gs[count])
            ax.set_ylim(ymin, ymax)

        ax.set_xlim(-0.9, n_peaks - 0.1)

        # Mapping to peak_symbols
        if peak_symbols is not None:
            if adata_ccf.raw is not None and adata_ccf.uns[key]['params']['use_raw']:
                peak_names = adata_ccf.raw.var[peak_symbols][peak_names]
            else:
                peak_names = adata_ccf.var[peak_symbols][peak_names]

        # Making labels
        for ig, peak_name in enumerate(peak_names):
            ax.text(
                ig,
                pvalues[ig],
                peak_name,
                rotation='vertical',
                verticalalignment='bottom',
                horizontalalignment='center',
                fontsize=fontsize,
            )

        ax.set_title('{} vs. {}'.format(group_name, reference))
        if count >= n_panels_x * (n_panels_y - 1):
            ax.set_xlabel('ranking')

        # print the 'score' label only on the first panel per row.
        if count % n_panels_x == 0:
            ax.set_ylabel('pvalue')

    if sharey is True:
        ymax += 0.3 * (ymax - ymin)
        ax.set_ylim(ymin, ymax)

    if save != False:
        if save == True:
            save = f"rank_peak_groups_{adata_ccf.uns[key]['params']['groupby']}" + '.png'
        plt.savefig(save, bbox_inches='tight')
        
    if show:
        plt.show()

    plt.close() 


