from typing import Iterable, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import tqdm
from anndata import AnnData
from mudata import MuData

_Method_rank_peak_groups = Optional[Literal["binomtest", "binomtest2", "fisher_exact"]]
_alternative = Optional[Literal["two-sided", "greater"]]
_alternative_no = Optional[Literal["two-sided", "greater", "None"]]
_rankby = Optional[Literal["pvalues", "logfoldchanges"]]


def DE_pvalue(
    number1: int,
    number2: int,
    total1: int,
    total2: int,
    method: _Method_rank_peak_groups = "fisher_exact",
    alternative: _alternative = "greater",
):

    """\
    Compare the peak difference between two groups for specific peak.

    :param number1:
        The total number of insertions (or the number of cells that contain insertions) in group 1.
    :param number2:
        The total number of insertions (or the number of cells that contain insertions) in group 2.
    :param total1:
        The total number of cells in group 1.
    :param total2:
        The total number of cells in group 2.
    :param method:
        The default method is 'fisher_exact', `binomtest` uses binomial test, `binomtest2` uses
        binomial test but stands on different hypothesis of `binomtest`, 'fisher_exact' uses
        fisher exact test.
    :param alternative:
        If it has two samples/cluster, `'two-sided'` is recommended. Otherwise, please use `'greater'`.


    :return:
        Pvalue for the specific hypothesis.

    :example:
    >>> import pycallingcards as cc
    >>> cc.tl.DE_pvalue(10,456,261,491)

    """

    if method == "binomtest":

        from scipy.stats import binomtest

        if number1 + number2 == 0:
            return 1
        else:
            return binomtest(
                int(number1),
                n=int(number1 + number2),
                p=float(total1 / (total1 + total2)),
                alternative=alternative,
            ).pvalue

    if method == "binomtest2":

        from scipy.stats import binomtest

        return binomtest(
            int(number1), n=total1, p=number2 / total2, alternative=alternative
        ).pvalue

    elif method == "fisher_exact":

        table = np.array([[number1, number2], [total1 - number1, total2 - number2]])
        from scipy.stats import fisher_exact

        _, p = fisher_exact(table, alternative=alternative)

        return p
    else:

        raise ValueError(
            "Please input a correct method: binomtest/binomtest2/fisher_exact."
        )


def diff2group_bygroup(
    adata_cc: AnnData,
    groupby: str,
    name1: str,
    name2: Optional[str] = None,
    peakname: Optional[str] = None,
    test_method: _Method_rank_peak_groups = "fisher_exact",
    alternative: _alternative = "greater",
) -> Union[List[float], float]:

    """\
    Compare the peak difference between two groups for specific peak by group.


    :param adata_cc:
        Annotated data matrix.
    :param groupby:
        The key in adata_cc of the observation groups to consider.
    :param name1:
        The name of the first group.
    :param name2:
        The name of the second group.
    :param peakname:
        The name of the peak used for comparing.
    :param test_method:
        `binomtest` uses binomial test, `binomtest2` uses
        binomial test but stands on a different hypothesis of `binomtest`, `fisher_exact` uses
        fisher exact test.
    :param alternative:
        If it has two clusters, `'two-sided'` is recommended. Otherwise, please use `'greater'`.


    :return:
        Pvalue for the specific hypothesis.


    :example:
    >>> import pycallingcards as cc
    >>> adata_cc = cc.datasets.mousecortex_data(data="CC")
    >>> cc.tl.diff2group_bygroup(adata_cc, 'cluster', 'Neuron_Excit_L5_Mixed','Astrocyte','chr2_28188592_28188996')
    """

    if peakname != None:

        cluster1 = adata_cc[
            (adata_cc.obs[[groupby]] == name1)[groupby],
            adata_cc.var.index.get_loc(peakname),
        ].X
        if name2 == None:
            cluster2 = adata_cc[
                (adata_cc.obs[[groupby]] != name1)[groupby],
                adata_cc.var.index.get_loc(peakname),
            ].X
        else:
            cluster2 = adata_cc[
                (adata_cc.obs[[groupby]] == name2)[groupby],
                adata_cc.var.index.get_loc(peakname),
            ].X

        total1 = cluster1.shape[0]
        total2 = cluster2.shape[0]

        if test_method == "binomtest2" or test_method == "fisher_exact":
            number1 = cluster1.nnz
            number2 = cluster2.nnz
        elif test_method == "binomtest":
            number1 = cluster1.sum()
            number2 = cluster2.sum()
        else:
            raise ValueError(
                "Please input a correct method: binomtest/binomtest2/fisher_exact."
            )

        return DE_pvalue(
            number1,
            number2,
            total1,
            total2,
            method=test_method,
            alternative=alternative,
        )

    else:

        print(
            "No peak name is provided, the pvalue for all the peaks will be returned."
        )
        pvaluelist = []

        for peak in list(adata_cc.var.index):

            cluster1 = adata_cc[
                (adata_cc.obs[[groupby]] == name1)[groupby],
                adata_cc.var.index.get_loc(peak),
            ].X
            cluster2 = adata_cc[
                (adata_cc.obs[[groupby]] == name2)[groupby],
                adata_cc.var.index.get_loc(peak),
            ].X

            total1 = cluster1.shape[0]
            total2 = cluster2.shape[0]

            if test_method == "binomtest2" or test_method == "fisher_exact":
                number1 = cluster1.nnz
                number2 = cluster2.nnz
            elif test_method == "binomtest":
                number1 = cluster1.sum()
                number2 = cluster2.sum()
            else:
                raise ValueError(
                    "Please input a correct method: binomtest/binomtest2/fisher_exact."
                )

            pvaluelist.append(
                DE_pvalue(
                    number1,
                    number2,
                    total1,
                    total2,
                    method=test_method,
                    alternative=alternative,
                )
            )

        return pvaluelist


def diff2group_bysample(
    adata_cc: AnnData,
    name1: str,
    name2: Optional[str] = None,
    peakname: Optional[str] = None,
    test_method: _Method_rank_peak_groups = "binomtest",
    alternative: _alternative = "greater",
) -> Union[List[float], float]:

    """\
    Comapare the peak difference between two groups for a specific peak by sample.


    :param adata_cc:
        Annotated data matrix.
    :param name1:
        The name of the first group.
    :param name2:
        The name of the second group.
    :param peakname:
        The name of the peak used for comparing.
    :param test_method: ["binomtest", "binomtest2","fisher_exact"].
        `binomtest` uses binomial test, `binomtest2` uses
        binomial test but stands on a different hypothesis of `binomtest`, `fisher_exact` uses
        fisher exact test.
    :param alternative: `['two-sided', 'greater']`.
        If it has two samples, `'two-sided'` is recommended. Otherwise, please use `'greater'`.



    :return:
        Pvalue for the specific hypothesis.

    :example:
    >>> import pycallingcards as cc
    >>> adata_cc = cc.datasets.mouse_brd4_data(data="CC")
    >>> cc.tl.diff2group_bysample(adata_cc,'F6_Brd4','M6_Brd4','chr1_4196845_4200095','fisher_exact')

    """

    if peakname != None:

        cluster1 = adata_cc[name1, :].X

        if name2 == None:
            name2 = list(set(adata_cc.obs.index).difference(set([name1])))
            cluster2 = adata_cc[name2, :].X
        else:
            cluster2 = adata_cc[name2, :].X

        total1 = int(cluster1.sum())
        total2 = int(cluster2.sum())

        number1 = adata_cc[name1, peakname].X[0, 0]
        number2 = adata_cc[name2, peakname].X.sum()

        return DE_pvalue(
            number1,
            number2,
            total1,
            total2,
            method=test_method,
            alternative=alternative,
        )

    else:

        print(
            "No peak name is provided, the pvalue for all the peaks will be returned."
        )
        pvaluelist = []

        cluster1 = adata_cc[name1, :].X

        if name2 == None:
            name2 = list(set(adata_cc.obs.index).difference(set([name1])))
            cluster2 = adata_cc[name2, :].X
        else:
            cluster2 = adata_cc[name2, :].X

        total1 = int(cluster1.sum())
        total2 = int(cluster2.sum())

        for peak in list(adata_cc.var.index):

            number1 = adata_cc[name1, peak].X[0, 0]
            number2 = adata_cc[name2, peak].X.sum()

            pvaluelist.append(
                DE_pvalue(
                    number1,
                    number2,
                    total1,
                    total2,
                    method=test_method,
                    alternative=alternative,
                )
            )

        return pvaluelist


def rank_peak_groups(
    adata_cc: AnnData,
    groupby: str,
    groups: Union[Literal["all"], Iterable[str]] = "all",
    reference: str = None,
    n_peaks: Optional[int] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
    rankby: _rankby = "pvalues",
    method: _Method_rank_peak_groups = "fisher_exact",
    alternative: _alternative_no = "None",
) -> Optional[AnnData]:

    """\
    Rank peaks for characterizing groups.

    :param adata_cc:
        Annotated data matrix.
    :param groupby:
        The key of the groups.
    :param groups:
        Subset of groups (list), e.g. [`'g1'`, `'g2'`, `'g3'`], to which comparison
        shall be restricted, or `all` (default), for all groups.
    :param reference:
        If `rest`, compare each group to the union of the rest of the group.
        If a group identifier, compare with respect to this group.
    :param n_peaks:
        The number of peaks that appear in the returned tables.
        The default includes all peaks.
    :param key_added:
        The key in `adata.uns` information is saved to.
    :param rankby:
        The list we rank by.
    :param copy:
        If copy, it will return a copy of the AnnData object and leave the passed adata unchanged.
    :param method:
        `binomtest` uses binomial test,
        `binomtest2` uses binomial test but stands on a different hypothesis of `binomtest`,
        `fisher_exact` uses fisher exact test.
    :param alternative:
        If it has two samples/cluster, `'two-sided'` is recommended. Otherwise, please use `'greater'`.
        For default (`'None'`), if groupby == "Index", it will be 'two-sided'. Otherwise, please use `'greater'`.



    :Returns:
        | **names** - structured `np.ndarray` (`.uns['rank_peaks_groups']`). Structured array is to be indexed by the group ID storing the peak names. It's ordered according to scores.
        | **return pvalues** - structured `np.ndarray` (`.uns['rank_peaks_groups']`)
        | **return logfoldchanges** - structured `np.ndarray` (`.uns['rank_peaks_groups']`)
        | **number** -  `pandas.DataFrame` (`.uns['rank_peaks_groups']`). The number of peaks or the number of cells that contain peaks (depending on the method).
        | **number_rest** - `pandas.DataFrame` (`.uns['rank_peaks_groups']`). The number of peaks or the number of cells that contain peaks (depending on the method).
        | **total** -  `pandas.DataFrame` (`.uns['rank_peaks_groups']`). The total number of cells that contain peaks.
        | **total_rest** -  `pandas.DataFrame` (`.uns['rank_peaks_groups']`). The total number of cells that contain peaks.

    :example:
    >>> import pycallingcards as cc
    >>> adata_cc = cc.datasets.mousecortex_data(data="CC")
    >>> cc.tl.rank_peak_groups(adata_cc,'cluster',method = 'binomtest',key_added = 'binomtest')
    """

    if groupby == "Index":
        if alternative == "None":
            _rank_peak_groups_bygroup(
                adata_cc=adata_cc,
                groups=groups,
                reference=reference,
                n_peaks=n_peaks,
                key_added=key_added,
                copy=copy,
                rankby=rankby,
                method=method,
                alternative="two-sided",
            )
        else:
            _rank_peak_groups_bygroup(
                adata_cc=adata_cc,
                groups=groups,
                reference=reference,
                n_peaks=n_peaks,
                key_added=key_added,
                copy=copy,
                rankby=rankby,
                method=method,
                alternative=alternative,
            )
    else:
        if alternative == "None":
            _rank_peak_groups_bycell(
                adata_cc=adata_cc,
                groupby=groupby,
                groups=groups,
                reference=reference,
                n_peaks=n_peaks,
                key_added=key_added,
                copy=copy,
                rankby=rankby,
                method=method,
                alternative="greater",
            )
        else:
            _rank_peak_groups_bycell(
                adata_cc=adata_cc,
                groupby=groupby,
                groups=groups,
                reference=reference,
                n_peaks=n_peaks,
                key_added=key_added,
                copy=copy,
                rankby=rankby,
                method=method,
                alternative=alternative,
            )


def _rank_peak_groups_bycell(
    adata_cc: AnnData,
    groupby: str,
    groups: Union[Literal["all"], Iterable[str]] = "all",
    reference: str = None,
    n_peaks: Optional[int] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
    rankby: _rankby = "pvalues",
    method: _Method_rank_peak_groups = None,
    alternative: _alternative = "greater",
) -> Optional[AnnData]:

    avail_method = ["binomtest", "binomtest2", "fisher_exact"]
    if method == None:
        method = "binomtest"
    elif method not in avail_method:
        raise ValueError(f"Correction method must be one of {avail_method}.")

    possible_group = list(adata_cc.obs[groupby].unique())
    possible_group.sort()

    if reference == None:
        reference = "rest"
    elif reference not in possible_group:
        raise ValueError(
            f"Invalid reference, should be all or one of {possible_group}."
        )

    if groups == "all":
        group_list = possible_group
    elif type(groups) == str:
        group_list = groups
    elif type(groups) == list:
        group_list = groups
    else:
        raise ValueError("Invalid groups.")

    if key_added == None:
        key_added = "rank_peak_groups"
    elif type(key_added) != str:
        raise ValueError("key_added should be str.")

    adata_cc = adata_cc.copy() if copy else adata_cc

    adata_cc.uns[key_added] = {}
    adata_cc.uns[key_added]["params"] = dict(
        groupby=groupby, reference=reference, method=method
    )

    peak_list = list(adata_cc.var.index)

    if n_peaks == None:
        n_peaks = len(peak_list)
    elif type(n_peaks) != int or n_peaks < 1 or n_peaks > len(peak_list):
        raise ValueError(
            "n_peaks should be a int larger than 0 and smaller than the total number of peaks "
        )

    finalresult_name = np.empty([n_peaks, len(group_list)], dtype="<U100")
    finalresult_pvalue = np.empty([n_peaks, len(group_list)], dtype=float)
    finalresult_logfoldchanges = np.empty([n_peaks, len(group_list)], dtype=float)
    finalresult_number1 = np.empty([n_peaks, len(group_list)], dtype=int)
    finalresult_number2 = np.empty([n_peaks, len(group_list)], dtype=int)
    finalresult_total1 = np.empty([n_peaks, len(group_list)], dtype=int)
    finalresult_total2 = np.empty([n_peaks, len(group_list)], dtype=int)

    i = 0

    for cluster in tqdm.tqdm(group_list):

        if reference == "rest":
            clusterdata = adata_cc[(adata_cc.obs[[groupby]] == cluster)[groupby]]
            clusterdatarest = adata_cc[(adata_cc.obs[[groupby]] != cluster)[groupby]]
        else:
            clusterdata = adata_cc[(adata_cc.obs[[groupby]] == cluster)[groupby]]
            clusterdatarest = adata_cc[(adata_cc.obs[[groupby]] == reference)[groupby]]

        pvaluelist = []
        number1list = []
        number2list = []
        total1list = []
        total2list = []

        total1 = clusterdata.X.shape[0]
        total2 = clusterdatarest.X.shape[0]

        for peak in peak_list:

            cluster1 = clusterdata[:, adata_cc.var.index.get_loc(peak)].X
            # total1 = cluster1.shape[0]

            cluster2 = clusterdatarest[:, adata_cc.var.index.get_loc(peak)].X
            # total2 = cluster2.shape[0]

            if method == "binomtest2" or method == "fisher_exact":
                number1 = cluster1.nnz
                number2 = cluster2.nnz
            elif method == "binomtest":
                number1 = cluster1.sum()
                number2 = cluster2.sum()
            else:
                raise ValueError(
                    "Please input a correct method: binomtest/binomtest2/fisher_exact."
                )

            pvaluelist.append(
                DE_pvalue(
                    number1,
                    number2,
                    total1,
                    total2,
                    method=method,
                    alternative=alternative,
                )
            )
            number1list.append(number1)
            number2list.append(number2)
            total1list.append(total1)
            total2list.append(total2)

        pvaluelistnp = np.array(pvaluelist)
        number1listnp = np.array(number1list)
        number2listnp = np.array(number2list)
        total1listnp = np.array(total1list)
        total2listnp = np.array(total2list)
        logfoldchangenp = np.log2(
            ((number1listnp / total1listnp) + 0.000001)
            / ((number2listnp / total2listnp) + 0.000001)
        )

        if rankby == "pvalues":
            rankarg = pvaluelistnp.argsort()
        elif rankby == "logfoldchanges":
            rankarg = (-1 * logfoldchangenp).argsort()
        else:
            raise ValueError(f"rankby method must be one of {_rankby}.")

        finalresult_name[:, i] = np.array(peak_list)[rankarg][:n_peaks]
        finalresult_pvalue[:, i] = pvaluelistnp[rankarg][:n_peaks]
        finalresult_logfoldchanges[:, i] = logfoldchangenp[rankarg][:n_peaks]
        finalresult_number1[:, i] = number1listnp[rankarg][:n_peaks]
        finalresult_number2[:, i] = number2listnp[rankarg][:n_peaks]
        finalresult_total1[:, i] = total1listnp[rankarg][:n_peaks]
        finalresult_total2[:, i] = total2listnp[rankarg][:n_peaks]

        i += 1

    temppvalue = np.array([group_list, ["float"] * len(group_list)]).transpose()
    tempname = np.array([group_list, ["<U100"] * len(group_list)]).transpose()
    tempnamenumber = np.array([group_list, ["int"] * len(group_list)]).transpose()

    adata_cc.uns[key_added]["names"] = np.rec.array(
        list(map(tuple, finalresult_name)), dtype=list(map(tuple, tempname))
    )
    adata_cc.uns[key_added]["pvalues"] = np.rec.array(
        list(map(tuple, finalresult_pvalue)), dtype=list(map(tuple, temppvalue))
    )
    adata_cc.uns[key_added]["logfoldchanges"] = np.rec.array(
        list(map(tuple, finalresult_logfoldchanges)), dtype=list(map(tuple, temppvalue))
    )
    adata_cc.uns[key_added]["number"] = np.rec.array(
        list(map(tuple, finalresult_number1)), dtype=list(map(tuple, tempnamenumber))
    )
    adata_cc.uns[key_added]["number_rest"] = np.rec.array(
        list(map(tuple, finalresult_number2)), dtype=list(map(tuple, tempnamenumber))
    )
    adata_cc.uns[key_added]["total"] = np.rec.array(
        list(map(tuple, finalresult_total1)), dtype=list(map(tuple, tempnamenumber))
    )
    adata_cc.uns[key_added]["total_rest"] = np.rec.array(
        list(map(tuple, finalresult_total2)), dtype=list(map(tuple, tempnamenumber))
    )

    return adata_cc if copy else None


def _rank_peak_groups_bygroup(
    adata_cc: AnnData,
    groups: Union[Literal["all"], Iterable[str]] = "all",
    reference: str = None,
    n_peaks: Optional[int] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
    rankby: _rankby = "pvalues",
    method: _Method_rank_peak_groups = None,
    alternative: _alternative = "greater",
) -> Optional[AnnData]:

    avail_method = ["binomtest", "binomtest2", "fisher_exact"]
    if method == None:
        method = "binomtest"
    elif method not in avail_method:
        raise ValueError(f"Correction method must be one of {avail_method}.")

    possible_group = list(adata_cc.obs.index.unique())

    if reference == None:
        reference = "rest"
    elif reference not in possible_group:
        raise ValueError(
            f"Invalid reference, should be all or one of {possible_group}."
        )

    if groups == "all":
        group_list = possible_group
    elif type(groups) == str:
        group_list = groups
    elif type(groups) == list:
        group_list = groups
    else:
        raise ValueError("Invalid groups.")

    if key_added == None:
        key_added = "rank_peak_groups"
    elif type(key_added) != str:
        raise ValueError("key_added should be str.")

    adata_cc = adata_cc.copy() if copy else adata_cc

    adata_cc.uns[key_added] = {}
    adata_cc.uns[key_added]["params"] = dict(
        groupby="Index", reference=reference, method=method
    )

    peak_list = list(adata_cc.var.index)

    if n_peaks == None:
        n_peaks = len(peak_list)
    elif type(n_peaks) != int or n_peaks < 1 or n_peaks > len(peak_list):
        raise ValueError(
            "n_peaks should be a int larger than 0 and smaller than the total number of peaks "
        )

    finalresult_name = np.empty([n_peaks, len(group_list)], dtype="<U100")
    finalresult_pvalue = np.empty([n_peaks, len(group_list)], dtype=float)
    finalresult_logfoldchanges = np.empty([n_peaks, len(group_list)], dtype=float)
    finalresult_number1 = np.empty([n_peaks, len(group_list)], dtype=int)
    finalresult_number2 = np.empty([n_peaks, len(group_list)], dtype=int)
    finalresult_total1 = np.empty([n_peaks, len(group_list)], dtype=int)
    finalresult_total2 = np.empty([n_peaks, len(group_list)], dtype=int)

    i = 0

    for cluster in tqdm.tqdm(group_list):

        cluster1 = adata_cc[cluster, :].X

        if reference == "rest":
            name2 = list(set(adata_cc.obs.index).difference(set([cluster])))
            cluster2 = adata_cc[name2, :].X
        else:
            name2 = reference
            cluster2 = adata_cc[name2, :].X

        total1 = int(cluster1.sum())
        total2 = int(cluster2.sum())

        pvaluelist = []
        number1list = []
        number2list = []
        total1list = []
        total2list = []

        for peak in peak_list:

            number1 = adata_cc[cluster, peak].X[0, 0]
            number2 = adata_cc[name2, peak].X.sum()

            pvaluelist.append(
                DE_pvalue(
                    number1,
                    number2,
                    total1,
                    total2,
                    method=method,
                    alternative=alternative,
                )
            )
            number1list.append(number1)
            number2list.append(number2)
            total1list.append(total1)
            total2list.append(total2)

        pvaluelistnp = np.array(pvaluelist)
        number1listnp = np.array(number1list)
        number2listnp = np.array(number2list)
        total1listnp = np.array(total1list)
        total2listnp = np.array(total2list)
        logfoldchangenp = np.log2(
            ((number1listnp / total1listnp) + 0.000001)
            / ((number2listnp / total2listnp) + 0.000001)
        )

        if rankby == "pvalues":
            rankarg = pvaluelistnp.argsort()
        elif rankby == "logfoldchanges":
            rankarg = (-1 * logfoldchangenp).argsort()
        else:
            raise ValueError(f"rankby method must be one of {_rankby}.")

        finalresult_name[:, i] = np.array(peak_list)[rankarg][:n_peaks]
        finalresult_logfoldchanges[:, i] = logfoldchangenp[rankarg][:n_peaks]
        finalresult_pvalue[:, i] = pvaluelistnp[rankarg][:n_peaks]
        finalresult_number1[:, i] = number1listnp[rankarg][:n_peaks]
        finalresult_number2[:, i] = number2listnp[rankarg][:n_peaks]
        finalresult_total1[:, i] = total1listnp[rankarg][:n_peaks]
        finalresult_total2[:, i] = total2listnp[rankarg][:n_peaks]

        i += 1

    temppvalue = np.array([group_list, ["float"] * len(group_list)]).transpose()
    tempname = np.array([group_list, ["<U100"] * len(group_list)]).transpose()
    tempnamenumber = np.array([group_list, ["int"] * len(group_list)]).transpose()

    adata_cc.uns[key_added]["names"] = np.rec.array(
        list(map(tuple, finalresult_name)), dtype=list(map(tuple, tempname))
    )
    adata_cc.uns[key_added]["pvalues"] = np.rec.array(
        list(map(tuple, finalresult_pvalue)), dtype=list(map(tuple, temppvalue))
    )
    adata_cc.uns[key_added]["logfoldchanges"] = np.rec.array(
        list(map(tuple, finalresult_logfoldchanges)), dtype=list(map(tuple, temppvalue))
    )
    adata_cc.uns[key_added]["number"] = np.rec.array(
        list(map(tuple, finalresult_number1)), dtype=list(map(tuple, tempnamenumber))
    )
    adata_cc.uns[key_added]["number_rest"] = np.rec.array(
        list(map(tuple, finalresult_number2)), dtype=list(map(tuple, tempnamenumber))
    )
    adata_cc.uns[key_added]["total"] = np.rec.array(
        list(map(tuple, finalresult_total1)), dtype=list(map(tuple, tempnamenumber))
    )
    adata_cc.uns[key_added]["total_rest"] = np.rec.array(
        list(map(tuple, finalresult_total2)), dtype=list(map(tuple, tempnamenumber))
    )

    return adata_cc if copy else None


def rank_peak_groups_df(
    adata: AnnData,
    key: str = "rank_peak_groups",
    group: Optional[list] = None,
    pval_cutoff: Optional[float] = None,
    logfc_min: Optional[float] = None,
    logfc_max: Optional[float] = None,
) -> pd.DataFrame:

    """\
    :func:`pycallingcards.tl.rank_peak_groups` results in the form of a :class:`~pandas.DataFrame`.

    Params
    ------
    adata
        Object to get results from.
    key
        Key differential expression groups were stored under.
    group
        Which group (as in :func:`scanpy.tl.rank_genes_groups`'s `groupby`
        argument) to return results from. Can be a list. All groups are
        returned if groups is `None`.
    pval_cutoff
        Return only adjusted p-values below the  cutoff.
    logfc_min
        Minimum logfc to return.
    logfc_max
        Maximum logfc to return.


    Example
    -------
    >>> import pycallingcards as cc
    >>> adata_cc = cc.datasets.mousecortex_data(data="CC")
    >>> cc.tl.rank_peak_groups(adata_cc,'cluster',method = 'binomtest',key_added = 'binomtest')
    >>> cc.tl.rank_peak_groups_df(adata_cc,'Astrocyte','binomtest')
    """

    if group is None:
        group = list(adata.uns[key]["names"].dtype.names)

    colnames = [
        "names",
        "logfoldchanges",
        "pvalues",
        "number",
        "number_rest",
        "total",
        "total_rest",
    ]

    if len(group) == 1:
        d = [pd.DataFrame(adata.uns[key][c])[group] for c in colnames]
        d = pd.concat(d, axis=1)
        d.columns = colnames

    else:
        d = [pd.DataFrame(adata.uns[key][c])[group[0]] for c in colnames]
        d = pd.concat(d, axis=1)
        d.columns = colnames
        d["group"] = group[0]

        for num in range(len(group) - 1):

            f = [pd.DataFrame(adata.uns[key][c])[group[num + 1]] for c in colnames]
            f = pd.concat(f, axis=1)
            f.columns = colnames
            f["group"] = group[num + 1]
            d = pd.concat([d, f], axis=0)

    if pval_cutoff is not None:
        d = d[d["pvalues"] <= pval_cutoff]
    if logfc_min is not None:
        d = d[d["logfoldchanges"] >= logfc_min]
    if logfc_max is not None:
        d = d[d["logfoldchanges"] <= logfc_max]

    return d.reset_index(drop=True)


def rank_peak_groups_mu(
    mdata: MuData,
    groupby: str,
    adata_cc: str = "CC",
    groups: Union[Literal["all"], Iterable[str]] = "all",
    reference: str = None,
    n_peaks: Optional[int] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
    rankby: _rankby = "pvalues",
    method: _Method_rank_peak_groups = "fisher_exact",
    alternative: _alternative_no = "None",
) -> Optional[AnnData]:

    """\
    Rank peaks for characterizing groups. Designed for mudata object.

    :param mdata:
        mdata for both RNA and CC data.
    :param groupby:
        The key of the groups.
    :param adata_cc:
        Name for Anndata of CC. Anndata is mdata[adata_cc].
    :param groups:
        Subset of groups (list), e.g. [`'g1'`, `'g2'`, `'g3'`], to which comparison
        shall be restricted, or `all` (default), for all groups.
    :param reference:
        If `rest`, compare each group to the union of the rest of the group.
        If a group identifier, compare with respect to this group.
    :param n_peaks:
        The number of peaks that appear in the returned tables.
        Default includes all peaks.
    :param key_added:
        The key in `adata.uns` information is saved to.
    :param rankby: ['pvalues', 'logfoldchanges'].
        The list we rank by.
    :param copy:
        If copy, it will return a copy of the AnnData object and leave the passed adata unchanged.
    :param  method: ["binomtest", "binomtest2","fisher_exact"].
        `binomtest` uses binomial test,
        `binomtest2` uses binomial test but stands on a different hypothesis of `binomtest`,
        `fisher_exact` uses fisher exact test.
    :param alternative: `['two-sided', 'greater','None']`.
        If it has two samples/cluster, `'two-sided'` is recommended. Otherwise, please use `'greater'`.
        For default (`'None'`), if groupby == "Index", it will be 'two-sided'. Otherwise, please use `'greater'`.



    :Returns:
        | **names** - structured `np.ndarray` (`.uns['rank_peaks_groups']`). Structured array is to be indexed by the group ID storing the peak names. Ordered according to scores.
        | **return pvalues** - structured `np.ndarray` (`.uns['rank_peaks_groups']`)
        | **return logfoldchanges** - structured `np.ndarray` (`.uns['rank_peaks_groups']`)
        | **number** -  `pandas.DataFrame` (`.uns['rank_peaks_groups']`). The number of peaks or the number of cells that contain peaks (depending on the method).
        | **number_rest** - `pandas.DataFrame` (`.uns['rank_peaks_groups']`). The number of peaks or the number of cells that contain peaks (depending on the method).
        | **total** -  `pandas.DataFrame` (`.uns['rank_peaks_groups']`). The total number of cells that contain peaks.
        | **total_rest** -  `pandas.DataFrame` (`.uns['rank_peaks_groups']`). The total number of cells that contain peaks.

    :example:
    >>> import pycallingcards as cc
    >>> mdata = cc.datasets.mousecortex_data(data="Mudata")
    >>> cc.tl.rank_peak_groups_mu(mdata,"RNA:cluster",method = 'binomtest',key_added = 'binomtest')
    """

    if alternative == "None":
        _rank_peak_groups_bycell_mu(
            mdata=mdata,
            adata_cc=adata_cc,
            groupby=groupby,
            groups=groups,
            reference=reference,
            n_peaks=n_peaks,
            key_added=key_added,
            copy=copy,
            rankby=rankby,
            method=method,
            alternative="greater",
        )
    else:
        _rank_peak_groups_bycell_mu(
            mdata=mdata,
            adata_cc=adata_cc,
            groupby=groupby,
            groups=groups,
            reference=reference,
            n_peaks=n_peaks,
            key_added=key_added,
            copy=copy,
            rankby=rankby,
            method=method,
            alternative=alternative,
        )


def _rank_peak_groups_bycell_mu(
    mdata: MuData,
    groupby: str,
    adata_cc: str = "CC",
    groups: Union[Literal["all"], Iterable[str]] = "all",
    reference: str = None,
    n_peaks: Optional[int] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
    rankby: _rankby = "pvalues",
    method: _Method_rank_peak_groups = None,
    alternative: _alternative = "greater",
) -> Optional[AnnData]:

    avail_method = ["binomtest", "binomtest2", "fisher_exact"]
    if method == None:
        method = "binomtest"
    elif method not in avail_method:
        raise ValueError(f"Correction method must be one of {avail_method}.")

    possible_group = list(mdata.obs[groupby].unique())
    possible_group.sort()

    if reference == None:
        reference = "rest"
    elif reference not in possible_group:
        raise ValueError(
            f"Invalid reference, should be all or one of {possible_group}."
        )

    if groups == "all":
        group_list = possible_group
    elif type(groups) == str:
        group_list = groups
    elif type(groups) == list:
        group_list = groups
    else:
        raise ValueError("Invalid groups.")

    if key_added == None:
        key_added = "rank_peak_groups"
    elif type(key_added) != str:
        raise ValueError("key_added should be str.")

    mdata[adata_cc].uns[key_added] = {}
    mdata[adata_cc].uns[key_added]["params"] = dict(
        groupby=groupby, reference=reference, method=method
    )

    peak_list = list(mdata[adata_cc].var.index)

    if n_peaks == None:
        n_peaks = len(peak_list)
    elif type(n_peaks) != int or n_peaks < 1 or n_peaks > len(peak_list):
        raise ValueError(
            "n_peaks should be a int larger than 0 and smaller than the total number of peaks "
        )

    finalresult_name = np.empty([n_peaks, len(group_list)], dtype="<U100")
    finalresult_pvalue = np.empty([n_peaks, len(group_list)], dtype=float)
    finalresult_logfoldchanges = np.empty([n_peaks, len(group_list)], dtype=float)
    finalresult_number1 = np.empty([n_peaks, len(group_list)], dtype=int)
    finalresult_number2 = np.empty([n_peaks, len(group_list)], dtype=int)
    finalresult_total1 = np.empty([n_peaks, len(group_list)], dtype=int)
    finalresult_total2 = np.empty([n_peaks, len(group_list)], dtype=int)

    i = 0

    for cluster in tqdm.tqdm(group_list):

        if reference == "rest":
            clusterdata = mdata[adata_cc][(mdata.obs[[groupby]] == cluster)[groupby]]
            clusterdatarest = mdata[adata_cc][
                (mdata.obs[[groupby]] != cluster)[groupby]
            ]
        else:
            clusterdata = mdata[adata_cc][(mdata.obs[[groupby]] == cluster)[groupby]]
            clusterdatarest = mdata[adata_cc][
                (mdata.obs[[groupby]] == reference)[groupby]
            ]

        pvaluelist = []
        number1list = []
        number2list = []
        total1list = []
        total2list = []

        total1 = clusterdata.X.shape[0]
        total2 = clusterdatarest.X.shape[0]

        for peak in peak_list:

            cluster1 = clusterdata[:, mdata[adata_cc].var.index.get_loc(peak)].X

            cluster2 = clusterdatarest[:, mdata[adata_cc].var.index.get_loc(peak)].X

            if method == "binomtest2" or method == "fisher_exact":
                number1 = cluster1.nnz
                number2 = cluster2.nnz
            elif method == "binomtest":
                number1 = cluster1.sum()
                number2 = cluster2.sum()
            else:
                raise ValueError(
                    "Please input a correct method: binomtest/binomtest2/fisher_exact."
                )

            pvaluelist.append(
                DE_pvalue(
                    number1,
                    number2,
                    total1,
                    total2,
                    method=method,
                    alternative=alternative,
                )
            )
            number1list.append(number1)
            number2list.append(number2)
            total1list.append(total1)
            total2list.append(total2)

        pvaluelistnp = np.array(pvaluelist)
        number1listnp = np.array(number1list)
        number2listnp = np.array(number2list)
        total1listnp = np.array(total1list)
        total2listnp = np.array(total2list)
        logfoldchangenp = np.log2(
            ((number1listnp / total1listnp) + 0.000001)
            / ((number2listnp / total2listnp) + 0.000001)
        )

        if rankby == "pvalues":
            rankarg = pvaluelistnp.argsort()
        elif rankby == "logfoldchanges":
            rankarg = (-1 * logfoldchangenp).argsort()
        else:
            raise ValueError(f"rankby method must be one of {_rankby}.")

        finalresult_name[:, i] = np.array(peak_list)[rankarg][:n_peaks]
        finalresult_pvalue[:, i] = pvaluelistnp[rankarg][:n_peaks]
        finalresult_logfoldchanges[:, i] = logfoldchangenp[rankarg][:n_peaks]
        finalresult_number1[:, i] = number1listnp[rankarg][:n_peaks]
        finalresult_number2[:, i] = number2listnp[rankarg][:n_peaks]
        finalresult_total1[:, i] = total1listnp[rankarg][:n_peaks]
        finalresult_total2[:, i] = total2listnp[rankarg][:n_peaks]

        i += 1

    temppvalue = np.array([group_list, ["float"] * len(group_list)]).transpose()
    tempname = np.array([group_list, ["<U100"] * len(group_list)]).transpose()
    tempnamenumber = np.array([group_list, ["int"] * len(group_list)]).transpose()

    mdata[adata_cc].uns[key_added]["names"] = np.rec.array(
        list(map(tuple, finalresult_name)), dtype=list(map(tuple, tempname))
    )
    mdata[adata_cc].uns[key_added]["pvalues"] = np.rec.array(
        list(map(tuple, finalresult_pvalue)), dtype=list(map(tuple, temppvalue))
    )
    mdata[adata_cc].uns[key_added]["logfoldchanges"] = np.rec.array(
        list(map(tuple, finalresult_logfoldchanges)), dtype=list(map(tuple, temppvalue))
    )
    mdata[adata_cc].uns[key_added]["number"] = np.rec.array(
        list(map(tuple, finalresult_number1)), dtype=list(map(tuple, tempnamenumber))
    )
    mdata[adata_cc].uns[key_added]["number_rest"] = np.rec.array(
        list(map(tuple, finalresult_number2)), dtype=list(map(tuple, tempnamenumber))
    )
    mdata[adata_cc].uns[key_added]["total"] = np.rec.array(
        list(map(tuple, finalresult_total1)), dtype=list(map(tuple, tempnamenumber))
    )
    mdata[adata_cc].uns[key_added]["total_rest"] = np.rec.array(
        list(map(tuple, finalresult_total2)), dtype=list(map(tuple, tempnamenumber))
    )

    return mdata if copy else None
