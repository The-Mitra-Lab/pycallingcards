
import numpy as np
from anndata import AnnData
from typing import Union, Optional, List, Sequence, Iterable, Mapping, Literal, Tuple


_Method_rank_peak_groups = Optional[Literal['binomtest', 'binomtest2','fisher_exact', 'chisquare']]

def _calculatePvalue(number1,number2,total1,total2,method = "binomtest"):

    if method == "binomtest":

        from scipy.stats import binomtest

        if number1 + number2 ==0:
            return 1
        else:
            return binomtest(int(number1), n=int(number1+number2), p=float(total1/(total1+total2))).pvalue

    if method == "binomtest2":

        from scipy.stats import binomtest

        return max(binomtest(int(number1), n=total1, p=number2/total2).pvalue,
                   binomtest(int(number2), n=total2, p=number1/total1).pvalue)

    elif method == "fisher_exact":

        table = np.array([[number1, number2], [total1- number1, total2 - number2]])
        from scipy.stats import fisher_exact
        _,p = fisher_exact(table, alternative='two-sided')

        return p

    elif method == "chisquare":

        from scipy.stats import chisquare
        ratio = (number1+number2) /(total1+total2)

        return chisquare([number1, number2], f_exp=[ratio*total1, ratio*total2]).pvalue

    else:

        raise ValueError("Please input a correct method: binomtest/binomtest2/fisher_exact/chisquare.")

def diff2group(
    adata_ccf: AnnData,
    name1: str,
    name2: Optional[str] = None ,
    peakname: Optional[str] = None ,
    test_method: _Method_rank_peak_groups = "binomtest"
) -> Union[List[float], float]:

    """\
    Comaparing the peak difference between two groups for specific peak.

    :param adata_ccf:
        Annotated data matrix.
    :param name1:
        The name of the first group
    :param name2:
        The name of the second group
    :param peakname:
        The name of the peak for comparing
    :param method:
        The default method is `'binomtest'`, `'binomtest'` uses binomial test, `'binomtest2'` uses
        binomial test but stands on different hypothesis of `'binomtest'`, `'fisher_exact'` uses
        fisher exact test,`'chisquare'` uses chi-squeare test.

    :return:
        Pvalue for the specific hypothesis.

    :example:
    >>> import pycallingcards as cc
    >>> adata_ccf = cc.datasets.mousecortex_CCF()
    >>> cc.tl.diff2group(adata_ccf, 'Neuron_Excit_L5_Mixed','Astrocyte','chr2_28188592_28188996')
    """

    if peakname != None:

        cluster1 = adata_ccf[(adata_ccf.obs[["cluster"]] == name1)["cluster"],
                             adata_ccf.var.index.get_loc(peakname)].X
        if name2 == None:
            cluster2 = adata_ccf[(adata_ccf.obs[["cluster"]] != name1)["cluster"],
                                adata_ccf.var.index.get_loc(peakname)].X
        else:
            cluster2 = adata_ccf[(adata_ccf.obs[["cluster"]] == name2)["cluster"],
                                adata_ccf.var.index.get_loc(peakname)].X

        total1 = cluster1.shape[0]
        total2 = cluster2.shape[0]

        if test_method == "binomtest2" or test_method == "fisher_exact":
            number1 = cluster1.nnz
            number2 = cluster2.nnz
        elif test_method == "binomtest" or test_method == "chisquare":
            number1 = cluster1.sum()
            number2 = cluster2.sum()
        else:
            raise ValueError("Please input a correct method: binomtest/binomtest2/fisher_exact/chisquare.")

        return _calculatePvalue(number1,number2,total1,total2,method = test_method)

    else:

        print("No peak name is provided, the pvalue for all the peaks would be returned.")
        pvaluelist = []

        for peak in list(adata_ccf.var.index):


            cluster1 = adata_ccf[(adata_ccf.obs[["cluster"]] == name1)["cluster"],
                             adata_ccf.var.index.get_loc(peak)].X
            cluster2 = adata_ccf[(adata_ccf.obs[["cluster"]] == name2)["cluster"],
                                 adata_ccf.var.index.get_loc(peak)].X

            total1 = cluster1.shape[0]
            total2 = cluster2.shape[0]

            if test_method == "binomtest2" or test_method == "fisher_exact":
                number1 = cluster1.nnz
                number2 = cluster2.nnz
            elif test_method == "binomtest" or test_method == "chisquare":
                number1 = cluster1.sum()
                number2 = cluster2.sum()
            else:
                raise ValueError("Please input a correct method: binomtest/binomtest2/fisher_exact/chisquare.")

            pvaluelist.append(_calculatePvalue(number1,number2,total1,total2,method = test_method))

        return pvaluelist

def rank_peak_groups(
    adata_ccf: AnnData,
    groupby: str,
    use_raw: bool = False,
    groups: Union[Literal['all'], Iterable[str]] = 'all',
    reference: str = None,
    n_peaks: Optional[int] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
    method: _Method_rank_peak_groups = None
) -> Optional[AnnData]:

    """\
    Rank peaks for characterizing groups.

    :param adata_ccf:
        Annotated data matrix.
    :param groupby:
        The key of the observations grouping to consider.
    :param use_raw:
        Use `raw` attribute of `adata` if present.
    :param groups:
        Subset of groups, e.g. [`'g1'`, `'g2'`, `'g3'`], to which comparison
        shall be restricted, or `'all'` (default), for all groups.
    :param reference:
        If `'rest'`, compare each group to the union of the rest of the group.
        If a group identifier, compare with respect to this group.
    :param n_peaks:
        The number of peaks that appear in the returned tables.
        Default includes all peaks.
    :param key_added:
        The key in `adata.uns` information is saved to.
    :param  method:
        The default method is `'binomtest'`,
        `'binomtest'` uses binomial test,
        `'binomtest2'` uses binomial test but stands on different hypothesis of `'binomtest'`,
        `'fisher_exact'` uses fisher exact test,
        `'chisquare'` uses chi-squeare test.


    :return:
        :names: structured `np.ndarray` (`.uns['rank_peaks_groups']`). Structured array to be indexed by group id storing the peak names. Ordered according to scores.
        :pvalues: structured `np.ndarray` (`.uns['rank_peaks_groups']`)
        :number: `pandas.DataFrame` (`.uns['rank_peaks_groups']`). The number of peaks/ or the number of cells contian peaks (depending on the method) expressing the peaks for each group.
        :number_rest: `pandas.DataFrame` (`.uns['rank_peaks_groups']`). The number of peaks/ or the number of cells contian peaks (depending on the method) expressing the peaks for each group in the reference data.
        :total: `pandas.DataFrame` (`.uns['rank_peaks_groups']`). The total number of cells contian peaks expressing the peaks for each group.
        :total_rest: `pandas.DataFrame` (`.uns['rank_peaks_groups']`). The total number of cells contian peaks expressing the peaks for each group in the reference data.

    :example:
    >>> import pycallingcards as cc
    >>> adata_ccf = cc.datasets.mousecortex_CCF()
    >>> cc.tl.rank_peak_groups(adata_ccf,"cluster",method = 'binomtest',key_added = 'binomtest')
    """

    avail_method = ['binomtest', 'binomtest2','fisher_exact', 'chisquare']
    if method == None:
        method = 'binomtest'
    elif method not in avail_method:
        raise ValueError(f'Correction method must be one of {avail_method}.')

    possible_group = list(adata_ccf.obs[groupby].unique())

    if reference == None:
        reference = "rest"
    elif reference not in possible_group:
        raise ValueError(f'Invalid reference, should be all or one of {possible_group}.')

    if groups == 'all':
        group_list = possible_group
    elif type(groups) == str:
        group_list = groups
    elif type(groups) == list:
        group_list = groups
    else:
        raise ValueError("Invalid groups.")

    if key_added == None:
        key_added = 'rank_peak_groups'
    elif type(key_added) != str:
        raise ValueError("key_added should be str.")

    if type(use_raw) != bool:
        print( "use_row should be bool.")

    adata_ccf = adata_ccf.copy() if copy else adata_ccf

    adata_ccf.uns[key_added] = {}
    adata_ccf.uns[key_added]['params'] = dict(
    groupby=groupby,
    reference=reference,
    method=method,
    use_raw=use_raw)

    peak_list  = list(adata_ccf.var.index)

    if n_peaks == None:
        n_peaks = len(peak_list)
    elif type(n_peaks) != int or n_peaks < 1 or n_peaks > len(peak_list):
        raise ValueError("n_peaks should be a int larger than 0 and smaller than the total number of peaks ")


    finalresult_name = np.empty([n_peaks, len(group_list)], dtype='<U100')
    finalresult_pvalue = np.empty([n_peaks, len(group_list)], dtype=float)
    finalresult_number1 = np.empty([n_peaks, len(group_list)], dtype=int)
    finalresult_number2 = np.empty([n_peaks, len(group_list)], dtype=int)
    finalresult_total1 = np.empty([n_peaks, len(group_list)], dtype=int)
    finalresult_total2 = np.empty([n_peaks, len(group_list)], dtype=int)


    i = 0

    for cluster in group_list:

        if reference == "rest":
            clusterdata = adata_ccf[(adata_ccf.obs[["cluster"]] == cluster)["cluster"]]
            clusterdatarest = adata_ccf[(adata_ccf.obs[["cluster"]] != cluster)["cluster"]]
        else:
            clusterdata = adata_ccf[(adata_ccf.obs[["cluster"]] == cluster)["cluster"]]
            clusterdatarest = adata_ccf[(adata_ccf.obs[["cluster"]] == reference)["cluster"]]


        pvaluelist = []
        number1list = []
        number2list = []
        total1list = []
        total2list = []


        total1 = clusterdata.X.shape[0]
        total2  = clusterdatarest.X.shape[0]

        for peak in peak_list:

            cluster1 = clusterdata[:,adata_ccf.var.index.get_loc(peak)].X
            #total1 = cluster1.shape[0]

            cluster2 = clusterdatarest[:,adata_ccf.var.index.get_loc(peak)].X
            #total2 = cluster2.shape[0]

            if method == "binomtest2" or method == "fisher_exact":
                number1 = cluster1.nnz
                number2 = cluster2.nnz
            elif method == "binomtest" or method == "chisquare":
                number1 = cluster1.sum()
                number2 = cluster2.sum()
            else:
                raise ValueError("Please input a correct method: binomtest/binomtest2/fisher_exact/chisquare.")

            pvaluelist.append(_calculatePvalue(number1,number2,total1,total2,method =  method))
            number1list.append(number1)
            number2list.append(number2)
            total1list.append(total1)
            total2list.append(total2)

        pvaluelistnp = np.array(pvaluelist)
        number1listnp = np.array(number1list)
        number2listnp = np.array(number2list)
        total1listnp = np.array(total1list)
        total2listnp = np.array(total2list)


        pvaluelistarg = pvaluelistnp.argsort()


        finalresult_name[:,i] = np.array(peak_list)[pvaluelistarg][:n_peaks]
        finalresult_pvalue[:,i] = pvaluelistnp[pvaluelistarg][:n_peaks]
        finalresult_number1[:,i] = number1listnp[pvaluelistarg][:n_peaks]
        finalresult_number2[:,i] = number2listnp[pvaluelistarg][:n_peaks]
        finalresult_total1[:,i] = total1listnp[pvaluelistarg][:n_peaks]
        finalresult_total2[:,i] = total2listnp[pvaluelistarg][:n_peaks]

        i += 1


    temppvalue = np.array([group_list, ['float']*len(group_list)]).transpose()
    tempname = np.array([group_list, ['<U100']*len(group_list)]).transpose()
    tempnamenumber = np.array([group_list, ['int']*len(group_list)]).transpose()



    adata_ccf.uns[key_added]['names'] = np.rec.array(list(map(tuple, finalresult_name)), dtype=list(map(tuple, tempname)))
    adata_ccf.uns[key_added]['pvalues'] = np.rec.array(list(map(tuple, finalresult_pvalue)), dtype=list(map(tuple, temppvalue)))
    adata_ccf.uns[key_added]['number'] = np.rec.array(list(map(tuple, finalresult_number1)), dtype=list(map(tuple, tempnamenumber)))
    adata_ccf.uns[key_added]['number_rest'] = np.rec.array(list(map(tuple, finalresult_number2)), dtype=list(map(tuple, tempnamenumber)))
    adata_ccf.uns[key_added]['total'] = np.rec.array(list(map(tuple, finalresult_total1)), dtype=list(map(tuple, tempnamenumber)))
    adata_ccf.uns[key_added]['total_rest'] = np.rec.array(list(map(tuple, finalresult_total2)), dtype=list(map(tuple, tempnamenumber)))



    return adata_ccf if copy else None

