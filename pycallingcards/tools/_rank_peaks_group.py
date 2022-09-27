
from ast import Raise
import numpy as np
from anndata import AnnData
from typing import Union, Optional, List,  Iterable,  Literal


_Method_rank_peak_groups = Optional[Literal['binomtest', 'binomtest2','fisher_exact']]


def DE_pvalue(
    number1: int,
    number2: int,
    total1: int,
    total2: int,
    method: _Method_rank_peak_groups = 'fisher_exact'):

    """\
    Comaparing the peak difference between two groups for specific peak.

    :param number1:
        The total number of htops (or the number of cells containing htops) in group 1.
    :param number2:
        The total number of htops (or the number of cells containing htops) in group 2.
    :param total1:
        The total number of cells in group 1.
    :param total2:
        The total number of cells in group 2.
    :param peakname:
        The name of the peak for comparing.
    :param copy: Default is `False`.
        Whether to modify copied input object. 
    :param method: ['binomtest', 'binomtest2','fisher_exact']. Default is `fisher_exact`
        The default method is 'fisher_exact', `binomtest` uses binomial test, `binomtest2` uses
        binomial test but stands on different hypothesis of `binomtest`, 'fisher_exact' uses
        fisher exact test.
    

    :return:
        Pvalue for the specific hypothesis.

    :example:
    >>> import pycallingcards as cc
    >>> adata_ccf = cc.tl.DE_pvalue(10,456,261,491)

    """


    if method == "binomtest":

        from scipy.stats import binomtest

        if number1 + number2 ==0:
            return 1
        else:
            return binomtest(int(number1), n=int(number1+number2), p=float(total1/(total1+total2)), alternative='greater').pvalue

    if method == "binomtest2":

        from scipy.stats import binomtest

        return binomtest(int(number1), n=total1, p=number2/total2, alternative='greater').pvalue
                 

    elif method == "fisher_exact":

        table = np.array([[number1, number2], [total1- number1, total2 - number2]])
        from scipy.stats import fisher_exact
        _,p = fisher_exact(table, alternative='greater')

        return p
    else:

        raise ValueError("Please input a correct method: binomtest/binomtest2/fisher_exact.")


def diff2group_bygroup(
    adata_ccf: AnnData,
    groupby: str,
    name1: str,
    name2: Optional[str] = None ,
    peakname: Optional[str] = None ,
    test_method: _Method_rank_peak_groups = "fisher_exact"
) -> Union[List[float], float]:

    """\
    Comaparing the peak difference between two groups for specific peak by group.


    :param adata_ccf:
        Annotated data matrix.
    :param groupby:
        The key in adata_ccf of the observation groups to consider.
    :param name1:
        The name of the first group.
    :param name2: Default is `None`.
        The name of the second group.
    :param peakname: Default is `None`.
        The name of the peak used for comparing.
    :param test_method: ['binomtest', 'binomtest2','fisher_exact']. Default is `fisher_exact`
        `binomtest` uses binomial test, `binomtest2` uses
        binomial test but stands on a different hypothesis of `binomtest`, `fisher_exact` uses
        fisher exact test.


    :return:
        Pvalue for the specific hypothesis.


    :example:
    >>> import pycallingcards as cc
    >>> adata_ccf = cc.datasets.mousecortex_CCF()
    >>> cc.tl.diff2group_bygroup(adata_ccf, 'Neuron_Excit_L5_Mixed','Astrocyte','chr2_28188592_28188996')
    """

    if peakname != None:

        cluster1 = adata_ccf[(adata_ccf.obs[[groupby]] == name1)[groupby],
                             adata_ccf.var.index.get_loc(peakname)].X
        if name2 == None:
            cluster2 = adata_ccf[(adata_ccf.obs[[groupby]] != name1)[groupby],
                                adata_ccf.var.index.get_loc(peakname)].X
        else:
            cluster2 = adata_ccf[(adata_ccf.obs[[groupby]] == name2)[groupby],
                                adata_ccf.var.index.get_loc(peakname)].X

        total1 = cluster1.shape[0]
        total2 = cluster2.shape[0]

        if test_method == "binomtest2" or test_method == "fisher_exact":
            number1 = cluster1.nnz
            number2 = cluster2.nnz
        elif test_method == "binomtest":
            number1 = cluster1.sum()
            number2 = cluster2.sum()
        else:
            raise ValueError("Please input a correct method: binomtest/binomtest2/fisher_exact.")

        return DE_pvalue(number1,number2,total1,total2,method = test_method)

    else:

        print("No peak name is provided, the pvalue for all the peaks would be returned.")
        pvaluelist = []

        for peak in list(adata_ccf.var.index):


            cluster1 = adata_ccf[(adata_ccf.obs[[groupby]] == name1)[groupby],
                             adata_ccf.var.index.get_loc(peak)].X
            cluster2 = adata_ccf[(adata_ccf.obs[[groupby]] == name2)[groupby],
                                 adata_ccf.var.index.get_loc(peak)].X

            total1 = cluster1.shape[0]
            total2 = cluster2.shape[0]

            if test_method == "binomtest2" or test_method == "fisher_exact":
                number1 = cluster1.nnz
                number2 = cluster2.nnz
            elif test_method == "binomtest" :
                number1 = cluster1.sum()
                number2 = cluster2.sum()
            else:
                raise ValueError("Please input a correct method: binomtest/binomtest2/fisher_exact.")

            pvaluelist.append(DE_pvalue(number1,number2,total1,total2,method = test_method))

        return pvaluelist


        
def diff2group_bysample(
    adata_ccf: AnnData,
    name1: str,
    name2: Optional[str] = None ,
    peakname: Optional[str] = None ,
    test_method: _Method_rank_peak_groups = "binomtest"
) -> Union[List[float], float]:

    """\
    Comaparing the peak difference between two groups for a specific peak by sample.


    :param adata_ccf:
        Annotated data matrix.
    :param groupby:
        The key in adata_ccf of the observation groups to consider.
    :param name1:
        The name of the first group.
    :param name2: Default is `None`.
        The name of the second group.
    :param peakname: Default is `None`.
        The name of the peak used for comparing.
    :param test_method: ["binomtest", "binomtest2","fisher_exact"]. Default is `fisher_exact`
        `binomtest` uses binomial test, `binomtest2` uses
        binomial test but stands on a different hypothesis of `binomtest`, `fisher_exact` uses
        fisher exact test.


    :return:
        Pvalue for the specific hypothesis.


    """

    if peakname != None:

        cluster1 = adata_ccf[name1,:].X

        if name2 == None:
            name2 = list(set(adata_ccf.obs.index).difference(set([name1])))
            cluster2 = adata_ccf[name2,:].X
        else:
            cluster2 = adata_ccf[name2,:].X

        total1 = int(cluster1.sum())
        total2 = int(cluster2.sum())

        number1 = adata_ccf[name1,peakname].X[0,0]
        number2 = adata_ccf[name2,peakname].X.sum()

        return DE_pvalue(number1,number2,total1,total2,method = test_method)

    else:

        print("No peak name is provided, the pvalue for all the peaks would be returned.")
        pvaluelist = []

        cluster1 = adata_ccf[name1,:].X

        if name2 == None:
            name2 = list(set(adata_ccf.obs.index).difference(set([name1])))
            cluster2 = adata_ccf[name2,:].X
        else:
            cluster2 = adata_ccf[name2,:].X

        total1 = int(cluster1.sum())
        total2 = int(cluster2.sum())

        for peak in list(adata_ccf.var.index):

            number1 = adata_ccf[name1,peak].X[0,0]
            number2 = adata_ccf[name2,peak].X.sum()


            pvaluelist.append(DE_pvalue(number1,number2,total1,total2,method = test_method))

        return pvaluelist


def rank_peak_groups(
    adata_ccf: AnnData,
    groupby: str,
    groups: Union[Literal['all'], Iterable[str]] = 'all',
    reference: str = None,
    n_peaks: Optional[int] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
    method: _Method_rank_peak_groups = 'fisher_exact'
) -> Optional[AnnData]:

    """\
    Rank peaks for characterizing groups.

    :param adata_ccf:
        Annotated data matrix.
    :param groupby:
        The key of the groups.
    :param groups: Default is `all`.
        Subset of groups (list), e.g. [`'g1'`, `'g2'`, `'g3'`], to which comparison
        shall be restricted, or `all` (default), for all groups.
    :param reference: Defaulf is `rest`.
        If `rest`, compare each group to the union of the rest of the group.
        If a group identifier, compare with respect to this group.
    :param n_peaks:
        The number of peaks that appear in the returned tables.
        Default includes all peaks.
    :param key_added:
        The key in `adata.uns` information is saved to.
    :param  method: ["binomtest", "binomtest2","fisher_exact"]. Default is `fisher_exact`.
        `binomtest` uses binomial test,
        `binomtest2` uses binomial test but stands on a different hypothesis of `binomtest`,
        `fisher_exact` uses fisher exact test.


    :Returns: 
        | **names** - structured `np.ndarray` (`.uns['rank_peaks_groups']`). Structured array is to be indexed by the group ID storing the peak names. Ordered according to scores.
        | **return pvalues** - structured `np.ndarray` (`.uns['rank_peaks_groups']`)
        | **number** -  `pandas.DataFrame` (`.uns['rank_peaks_groups']`). The number of peaks or the number of cells contian peaks (depending on the method).
        | **number_rest** - `pandas.DataFrame` (`.uns['rank_peaks_groups']`). The number of peaks or the number of cells contianing peaks (depending on the method).
        | **total** -  `pandas.DataFrame` (`.uns['rank_peaks_groups']`). The total number of cells contianing peaks.
        | **total_rest** -  `pandas.DataFrame` (`.uns['rank_peaks_groups']`). The total number of cells contianing peaks.

    :example:
    >>> import pycallingcards as cc
    >>> adata_ccf = cc.datasets.mousecortex_data(data="CCF")
    >>> cc.tl.rank_peak_groups(adata_ccf,groupby,method = 'binomtest',key_added = 'binomtest')
    """

    if groupby == "Index":
        _rank_peak_groups_bysample( adata_ccf = adata_ccf, groups = groups, 
        reference = reference, n_peaks = n_peaks, key_added = key_added,copy = copy, method = method)
    else:
        _rank_peak_groups_bygroup( adata_ccf = adata_ccf, groupby = groupby, groups = groups, 
        reference = reference, n_peaks = n_peaks, key_added = key_added,copy = copy, method = method)
 

def _rank_peak_groups_bygroup(
    adata_ccf: AnnData,
    groupby: str,
    groups: Union[Literal['all'], Iterable[str]] = 'all',
    reference: str = None,
    n_peaks: Optional[int] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
    method: _Method_rank_peak_groups = None
) -> Optional[AnnData]:

    avail_method = ['binomtest', 'binomtest2','fisher_exact']
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



    adata_ccf = adata_ccf.copy() if copy else adata_ccf

    adata_ccf.uns[key_added] = {}
    adata_ccf.uns[key_added]['params'] = dict(
    groupby=groupby,
    reference=reference,
    method=method)

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
            clusterdata = adata_ccf[(adata_ccf.obs[[groupby]] == cluster)[groupby]]
            clusterdatarest = adata_ccf[(adata_ccf.obs[[groupby]] != cluster)[groupby]]
        else:
            clusterdata = adata_ccf[(adata_ccf.obs[[groupby]] == cluster)[groupby]]
            clusterdatarest = adata_ccf[(adata_ccf.obs[[groupby]] == reference)[groupby]]


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
            elif method == "binomtest":
                number1 = cluster1.sum()
                number2 = cluster2.sum()
            else:
                raise ValueError("Please input a correct method: binomtest/binomtest2/fisher_exact.")

            pvaluelist.append(DE_pvalue(number1,number2,total1,total2,method =  method))
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



def _rank_peak_groups_bysample(
    adata_ccf: AnnData,
    groups: Union[Literal['all'], Iterable[str]] = 'all',
    reference: str = None,
    n_peaks: Optional[int] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
    method: _Method_rank_peak_groups = None
) -> Optional[AnnData]:

    avail_method = ['binomtest', 'binomtest2','fisher_exact']
    if method == None:
        method = 'binomtest'
    elif method not in avail_method:
        raise ValueError(f'Correction method must be one of {avail_method}.')

    possible_group = list(adata_ccf.obs.index.unique())

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



    adata_ccf = adata_ccf.copy() if copy else adata_ccf

    adata_ccf.uns[key_added] = {}
    adata_ccf.uns[key_added]['params'] = dict(
    groupby="Index",
    reference=reference,
    method=method)

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

        cluster1 = adata_ccf[cluster,:].X

        if reference == "rest":
            name2 = list(set(adata_ccf.obs.index).difference(set([cluster])))
            cluster2 = adata_ccf[name2,:].X
        else:
            name2 = reference
            cluster2 = adata_ccf[name2,:].X

        total1 = int(cluster1.sum())
        total2 = int(cluster2.sum())

        pvaluelist = []
        number1list = []
        number2list = []
        total1list = []
        total2list = []



        for peak in peak_list:

            number1 = adata_ccf[cluster,peak].X[0,0]
            number2 = adata_ccf[name2,peak].X.sum()

            pvaluelist.append(DE_pvalue(number1,number2,total1,total2,method =  method))
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

