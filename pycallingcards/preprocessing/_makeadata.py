from typing import Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import tqdm
from anndata import AnnData
from scipy.sparse import csr_matrix, lil_matrix

_reference = Optional[Literal["hg38", "mm10", "sacCer3"]]


def _myFuncsorting(e):
    try:
        return int(e.split("_")[0][3:])
    except:
        return int(ord(e.split("_")[0][3:]))


def make_Anndata(
    qbed: pd.DataFrame,
    peaks: pd.DataFrame,
    barcodes: Union[pd.DataFrame, List],
    reference: _reference = "hg38",
    key: Union[str, int] = "Barcodes",
) -> AnnData:

    """\
    Make cell(sample) by using peak anndata for calling cards.

    :param qbed:
        pd.DataFrame the first five with columns as chromosome, start, end, reads number, direction and barcodes.
        Chromosome, start, end and barcodes are the actual information needed.
    :param peaks:
        pd.DataFrame with first three columns as chromosome, start and end. Other information is contained after these.
    :param barcodes:
        pd.DataFrame or a list of all barcodes.
    :param reference: `['hg38','mm10','sacCer3']`.
        This information is only used to calculate the length of one insertion.
        `hg38` and `mm10` are the same.
    :param key:
        The name of the column in qbed file containing the barcodes information.



    :Returns:
        Annotated data matrix, where observations (cells/samples) are named by their barcode and
        variables/peaks by Chr_Start_End. The matrix stores the following information.

        | **anndata.AnnData.X** - Where the data matrix is stored
        | **anndata.AnnData.obs_names** -  Cell(sample) names
        | **anndata.AnnData.var_names** -  Peak names
        | **anndata.AnnData.var['peak_ids']** -  Peak information from the original file
        | **anndata.AnnData.var['feature_types']** -  Feature types


    :Example:
    >>> import pycallingcards as cc
    >>> cc_data = cc.datasets.mousecortex_data(data="qbed")
    >>> peak_data = cc.pp.callpeaks(cc_data, method = "test", reference = "mm10",  record = True)
    >>> barcodes = cc.datasets.mousecortex_data(data="barcodes")
    >>> adata_cc = cc.pp.makeAnndata(cc_data, peak_data, barcodes)


    """
    if reference == "hg38" or reference == "mm10":
        length = 3
    elif reference == "sacCer3":
        length = 0

    if type(barcodes) == list:
        barcodes = pd.DataFrame(barcodes, columns=["Index"])

    barcodes = barcodes.drop_duplicates()
    barcodesorigin = barcodes.copy()
    barcodes = list(barcodes.iloc[:, 0])
    barcodes_dict = {}
    for i in range(len(barcodes)):
        barcodes_dict[barcodes[i]] = i

    # peaks to order

    peaks1 = peaks.copy()
    peaks1["name"] = peaks1.iloc[:, [0, 1, 2]].T.astype(str).agg("_".join)

    peak_name_unique = list()

    for number in list(peaks1["name"]):
        if number not in peak_name_unique:
            peak_name_unique.append(number)
    if reference != "sacCer3":
        peak_name_unique.sort(key=_myFuncsorting)
    else:
        peak_name_unique.sort()

    peak_name_dict = {}
    for i in range(len(peak_name_unique)):
        peak_name_dict[peak_name_unique[i]] = i

    # create an empty matrix to store peaks * cell
    cellpeaks = lil_matrix((len(peak_name_unique), len(barcodes)), dtype=np.float32)

    # pairing
    chrolist = list(peaks1.iloc[:, 0].unique())

    peaksf = peaks1.iloc[:, [0, 1, 2, -1, -1]].to_numpy()

    for i in range(len(peaksf)):
        peaksf[i, 4] = peak_name_dict[peaksf[i, 3]]

    ccf1 = qbed.copy()

    if key == None:
        key = "Barcodes"

    ccfbar = (ccf1[key].apply(lambda x: barcodes_dict[x])).to_numpy()
    ccff = ccf1.iloc[:, [0, 1]].to_numpy()

    ccff = np.concatenate(
        (ccff, ccf1[key].to_numpy().reshape((-1, 1)), ccfbar.reshape((-1, 1))), axis=1
    )

    for chro in tqdm.tqdm(chrolist):

        peaksfchr = np.array(peaksf[peaksf[:, 0] == chro])
        ccffchr = np.array(ccff[ccff[:, 0] == chro])
        ccffchr = ccffchr[ccffchr[:, 1].argsort(), :]

        for i in range(len(peaksfchr)):
            ptemp = []
            start = 0

            for num in range(len(ccffchr)):

                if (ccffchr[num, 1] >= (peaksfchr[i, 1] - length)) and (
                    ccffchr[num, 1] <= (peaksfchr[i, 2])
                ):
                    start = 1

                    ptemp.append(ccffchr[num, 3])
                else:
                    if start == 1:
                        ccffchr = ccffchr[num:, :]
                        break

            for j in ptemp:
                cellpeaks[peaksfchr[i, 4], j] += 1

    peaks1 = peaks1.set_index(keys="name")
    peaks1 = peaks1.loc[peak_name_unique]

    adata = ad.AnnData(
        csr_matrix(cellpeaks).T,
        obs=barcodesorigin.set_index(keys=barcodesorigin.columns[0]),
        var=peaks1,
    )

    return adata


_method = Optional[
    Literal["avginsertions", "logAvginsertions", "suminsertions", "logSuminsertions"]
]


def adata_insertions(
    adata_cc: AnnData,
    adata: AnnData,
    name: str,
    groupby: str = "cluster",
    method: _method = "logAvginsertions",
    peak: str = "all",
):

    """\
    Calculate sum of peaks per cluster or average peaks per cell in different cluster and give it to anndata object.

    :param adata_cc:
        Anndata for callingcards
    :param adata:
        Anndata for RNA.
    :param name:
        The name to add to adata.obs.
    :param groupby:
        The name all the cells are grouped by.
    :param method: `["avginsertions","logAvginsertions","suminsertions","logSuminsertions"]`.
        The method to calculate the insertions.
    :param peak:
        The name of the peak we are looking into. If it is `'all'`, all the peaks would be counted.



    :return:
        pd.DataFrame with paired genes and peaks for different groups.

    :example:
    Please check `here <https://pycallingcards.readthedocs.io/en/latest/tutorials/notebooks/Mouse_cortex_Example.html>`__.

    """

    possible_group = list(adata_cc.obs[groupby].unique())
    my_dictionary = {}

    for groupname in possible_group:

        if peak == "all":
            tempadata = adata_cc[(adata_cc.obs[[groupby]] == groupname)[groupby], :].X
        else:
            tempadata = adata_cc[
                (adata_cc.obs[[groupby]] == groupname)[groupby], peak
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

    adata.obs[name] = adata.obs[groupby]
    adata.obs = adata.obs.replace({name: my_dictionary})
    adata.obs[name] = adata.obs[name].astype(float)

    return adata
