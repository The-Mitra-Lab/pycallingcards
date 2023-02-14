from typing import Union

import numpy as np
import pandas as pd
from anndata import AnnData
from liftover import get_lifter
from tqdm import tqdm


def liftover(
    data: Union[AnnData, pd.DataFrame],
    torlerance: int = 10,
    original_genome: str = "mm10",
    new_genome: str = "hg38",
) -> Union[AnnData, pd.DataFrame]:

    """\
    Use `liftover <https://genome.ucsc.edu/cgi-bin/hgLiftOver>`__  to transform from one genome to another :cite:`hinrichs2006ucsc`.

    :param data:
        pd.DataFrame with the first three columns as chromosome, start and end.
        Anndata with peak adata.var contains the first three columns as chromosome, start and end.
    :param torlerance:
        The max multiples allowsfor the length of new_genome to compare with the original one.
    :param original_genome:
        The original genome.
    :param new_genome:
        The new genome.


    :Example:
    >>> import pycallingcards as cc
    >>> adata_cc = cc.datasets.mousecortex_data(data="CC")
    >>> adata_cc = cc.tl.liftover(adata_cc)
    """

    if type(data) == AnnData:
        peak_data = data.var
    elif type(data) == pd.DataFrame:
        peak_data = data
    else:
        ValueError("Please input AnnData object or pd.DataFrame]")

    converter = get_lifter(original_genome, new_genome)

    finalresult = []
    for peak in tqdm(range(len(peak_data))):

        temp_result = []
        for i in range(int(peak_data.iloc[peak, 1]), int(peak_data.iloc[peak, 2])):
            re = converter[peak_data.iloc[peak, 0][3:]][i]
            if re != []:
                temp_result.append(np.array(re)[0])
                break
        for i in range(int(peak_data.iloc[peak, 2]), int(peak_data.iloc[peak, 1]), -1):
            re = converter[peak_data.iloc[peak, 0][3:]][i]
            if re != []:
                temp_result.append(np.array(re)[0])
                break

        if len(temp_result) > 1:

            length = abs(int(temp_result[1][1]) - int(temp_result[0][1]))

            if (
                torlerance
                * (int(peak_data.iloc[peak, 2]) - int(peak_data.iloc[peak, 1]))
                > length
                and temp_result[1][0] == temp_result[0][0]
            ):
                if temp_result[1][1] <= temp_result[0][1]:
                    finalresult.append(
                        [temp_result[1][0], temp_result[1][1], temp_result[0][1]]
                    )
                else:
                    finalresult.append(
                        [temp_result[1][0], temp_result[0][1], temp_result[1][1]]
                    )
            else:
                finalresult.append(["", "", ""])
        else:
            finalresult.append(["", "", ""])

    finalresult = pd.DataFrame(finalresult).set_index(peak_data.index)
    finalresult.columns = [
        "Chr_liftover",
        "Start_liftover",
        "End_liftover",
    ]

    if type(data) == AnnData:
        data.var = pd.concat([peak_data, finalresult], axis=1)
        return data
    elif type(data) == pd.DataFrame:
        return pd.concat([peak_data, finalresult], axis=1)


def find_location(
    data: pd.DataFrame,
    original_name: str,
    new_name: str = None,
    genome: str = "hg38",
) -> pd.DataFrame:

    """\
    Find the gene location for a specfic genome.

    :param data:
        pd.DataFrame with the first three columns as chromosome, start and end.
    :param original_name: .
        The name of the target column.
    :param new_name:
        The new name, default is the genome name.
    :param genome:
        The genome to search for .

    :Example:
    >>> import pycallingcards as cc
    >>> adata_cc = cc.datasets.mousecortex_data(data="CC")
    >>> adata_cc = cc.tl.find_location(adata_cc.var,'Gene Name1')
    """

    gene_names = list(data[original_name])

    newdata = []
    if genome == "mm10":

        refdata = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.mm10.Sorted.bed",
            sep="\t",
            header=None,
        )

    elif genome == "hg38":

        refdata = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.hg38.Sorted.bed",
            sep="\t",
            header=None,
        )

    elif genome == "sacCer3":

        refdata = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/refGene.sacCer3.Sorted.bed",
            sep="\t",
            header=None,
        )

    for gene_number in tqdm(range(len(gene_names))):

        gene = gene_names[gene_number]
        data_gene = refdata[refdata[4] == gene.upper()]

        if len(data_gene[0].unique()) != 1:
            newdata.append(["", "", ""])

        else:
            newdata.append(
                [data_gene[0].iloc[0], str(data_gene[1].min()), str(data_gene[2].max())]
            )

    if new_name == None:
        new_name = genome

    newdata = pd.DataFrame(newdata).set_index(data.index)
    newdata.columns = [
        "Chr_" + new_name,
        "Start_" + new_name,
        "End_" + new_name,
    ]

    return pd.concat([data, newdata], axis=1)


def result_mapping(
    data: pd.DataFrame,
    torlerance: int = 10,
    original_genome: str = "mm10",
    new_genome: str = "hg38",
) -> pd.DataFrame:

    """\
    Map from one genome to another for the result table :cite:`hinrichs2006ucsc`.

    :param data:
        pd.DataFrame of result. It contains 'Peak' and 'Gene'.
    :param torlerance:
        The max multiples allows for the length of new_genome to be compared with to the original one.
    :param original_genome:
        The original genome.
    :param new_genome:
        The new genome.

    :Example:
    >>> import pycallingcards as cc
    >>> import scanpy as sc
    >>> adata_cc = sc.read("Mouse-Cortex_cc.h5ad")
    >>> adata = cc.datasets.mousecortex_data(data="RNA")
    >>> qbed_data = cc.datasets.mousecortex_data(data="qbed")
    >>> peak_data = cc.pp.callpeaks(qbed_data, method = "CCcaller", reference = "mm10",  maxbetween = 2000, pvalue_cutoff = 0.01,
                lam_win_size = 1000000,  pseudocounts = 1, record = True)
    >>> peak_annotation = cc.pp.annotation(peak_data, reference = "mm10")
    >>> peak_annotation = cc.pp.combine_annotation(peak_data,peak_annotation)
    >>> sc.tl.rank_genes_groups(adata,'cluster')
    >>> result = cc.tl.pair_peak_gene_sc(adata_cc,adata,peak_annotation)
    >>> result = cc.tl.result_mapping(result)
    """

    data_temp = pd.concat(
        [data["Peak"].str.split("_", n=2, expand=True).set_index(data.index), data],
        axis=1,
    )

    print("Start mapping the peaks to the new genome.")
    data_temp = liftover(data_temp, torlerance, original_genome, new_genome)

    print("Start finding location of genes in the new genome.")
    data_temp = find_location(data_temp, "Gene", genome=new_genome)

    return data_temp.drop([0, 1, 2], axis=1)
