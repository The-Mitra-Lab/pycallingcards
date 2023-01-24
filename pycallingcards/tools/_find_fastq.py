def find_fastq(chrm: str, start: int, end: int, path: str) -> str:

    """\
    Find the fastq file of specific part of gene.

    :param chrm:
        The chromosome to search.
    :param start:
        Start site of the chomosome.
    :param end:
        End site of the chomosome.
    :param path:
        The path of data.
        Please download the reference data from `mm10 <https://hgdownload.cse.ucsc.edu/goldenpath/mm10/chromosomes/>`__ , `hg38 <https://hgdownload.cse.ucsc.edu/goldenpath/hg38/chromosomes/>`__ or `sacCer3 <https://hgdownload.cse.ucsc.edu/goldenpath/sacCer3/chromosomes/>`__ .
        Then unzip the fa data before using this function.

    """

    with open(path + "/" + chrm + ".fa") as f:
        s = f.read()
        s = s.replace("\n", "")
        s = s[(len(chrm) + 1) :]

        return s[start - 1 : end]
