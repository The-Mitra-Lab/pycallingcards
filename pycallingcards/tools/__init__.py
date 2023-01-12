"""Calling cards data analysis in Python"""

from ._call_motif import call_motif
from ._find_fastq import find_fastq
from ._find_related_genes import pair_peak_gene_bulk, pair_peak_gene_sc
from ._footprint import footprint
from ._liftover import get_lifter, liftover, result_mapping
from ._rank_peaks_group import (
    DE_pvalue,
    diff2group_bygroup,
    diff2group_bysample,
    rank_peak_groups,
)
