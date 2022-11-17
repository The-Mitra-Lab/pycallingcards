"""Single-cell calling cards in Python"""

from ._rank_peaks_group import rank_peak_groups, diff2group_bygroup, diff2group_bysample, DE_pvalue
from ._find_related_genes import pair_peak_gene_bulk,pair_peak_gene_sc
