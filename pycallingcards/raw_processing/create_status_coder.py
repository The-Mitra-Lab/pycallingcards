
#pylint:disable=W1203,W0102
import logging
from typing import Callable

from pysam import AlignedSegment

from .StatusFlags import StatusFlags

__all__ = ['create_status_coder']

logging.getLogger(__name__).addHandler(logging.NullHandler())

def create_status_coder(insert_seqs:list=['*'],mapq_threshold:int=10) -> Callable[[AlignedSegment], int]:
	"""A factory function which returns a function capable of determining 
	the status code of a read tagged by an AlignmentTagger object

	Args:
		insert_seqs (list): a list of acceptable insert sequences. Defaults to 
		['*'], which will skip the insert seq check altogether.
		mapq_threshold (int): a mapq_threshold. Less than this value will be 
		marked as failing the mapq threshold test. Default is 10.

	Returns:
		Callable[[pysam.AlignedSegment], int]: A function which given a tagged 
		pysam AlignedSegment will return the status code for a the read
	"""

	def coder(tagged_read:AlignedSegment):
		status_code = 0
		# if the read is unmapped, add the flag, but don't check 
		# other alignment metrics
		if tagged_read.is_unmapped:
			status_code += StatusFlags.UNMAPPED.flag()
		else:
			if tagged_read.is_qcfail:
				status_code += StatusFlags.ALIGNER_QC_FAIL.flag()
			if tagged_read.is_secondary or tagged_read.is_supplementary:
				status_code += StatusFlags.NOT_PRIMARY.flag()
			if tagged_read.mapping_quality < mapq_threshold:
				status_code +=  StatusFlags.MAPQ.flag()
			# if the read is clipped on the 5' end, flag
			if (tagged_read.is_forward and tagged_read.query_alignment_start != 0) \
				or (tagged_read.is_reverse and tagged_read.query_alignment_end != \
					tagged_read.infer_query_length()):
				status_code += StatusFlags.FIVE_PRIME_CLIP.flag()
		# check the insert sequence
		try:
			if insert_seqs != ["*"]:
				if tagged_read.get_tag("XZ") not in insert_seqs:
						status_code += StatusFlags.INSERT_SEQ.flag()
		except AttributeError as exc:
			logging.debug(f"insert sequence not found in Barcode Parser. {exc}")
		
		return status_code
	
	return coder