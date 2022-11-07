from pycallingcards.raw_processing.parse_bam import parse_bam
from argparse import Namespace
import os

import logging

def test_parse_bam_with_qc(barcode_details_file,genome_file,bam_file,tmpdir, caplog):

	caplog.set_level(logging.INFO)
	args = Namespace(input=bam_file, 
					 barcode_details=barcode_details_file, 
					 genome=genome_file, 
					 output_prefix=str(tmpdir),
					 record_barcode_qc=True,
					  mapq_threshold=10)
	parse_bam(args)
	# todo -- test that the number of hops recorded in the qbed 
	# is the same as the number of reads in the passing bam
	# note this is 3 b/c the summarize QC is not implemented at the moment
	assert len(os.listdir(tmpdir)) == 4
	
def test_parse_bam_without_qc(barcode_details_file,genome_file,bam_file,tmpdir,caplog):
	caplog.set_level(logging.INFO)
	args = Namespace(input=bam_file, 
					 barcode_details=barcode_details_file, 
					 genome=genome_file, 
					 output_prefix=str(tmpdir),
					 record_barcode_qc=False,
					  mapq_threshold=10)
	parse_bam(args)
	# todo -- test that the number of hops recorded in the qbed 
	# is the same as the number of reads in the passing bam
	assert len(os.listdir(tmpdir)) == 3



