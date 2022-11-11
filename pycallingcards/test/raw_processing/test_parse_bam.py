from pycallingcards.raw_processing.parse_bam import parse_bam
from argparse import Namespace
import os

def test_parse_bam(barcode_details_file,genome_file,bam_file,tmpdir):
	args = Namespace(input=bam_file, 
	                 barcode_details=barcode_details_file, 
					 genome=genome_file, 
					 output_prefix=str(tmpdir),
					  mapq_threshold=10)
	parse_bam(args)
    # todo -- test that the number of hops recorded in the qbed 
	# is the same as the number of reads in the passing bam
	assert len(os.listdir(tmpdir)) == 4

