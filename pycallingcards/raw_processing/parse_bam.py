#pylint:disable=W1203
import os
from argparse import Namespace
import logging

import pysam

from .AlignmentTagger import AlignmentTagger
from .ReadRecords import ReadRecords
from .StatusFlags import StatusFlags
from .create_status_coder import create_status_coder

__all__ = ['parse_bam']

logging.getLogger(__name__).addHandler(logging.NullHandler())

def parse_bam(args:Namespace) -> None:
	
	logging.info(f'beginning to parse {args.input}')
	output_bampath_dict = \
		{k: os.path.join(
			args.output_prefix,
			os.path.splitext(os.path.basename(args.input))[0]+'_'+k+'.bam') \
			for k in ['passing','failing']}
	
	qbed_output = os.path.join(
		args.output_prefix,
		os.path.splitext(os.path.basename(args.input))[0]+ '.qbed')

	# open the bam file
	bam_in = pysam.AlignmentFile(args.input)#pylint:disable=E0602

	# open output files for tagged bam reads
	output_bam_dict = \
		{k:pysam.AlignmentFile(v,"wb",header = bam_in.header) \
			for k,v in output_bampath_dict.items()}  
	
	# create a list to hold records which will form the qBed
	read_records = ReadRecords()
	
	# create an AlignmentTagger object
	at = AlignmentTagger(args.barcode_details, args.genome)#pylint:disable=C0103

	status_coder = create_status_coder(at.insert_seq,args.mapq_threshold)
	logging.info("iterating over bam file...")
	for read in bam_in.fetch():
		# only look at mapped primary alignments
		if not read.is_secondary and \
			not read.is_supplementary and \
				not read.is_unmapped:
			# parse the barcode, tag the read
			tagged_read = at.tag_read(read)
			
			status = status_coder(tagged_read['read'])
			# add the barcode status flag to status if the barcode is failing
			if not tagged_read['barcode_details']['passing']:
				status = status + StatusFlags.BARCODE.flag()
			# add the data to the qbed and qc records
			read_records.add_read_info(tagged_read,status)
			# write the read out to the appropriate tagged bamfile
			if status == 0:
				output_bam_dict['passing'].write(tagged_read['read'])
			else:
				output_bam_dict['failing'].write(tagged_read['read'])
	
	logging.info("writing qBed...")
	read_records.to_qbed(qbed_output)

	logging.info(f"file: {args.input} complete!")