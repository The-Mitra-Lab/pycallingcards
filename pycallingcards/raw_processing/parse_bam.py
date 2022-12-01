#pylint:disable=W1203
import os
from argparse import Namespace
import logging
import tempfile

import pysam

from .AlignmentTagger import AlignmentTagger
from .ReadRecords import ReadRecords
from .StatusFlags import StatusFlags
from .create_status_coder import create_status_coder

__all__ = ['parse_bam']

logging.getLogger(__name__).addHandler(logging.NullHandler())

def parse_bam(args: Namespace) -> None:
	# Check input paths
	logging.info('checking input...')
	input_path_list = [args.input,
					   args.barcode_details,
					   args.genome]
	if args.output_prefix:
		input_path_list.append(args.output_prefix)
	for input_path in input_path_list:
		if not os.path.exists(input_path):
			error_msg = f"Input file DNE: {input_path}"
			logging.debug(error_msg)
			raise FileNotFoundError(error_msg)
	
	logging.info(f'beginning to parse {args.input}')
	output_bampath_dict = \
		{k: os.path.join(
			args.output_prefix,
			os.path.splitext(os.path.basename(args.input))[0]+'_'+k+'.bam') 
			for k in ['passing', 'failing']}
	
	logging.info(args)
	
	qbed_output = os.path.join(
		args.output_prefix,
		os.path.splitext(os.path.basename(args.input))[0] + '.qbed')
	qc_output = os.path.join(
		args.output_prefix,
		os.path.splitext(os.path.basename(args.input))[0] + '_qc.tsv')

	logging.info(f"qbed output: {qbed_output}")
	logging.info(f"qc_output: {qc_output}")

	# open the bam file
	bam_in = pysam.AlignmentFile(args.input)#pylint:disable=E0602

	# open output files for tagged bam reads
	output_bam_dict = \
		{k:pysam.AlignmentFile(v,"wb",header = bam_in.header)
			for k, v in output_bampath_dict.items()}
		
	# create an AlignmentTagger object
	at = AlignmentTagger(args.barcode_details, args.genome)#pylint:disable=C0103

	# create a temp directory which will be destroyed when this context block 
	# exits
	with tempfile.TemporaryDirectory() as tmp_dir:
		# create tmp files and use them to instantiate the ReadRecords object
		qbed_tmp_file = os.path.join(tmp_dir, "tmp_qbed.tsv")
		qc_tmp_file = os.path.join(tmp_dir, "tmp_qc.tsv")
		# instantiate ReadRecords object to handle creating qbed, qc files
		read_records = ReadRecords(qbed_tmp_file,qc_tmp_file)
		# create a status coder object to handle adding status codes to the reads
		status_coder = create_status_coder(at.insert_seq,args.mapq_threshold)
		logging.info("iterating over bam file...")
		for read in bam_in.fetch():
			# only look at mapped primary alignments
			if not read.is_secondary and \
				not read.is_supplementary and \
					not read.is_unmapped:
				# parse the barcode, tag the read
				tagged_read = at.tag_read(read)
				# eval the read based on quality expectations, get the status
				status = status_coder(tagged_read['read'])
				# add the barcode status flag to status if the barcode is failing
				if not tagged_read['barcode_details']['passing']:
					status = status + StatusFlags.BARCODE.flag()
				# add the data to the qbed and qc records
				read_records\
					.add_read_info(
						tagged_read,
						status,
						insert_offset=at.insert_length,
						annotation_tags=at.annotation_tags,
						record_barcode_qc=args.record_barcode_qc)
				# write the read out to the appropriate tagged bamfile
				if status == 0:
					output_bam_dict['passing'].write(tagged_read['read'])
				else:
					output_bam_dict['failing'].write(tagged_read['read'])
		
		logging.info("writing qBed...")
		read_records.to_qbed(qbed_output)
		if args.record_barcode_qc:
			logging.info("writing qc...")
			read_records.summarize_qc(qc_output)

	logging.info(f"file: {args.input} complete!")