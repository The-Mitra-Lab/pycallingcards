
import argparse
from importlib.metadata import version

__all__=['parse_args']

def parse_args():
	# create the top-level parser

	script_descriptions = {
		'parse_bam':"Parse the output of the processing pipeline up "+\
			"to calling significant peaks" # TODO write an better description
	}
    
	# common options -- these can be applied to all scripts via the 'parent'---- 
	# argument -- see subparsers.add_parser for parse_bam below ----------------
	common_args = argparse.ArgumentParser(prog="pycallingcards", add_help=False)
	common_args_group = common_args.add_argument_group('general')
	common_args_group.add_argument(
		"-l",
		"--log_level",
		choices=("critical", "error", "warning", "info", "debug"),
		default="warning")
    
	# Create a top level parser ------------------------------------------------
	parser = argparse.ArgumentParser(
		prog='pycallingcards',
		description=f"pycallingcards: {version('pycallingcards')}")
	# add argument to get version
	parser.add_argument(
		"-v", 
		"--version", 
		action='version', 
		version='%(prog)s '+f'{version("pycallingcards")}')
	
	# parse_bam subparser ------------------------------------------------------
	subparsers = parser.add_subparsers(
		help="Alignment file processing",
		dest='parse_bam')

	parse_bam_parser = subparsers.add_parser(
		'parse_bam',
		help='Parse an alignment file',
		description=script_descriptions['parse_bam'],
		prog='parse_bam',
		parents=[common_args])
		
	parse_bam_input = parse_bam_parser.add_argument_group('input')
	parse_bam_input.add_argument(
		"-i",
		"--input",
		help="path to bam file. Note that this must be "+\
			"sorted, and that an index .bai file must "+\
				"exist in the same directory",
		required=True)
	parse_bam_input.add_argument(
		"-b",
		"--barcode_details",
		help="path to the barcode details json",
		required=True)
	parse_bam_input.add_argument(
		"-g",
		"--genome",
		help="path to genome fasta file. "+\
			"Note that a index .fai must exist "+\
				"in the same directory",
		required=True)
	
	parse_bam_output = parse_bam_parser.add_argument_group('output')
	parse_bam_output.add_argument(
		"-o",
		"--output_prefix",
		help="path to output directory. if not provided, "+\
			"output to current working directory",
		default = "",
		required=False)
	
	parse_bam_settings = parse_bam_parser.add_argument_group('settings')
	parse_bam_settings.add_argument(
		"-q",
		"--mapq_threshold",
		help="Reads less than or equal to mapq_threshold "+\
			"will be marked as failed",
		type=int,
		default=10)
	
	# return the top level parser to be used in 'entry point' functions --------
	return parser