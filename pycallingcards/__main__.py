import sys
from typing import Callable
from argparse import Namespace
import logging
from logging.config import dictConfig
import argparse
from importlib.metadata import version
# import functions called by the cmd line tools
from pycallingcards.raw_processing.parse_bam import parse_bam

def parse_args()->Callable[[list],Namespace]:
	"""Create a cmd line argument parser for pycallingcards

	Returns:
		Callable[[list],Namespace]: This function returns the main argparser. If 
		the subparser set_defaults method is set, it can make the correct 
		decision as to which function to call to execute a given tool. See 
		the main method below for usage
	"""
	# create the top-level parser

	script_descriptions = {
		'parse_bam':"Iterate over the reads in an alignment file (bam) and "+\
			"separate reads into passing.bam and failing.bam, a "+\
				"qBed format file of the passing reads, and a qc file which "+\
					"allows finer exploration of the barcode and alignment metrics" # TODO write an better description
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
		help="Available Tools")

	parse_bam_parser = subparsers.add_parser(
		'parse_bam',
		help=script_descriptions['parse_bam'],
		description=script_descriptions['parse_bam'],
		prog='parse_bam',
		parents=[common_args])
	
	# set the function to call when this subparser is used
	parse_bam_parser.set_defaults(func=parse_bam)
		
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
	parse_bam_output.add_argument(
		"-qc",
		"--record_barcode_qc",
		help="set to False to avoid recording barcode qc "+\
			"details while processing the bam. Defaults to True, but this "+\
				"can take an enormous amount of memory. Set to false if there "+\
					"is a memory problem.",
		default = True,
		type=bool,
		required=False)
	
	parse_bam_settings = parse_bam_parser.add_argument_group('settings')
	parse_bam_settings.add_argument(
		"-q",
		"--mapq_threshold",
		help="Reads less than or equal to mapq_threshold "+\
			"will be marked as failed",
		type=int,
		default=10)
	
	# return the top level parser to be used in the main method below ----------
	return parser

def main(args=None)->None:
	"""Entry point to pycallingcards"""

	# parse the cmd line arguments
	arg_parser = parse_args()

	args = arg_parser.parse_args(args)
    
	# this is a default setting -- if it is not set, it means 
	# that nothing was passed on the cmd line. Instead, print the 
	# help message
	try:
		log_level = args.log_level.upper()
	except AttributeError:
		sys.exit(arg_parser.print_help())

	# set the logging details
	log_config = {
		"version": 1,
		"root": {
			"handlers": ["console"],
			"level": f"{log_level}"
		},
		"handlers": {
			"console": {
				"formatter": "std_out",
				"class": "logging.StreamHandler"
			}
		},
		"formatters": {
			"std_out": {
				"format": "%(asctime)s : %(module)s : %(funcName)s : line: %(lineno)d\n"+\
					"\tprocess details : %(process)d, %(processName)s\n"+\
					"\tthread details : %(thread)d, %(threadName)s\n"+\
					"\t%(levelname)s : %(message)s",
				"datefmt": "%Y-%m-%d %H:%M:%S"
			}
		}
	}
	dictConfig(log_config)
    # log the cmd line arguments at the debug level
	logging.debug(sys.argv)
	logging.debug(str(args))
    
	# note that this works b/c the subparser set_defaults function attribute 
	# is set. see https://docs.python.org/3/library/argparse.html#parser-defaults
	# scroll up from that point to see a usage example
	args.func(args)
	
if __name__ == "__main__":
	sys.exit(main())