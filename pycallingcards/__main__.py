import sys
import os
import logging
from logging.config import dictConfig
from .cmd_line_parser import parse_args
from .raw_processing.parse_bam import parse_bam

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
    
	# call the appropriate tool
	if args.parse_bam:
		# Check input paths
		logging.info('checking input...')
		input_path_list = [args.input,
						args.barcode_details,
						args.genome]
		if args.output_prefix:
			input_path_list.append(args.output_prefix)
		for input_path in input_path_list:
			if not os.path.exists(input_path):
				error_msg=f"Input file DNE: {input_path}"
				logging.debug(error_msg)
				raise FileNotFoundError(error_msg)
		parse_bam(args)
	else:
		print(arg_parser.print_help())
	
if __name__ == "__main__":
	sys.exit(main())