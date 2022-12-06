import logging
from warnings import warn,simplefilter

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ['remove_suffix', 'convert_logger_level', 'deprecated']

def convert_logger_level(level:str)-> int:
    """Convert string logger level, eg 'info' to corresponding int
	Cite: Christian Tremblay https://github.com/ChristianTremblay/BAC0

    Args:
        level (str): one of info, debug, warning, error, critical

    Raises:
        ValueError: if the level is not recognized

    Returns:
        int: the integer corresponding to the input level, eg 'info' returns 20
    """
    if not level:
        return None
    _valid_levels = [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]
    if level in _valid_levels:
        return level
    if level.lower() == "info":
        return logging.INFO
    elif level.lower() == "debug":
        return logging.DEBUG
    elif level.lower() == "warning":
        return logging.WARNING
    elif level.lower() == "error":
        return logging.ERROR
    elif level.lower() == "critical":
        return logging.CRITICAL
    raise ValueError(f"Wrong log level use one of the following : {_valid_levels}")

def remove_suffix(input_string: str, suffix: str) -> str:
	"""a mimic of python 3.9 removesuffix

	Args:
		input_string (str): _description_
		suffix (str): _description_

	Returns:
		str: _description_
	"""
	if suffix and input_string.endswith(suffix):
		return input_string[:-len(suffix)]
	return input_string

def deprecated(message:str):
    """cite: https://stackoverflow.com/a/48632082/9708266 

    Args:
        message (str): _description_
    """
    def deprecated_decorator(func):
        def deprecated_func(*args, **kwargs):
            warn(
                "{} is a deprecated function. {}".format(func.__name__, message),
                category=DeprecationWarning,
                stacklevel=2)
            simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)
        return deprecated_func
    return deprecated_decorator