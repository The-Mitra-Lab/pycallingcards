## Pycallingcards

<p align="center">
    <img src="./docs/pycallingcards_icon.png", width="350">
</p>

This a python package designed for single-cell calling cards data. It also provides efficient peak calling implementation for both bulk and single-cell calling cards data.

## Documentation.

Please check [**here**](https://pycallingcards.readthedocs.io/en/latest/installation.html) for detailed documentation.

## Installation

```shell
pip install "git+https://github.com/The-Mitra-Lab/pycallingcards.git" --upgrade
```

## Command line tools

To see available command line tools, enter:

```shell
pycallingcards --help
```

All of the cmd line tools allow you to set the logging level. To redirect the 
log from the console to a file, redirect the std_err like so:

```shell
pycallingcards parse_bam \
    -i aln.bam \
    -b barcode_details.json \
    -g chr1.fa \
    -o . \
    -l info 2> pycallingcards.log
```

You can also call the cmd line scripts from the module like so:

```shell
python -m pycallingcards --help
```