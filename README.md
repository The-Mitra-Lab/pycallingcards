## Pycallingcards

<p align="center">
    <img src="./docs/pycallingcards_icon.png", width="350">
</p>

Pycallingcards is a python package designed for both bulk and single-cell calling cards data based on [Anndata](https://anndata.readthedocs.io/en/latest/) . 
It includes peak calling, annotation, differential peak analysis, pair analysis with RNA and visualization.

## Documentation.

Please check [**here**](https://pycallingcards.readthedocs.io/en/latest/tutorials.html) for detailed documentation.

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