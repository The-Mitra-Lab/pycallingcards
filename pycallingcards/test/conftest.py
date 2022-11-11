import pathlib
import pytest
import os

# note that this works b/c the tests scripts are in a subdir of tests named 
# by the module directory name
@pytest.fixture
def tests_dirpath(request):
    return pathlib.Path(os.path.dirname(os.path.dirname(request.node.fspath)))

@pytest.fixture(autouse=True)
def barcode_details_file(tests_dirpath):

    barcode_details = os.path.join(
        tests_dirpath,
        "test_data/barcode_details.json")

    if not os.path.exists(barcode_details):
        raise FileNotFoundError("file %s not found in test data" %barcode_details) 

    return barcode_details

@pytest.fixture(autouse=True)
def genome_file(tests_dirpath):
    
    fasta = os.path.join(
        tests_dirpath,
        "test_data/chr1.fa.gz")
    
    if not os.path.exists(fasta):
        raise FileNotFoundError("file %s not found in test data" %fasta)
    if not os.path.exists(fasta+".fai"):
        raise  FileNotFoundError('the fasta file is not indexed -- %s does not exist' %(fasta+'.fai'))
 
    return fasta

@pytest.fixture(autouse=True)
def bam_file(tests_dirpath):
    
    bam = os.path.join(
        tests_dirpath,
        "test_data/human_AY53-1_50_T1.bam")
    
    if not os.path.exists(bam):
        raise FileNotFoundError("file %s not found in test data" %bam)
    if not os.path.exists(bam+".bai"):
        raise  FileNotFoundError('the bam file is not indexed -- %s does not exist' %(bam+'.bai'))
 
    return bam

@pytest.fixture(autouse=True)
def barcodes():
    d = {
        "dist0":"TAGCGTCAATTTTACGCAGACTATCTTTCTAGGGTTAA",
        "pb_dist1":"TGGCGTCAATTTTACGCAGACTATCTTTCTAGGGTTAA",
        'lrt_dist1':"TAGCGTCAANTTTACGCAGACTATCTTTCTAGGGTTAA"
    }
    return d 
