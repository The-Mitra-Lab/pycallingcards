import pathlib
import pytest
import os

import pysam

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
        raise FileNotFoundError(
            "file %s not found in test data" % barcode_details)

    return barcode_details


@pytest.fixture(autouse=True)
def genome_file(tests_dirpath):

    fasta = os.path.join(
        tests_dirpath,
        "test_data/chr1.fa.gz")

    if not os.path.exists(fasta):
        raise FileNotFoundError("file %s not found in test data" % fasta)
    if not os.path.exists(fasta+".fai"):
        raise FileNotFoundError(
            'the fasta file is not indexed -- %s does not exist' % (fasta+'.fai'))

    return fasta


@pytest.fixture(autouse=True)
def bam_file(tests_dirpath):

    bam = os.path.join(
        tests_dirpath,
        "test_data/human_AY53-1_50_T1.bam")

    if not os.path.exists(bam):
        raise FileNotFoundError("file %s not found in test data" % bam)
    if not os.path.exists(bam+".bai"):
        raise FileNotFoundError(
            'the bam file is not indexed -- %s does not exist' % (bam+'.bai'))

    return bam


@pytest.fixture(autouse=True)
def barcodes():
    d = {
        "dist0": "TAGCGTCAATTTTACGCAGACTATCTTTCTAGGGTTAA",
        "pb_dist1": "TGGCGTCAATTTTACGCAGACTATCTTTCTAGGGTTAA",
        'lrt_dist1': "TAGCGTCAANTTTACGCAGACTATCTTTCTAGGGTTAA",
        "error": "TAGCGTCAATTTTACGCAGACTATCTTTCTCTGGTAAT"
    }
    return d


# @pytest.fixture(autouse=True)
# def bam_read1():

    #     Read name = A00118:503:HNGFYDSX3:4:2148:7337:12602_TAGCGTCAATTTTACGCAGACTATCTTTCTCTGGTAAT
    # Read length = 40bp
    # Flags = 16
    # ----------------------
    # Mapping = Primary @ MAPQ 60
    # Reference span = chr1:161,269,957-161,269,996 (-) = 40bp
    # Cigar = 40M
    # Clipping = None
    # ----------------------
    # L1 = CGTCAATTTTACGCAGACTATCTTT/0
    # L2 = GGTTAA/2
    # PB = TAG/0
    # XE = 161270000
    # XI = 161269996
    # NM = 0
    # AS = 40
    # XS = 161269995
    # ST = CTCT/0
    # XZ = ATTA
    # Hidden tags: MDLocation = chr1:161,269,996
    # Base = C @ QV 37

#     a = pysam.AlignedSegment()
#     a.query_name = "A00118:503:HNGFYDSX3:4:2148:7337:12602_TAGCGTCAATTTTACGCAGACTATCTTTCTCTGGTAAT"
#     a.query_sequence = "AGCTTAGCTAGCTACCTATATCTTGGTCTTGGCCG"
#     a.flag = 16
#     a.reference_id = 0
#     a.reference_start = 32
#     a.mapping_quality = 20
#     a.cigar = ((0, 40))
#     a.next_reference_id = 0
#     a.next_reference_start = 199
#     a.template_length = 167
#     a.query_qualities = pysam.qualitystring_to_array(
#         "<<<<<<<<<<<<<<<<<<<<<:<9/,&,22;;<<<")
#     a.tags = (("NM", 1),
#               ("RG", "L1"))
#     return a
