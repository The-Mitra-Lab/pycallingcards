import pysam

from pycallingcards.raw_processing.AlignmentTagger import AlignmentTagger

def test_constructor(barcode_details_file,genome_file):
	
	assert 2==2
	at = AlignmentTagger(barcode_details_file, genome_file)

	at.barcode_dict['tf'] == ""

def test_tagger(barcode_details_file,genome_file,bam_file):

	bam_in = pysam.AlignmentFile(bam_file)
	
	at = AlignmentTagger(barcode_details_file, genome_file)

	for read in bam_in.fetch():
		tagged_read = at.tag_read(read)
		assert isinstance(tagged_read,dict)





