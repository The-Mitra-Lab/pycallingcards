from pycallingcards.raw_processing.BarcodeParser import BarcodeParser

def test_constructor(barcode_details_file):
	bp = BarcodeParser(barcode_details_file)
	assert bp.barcode_dict['tf'] == ''

def test_barcode_breakdown(barcode_details_file,barcodes):

	bp = BarcodeParser(barcode_details_file)
	assert bp.decompose_barcode(barcodes.get('dist0')).get('passing') == True
	
	pb_dist1 = bp.decompose_barcode(barcodes.get('pb_dist1'))
	assert pb_dist1.get('passing') == False
	assert pb_dist1.get('details').get('r1_pb').get('dist') == 1
    
	# note: turns out not actually an error -- this was part of debugging
	# which needs to be removed. It ended up showing that current behavior 
	# is correct
	error_bc = bp.decompose_barcode(barcodes.get('error'))
	assert error_bc.get('details').get('r1_lrt2').get('dist') == 2

def test_annotation_tag_list(barcode_details_file):
	bp = BarcodeParser(barcode_details_file)
	assert bp.annotation_tags == ['ST']
