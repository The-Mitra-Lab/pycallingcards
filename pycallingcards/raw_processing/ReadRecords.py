# pylint:disable=W0622,C0103
# standard library
import os
import logging
import csv
import shutil
# outside dependencies
import pandas as pd

__all__ = ['ReadRecords']

logging.getLogger(__name__).addHandler(logging.NullHandler())


class ReadRecords():
	"""An object to write records from a tagged_read_dict to qbed file and 
	qc files.
	"""
	def __init__(self, qbed_tmpfile, qc_tmpfile):
		"""_summary_

		Args:
			qbed_tmpfile (_type_): path to a tmp file for the qbed. intention 
			is for this to be tmp and then print the grouped/aggregated to 
			user output
			qc_tmpfile (_type_): path to a file for the qc records. intention 
			is for this to be tmp, and then offer methods to summarize and 
			print summary to user output
		"""
		# set qbed fields
		self._qbed_fields = ['chr', 'start', 'end', 'strand', 'annotation']
		# set qc fields
		self._qc_fields = ['query_name', 'mapping_quality', 'read_id',
						   'status', 'component', 'seq', 'edit_dist']
		self._qbed_file = qbed_tmpfile
		self._qbed_filehandle = open(self._qbed_file, 'w+')
		self._qc_file = qc_tmpfile
		self._qc_filehandle = open(self._qc_file, 'w+')	
		# create writers to the tmpfiles
		self._qbed_writer = \
			csv.DictWriter(
				self._qbed_filehandle,
				self._qbed_fields,
				delimiter='\t')

		self._qc_writer = \
			csv.DictWriter(
				self._qc_filehandle,
				self._qc_fields,
				delimiter='\t')
		
		self._query_id = 0
	
	def add_read_info(self, tagged_read: dict, 
	                  status: int, insert_offset=1, annotation_tags: list = [],
					   record_barcode_qc: bool = True) -> None:
		"""write records to both the raw qbed tmpfile and raw qc tmpfile. 
		Note that these tempfiles will be destroyed when the object is destroyed.

		Args:
			tagged_read (dict): _description_
			status (int): A value which reflects how the read performs 
			based on pre-defined quality metrics. A status of 0 is considered 
			a pass. A status of greater than 0 is a read which fails 
			at least 1 quality metric
			insert_offset (int): number to add to tag XI value to calculate 
			the end coordinate. For instance, if the start coord is the first 
			T in TTAA, then the offset would be 4.
			annotation_tags (list): List of strings. Values in list are tags to 
			extract from tagged_read dictionary. Values of tag will be added to 
			the annotation column of the qbed as a string delimited by '/'. 
			record_barcode_qc (bool): Whether to save the QC dict. Right now, 
			can be extremely large. Set to True by default
		"""
		if len({'read', 'barcode_details'}-tagged_read.keys()) > 0:
			KeyError('tagged_read must have keys {"reads","barcode_details"}')

		if status == 0:
			# create the annotation field. If the annotation_tags list is not 
			# empty, this will try to extract the value in the tag from the 
			# tagged_read. KeyError is raised if that tag DNE. Empty string 
			# is created if annotation_tags is empty list
			annotation = "/".join(tagged_read['read'].get_tag(x).split('/')[0]
				for x in annotation_tags)
			qbed_record = {'chr': tagged_read['read'].reference_name,
						   'start': tagged_read['read'].get_tag('XI'),
						   'end': tagged_read['read'].get_tag('XI')+insert_offset,
						   'strand': '+' if tagged_read['read'].is_forward else '-',
						   'annotation': annotation}
			self._qbed_writer.writerow(qbed_record)
		# add barcode qc data
		if record_barcode_qc:
			qc_read_keys = ['query_name', 'mapping_quality']
			qc_read_info = {k: getattr(tagged_read['read'],k) for k in qc_read_keys}
			qc_read_info['read_id'] = self._query_id
			qc_read_info['status'] = status
			for component, comp_dict in tagged_read['barcode_details']['details'].items():
				complete_record = qc_read_info.copy()
				complete_record['component'] = component
				for k, v in comp_dict.items():
					if k == 'query':
						complete_record['seq'] = v
					if k == 'dist':
						complete_record['edit_dist'] = v
				self._qc_writer.writerow(complete_record)
			# increment query_id
		self._query_id = self._query_id+1

	def to_qbed(self, output: str = "") -> pd.DataFrame:
		"""_summary_

		Returns:
			pd.DataFrame: _description_
		"""
		self._qbed_filehandle.close()
		df = pd.read_csv(
			self._qbed_file, 
			sep='\t', 
			names=self._qbed_fields,
			dtype={'chr': str, 'start': int, 
			       'end': int, 'strand': str, 'annotation': str})
		self._qbed_filehandle = open(self._qbed_file,'w+')
		if len(df) == 0:
			logging.critical(
				"No records are written to the qbed -- "
				"cannot make qbed df/file!")
		else:
			df = df\
			.groupby(self._qbed_fields)\
			.size()\
			.reset_index()\
			.rename(columns={0:'depth'})\
				[['chr', 'start', 'end', 'depth', 'strand', 'annotation']]

			if output:
				if os.path.exists(output):
					raise FileExistsError(f'{output} already exists -- cannot overwrite')
				else:
					df.to_csv(output, sep='\t', index=False, header=False)
			
			return df
	
	def summarize_qc(self, output: str = "") -> None:
		self._qc_filehandle.close()
		shutil.move(self._qc_file, output)
		self._qc_filehandle = open(self._qc_file, 'w+')
