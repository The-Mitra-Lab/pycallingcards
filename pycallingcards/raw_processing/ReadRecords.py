# pylint:disable=W0622,C0103
# standard library
import os
import logging
# outside dependencies
import pandas as pd

__all__ = ['ReadRecords']

logging.getLogger(__name__).addHandler(logging.NullHandler())

class ReadRecords():
	def __init__(self):
		self._qbed_list = []
		self._qc_list = []
		self._query_id = 0

	@property
	def qbed_list(self):
		return self._qbed_list

	@property
	def qc_list(self):
		return self._qc_list
	
	@property
	def query_id(self):
		return self._query_id
	@query_id.setter
	def query_id(self,new_query_id):
		self._query_id = new_query_id

	def add_read_info(self, tagged_read: dict,status:int) -> None:
		"""_summary_

		Args:
			tagged_read (dict): _description_
			status (int): A value which reflects how the read performs 
			based on pre-defined quality metrics. A status of 0 is considered 
			a pass. A status of greater than 0 is a read which fails 
			at least 1 quality metric
		"""
		if len({'read', 'barcode_details'}-tagged_read.keys()) > 0:
			KeyError('tagged_read must have keys {"reads","barcode_details"}')

		if status == 0:
			qbed_record = {'chr': tagged_read['read'].reference_name,
						   'start': tagged_read['read'].get_tag('XI'),
						   'end': tagged_read['read'].get_tag('XI')+1,
						   'strand': '+' if tagged_read['read'].is_forward else '-'}
			self._qbed_list.append(qbed_record)
		# add barcode qc data
		qc_read_keys = ['query_name', 'mapping_quality']
		qc_read_info = {k: getattr(tagged_read['read'],k) for k in qc_read_keys}
		qc_read_info['read_id'] = self.query_id
		qc_read_info['status'] = status
		for component, comp_dict in tagged_read['barcode_details']['details'].items():
			complete_record = qc_read_info.copy()
			complete_record['component'] = component
			for k, v in comp_dict.items():
				if k == 'name':
					complete_record['seq'] = v
				if k == 'dist':
					complete_record['edit_dist'] = v
			self._qc_list.append(complete_record)
		# increment query_id
		self.query_id = self.query_id+1

	def to_qbed(self,output:str="") -> pd.DataFrame:
		"""_summary_

		Returns:
			pd.DataFrame: _description_
		"""
		
		df = pd.DataFrame(self.qbed_list)\
			.groupby(['chr','start','end','strand'])\
			.size()\
		.reset_index()\
		.rename(columns={0:'depth'})\
			[['chr','start','end','depth','strand']]

		if output:
			if os.path.exists(output):
				raise FileExistsError(f'{output} already exists -- cannot overwrite')
			else:
				df.to_csv(output,sep='\t',index=False,header=False)
		
		return df
	
	def summarize_qc(self, output:str="") -> pd.DataFrame:
		"""_summary_

		Args:
			output (str, optional): _description_. Defaults to "".

		Raises:
			FileExistsError: _description_

		Returns:
			pd.DataFrame: _description_
		"""
		df = pd.DataFrame(self.qc_list)\

		if output:
			if os.path.exists(output):
				raise FileExistsError(f'{output} already exists -- cannot overwrite')
			else:
				df.to_csv(output,sep='\t',index=False,header=False)
		
		return df