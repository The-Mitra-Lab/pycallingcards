# pylint:disable=W1201
"""An object which parses a barcode string extracted from a read according 
to a barcode_details json file. Note that all sequences are cast to upper"""
import json
import logging
import os
from math import inf as infinity
from typing import Literal
import random

# outside package
from edlib import align

__all__ = ['BarcodeParser']

logging.getLogger(__name__).addHandler(logging.NullHandler())


class BarcodeParser:
	"""Using a json which describes acceptable values for given barcodes, 
	check the edit distance between the barcode components (substrings) and 
	the """

	# attributes ---------------------------------------------------------------
	_key_dict = {
		"components": "components",
		"insert_seq": "insert_seq",
		"match_allowance": "match_allowance",
	}

	_barcode_dict = {}
	_barcode_details_json = ""
	_barcode = ""
	_max_r1 = -1
	_restricted_bam_tags = {'XS', 'XI', 'XE', 'XZ'}

	# constructor --------------------------------------------------------------
	def __init__(self, barcode_details_json: str) -> None:
		"""BarcodeParser Constructor

		Args:
			barcode_details_json (str): Path to the barcode details json file
		"""
		# set the initialized barcode details
		# note that this also sets the match allowances
		self.barcode_details_json = barcode_details_json

	# properties ---------------------------------------------------------------
	@property
	def restricted_bam_tags(self):
		"""tags which may not be set in barcode_details"""
		return self._restricted_bam_tags

	@property
	def barcode_details_json(self):
		"""path to the barcode details json file"""
		return self._barcode_details_json

	@barcode_details_json.setter
	def barcode_details_json(self, new_barcode_details_json):
		required_keys = {'r1', 'r2', 'batch'}

		# check that barcode json exists
		if not os.path.exists(new_barcode_details_json):
			raise FileNotFoundError(
				f"Invalid Path: {new_barcode_details_json}")
		# open json, read in as dict
		with open(new_barcode_details_json, 'r') as f1:  # pylint:disable=W1514
			barcode_dict = json.load(f1)
		# check keys
		if not len(required_keys - set(barcode_dict)) == 0:
			raise KeyError(f'the following keys are required: {required_keys}')
		# update the barcode_dict attribute
		logging.info(
			"Updating the barcode dict to reflect the new barcode details json...")
		self.barcode_dict = barcode_dict
		# if that works, update the barcode_details_json path
		logging.info("Success! Setting the barcode details json path")
		self._barcode_details_json = new_barcode_details_json
		# and update some properties
		self.max_r1 = self.__get_max_r1()
		self.max_mismatches = self.__calculate_max_mismatches()

	@property
	def key_dict(self):
		"""A dictionary which describes the defined fields of a valid 
		barcode dict json file"""
		return self._key_dict

	@property
	def barcode_dict(self):
		"""The barcode details json file verified and parsed into a 
		python dictionary"""
		return self._barcode_dict

	@barcode_dict.setter
	def barcode_dict(self, new_barcode_dict):
		# check that the indicies and components match
		# set barcode dict
		error_msg = 'Invalid barcode dict -- check the barcode_details json'
		try:
			self.__check_component_keys(new_barcode_dict)
			self._barcode_dict = new_barcode_dict
		except KeyError as exc:
			raise KeyError(error_msg) from exc
		except ValueError as exc:
			raise ValueError(error_msg) from exc

	@property
	def components(self) -> set:
		"""get the components described in the components field of the barcode 
		dict. Note that this does 'recurse' into the subdictionaries if necessary

		Returns:
			set: A set of components which make up a given read barcode (all 
			separate, eg for the compound TF, this returns the components separately)
		"""
		# extract keys in the components attribute
		component_keys = set()
		for k, v in self.barcode_dict['components'].items():
			if v.get('components', None):
				component_keys.update({x for x in v['components']})
			# else, add the key value
			else:
				component_keys.update({k})
		return component_keys

	@property
	def insert_length(self):
		"""Extract the insertion sequence length from the barcode dictionary"""
		try:
			l = len(self.barcode_dict[self.key_dict['insert_seq']][0])
		except (KeyError, IndexError):
			l = 1
		return l

	@property
	def insert_seq(self):
		"""Getter for the insert seq sequence from the barcode details json. 
		Returns upper case.

		Raises:
			AttributeError: Raised if the current barcode details json does 
			not have an insert seq key
		"""
		if self.key_dict['insert_seq'] in self.barcode_dict:
			return self.barcode_dict[self.key_dict['insert_seq']]
		else:
			raise AttributeError(f'Current barcode details '
								 f'{self.barcode_details_json} does not have an '
								 f'insert seq component')

	@property
	def tagged_components(self):
		return {k: v['bam_tag'] for k, v
				in self.barcode_dict['components'].items()
				if v.get('bam_tag', None)}

	@property
	def max_r1(self):
		if self.max_r1 < 0:
			raise AttributeError('Max R1 not set')
		return self._max_r1

	@max_r1.setter
	def max_r1(self, new_max_r1):
		"""getting for the maximum index on the r1 barcode strand"""
		if new_max_r1 < 0:
			raise ValueError(
				'Max R1 is invalid. Check the barcode details json')
		self._max_r1 = new_max_r1
	
	@property
	def max_mismatches(self):
		"""maximum number of mismatches allowed in a barcode"""
		return self._max_mismatches

	@max_mismatches.setter
	def max_mismatches(self, new_max_mismatches):
		self._max_mismatches = new_max_mismatches
	
	@property
	def annotation_tags(self):
		"""iterate over the component dictionaries and extract the bam tags"""
		annotation_tag_list = []
		for comp, comp_dict in self.barcode_dict['components'].items():
			if comp_dict.get('annotation', None):
				if comp_dict.get('bam_tag', None):
					annotation_tag_list.append(comp_dict.get('bam_tag'))
				else:
					# TODO check this condition in barcodeparser constructor
					raise KeyError(f'Component {comp} has annotation set to true, '
					'but no bam_tag. If annotation is true, the bam_tag field must '
					'be set')
		return annotation_tag_list

	# private methods ----------------------------------------------------------
	def __check_bam_tags(self):
		"""check the tags set by the user in barcode_details.json against 
		the list of restricted bam tags -- error if a restricted bam tag is 
		used"""
		# TODO implement
		raise NotImplementedError

	def __get_max_r1(self):
		max_r1 = -1
		for k, v in self.barcode_dict['r1'].items():
			if v['index'][1] > max_r1:
				max_r1 = v['index'][1]
		if max_r1 == -1:
			raise ValueError('Maximum index on the r1 barcode not found!')

		return max_r1

	def __check_component_keys(self, barcode_dict) -> bool:
		"""_summary_

		Args:
			barcode_dict (_type_): _description_

		Raises:
			KeyError: if keys 'r1', 'r2' and 'components' DNE
			ValueError: _description_

		Returns:
			bool: _description_
		"""
		# extract keys in r1 and r2 and append appropriate read in as prefix
		read_keys = {'r1_' + x for x in barcode_dict['r1'].keys()}\
			.union({'r2_' + x for x in barcode_dict['r2'].keys()})

		# check to make sure that the keys within r1 and r2 have unique names
		if len(read_keys) != len(barcode_dict['r1']) + len(barcode_dict['r2']):
			raise ValueError('keys within r1 and r2 must have unique names')

		# extract keys in the components attribute
		component_keys = set()
		for k, v in barcode_dict['components'].items():

			if not isinstance(v, dict):
				ValueError('Entries in components must be dictionaries')
			# Admittedly, this is confusing. but, a subkey of component may
			# also be components in the event that this given component is
			# compound, eg the yeast TF where the TF bc is made up of r1_primer
			# and r2_transposon. If this is the case, then extract those
			# items listed in the subkey 'components'
			if v.get('components', None):
				component_keys.update({x for x in v['components']})
			# else, add the key value
			else:
				component_keys.update({k})

		if len(read_keys - component_keys) != 0:
			raise KeyError("The components do not fully describe the " +
						   "sequences extracted from the barcode")

		# return true if no errors are raised
		return True
	
	def __calculate_max_mismatches(self) -> int:
		"""either extract or calculate the maximum number of mismatches allowed 
		in a given read's barcode. If not provided, the max is the sum of 
		the match_allowances for each component. These are 0 if not set in 
		barcode zeroes. This allows the user to allow a number of mismatches 
		in each component, but to set a value less that the sum of those 
		allowances in the max_mismatch attribute to avoid a barcode with too 
		many total mismatches.

		Raises:
			TypeError: if the extracted or calculated max_mismatch value is not 
			an integer

		Returns:
			int: an integer describing the total number of allowable mismatches 
			between a barcode and a set of barcode components.
		"""
		max_mismatch = self.barcode_dict.get('max_mismatch', None)
		
		component_mismatch_sum = sum([
				self.barcode_dict['components'][k].get('match_allowance', 0) 
				for k in self.barcode_dict['components'].keys()])
		if max_mismatch:
			if max_mismatch > component_mismatch_sum:
				logging.info('max_mismatch in barcode_details: '
				f'{max_mismatch} which is greater than the sum of '
				f'component mismatch allowances: {component_mismatch_sum}')
		if not max_mismatch:
			max_mismatch = component_mismatch_sum
		if not isinstance(max_mismatch, int):
			raise TypeError('max_mismatch must be an integer. '
				 'check barcode_details.__get_max_mismatches()')
		return max_mismatch

	# public methods -----------------------------------------------------------
	def decompose_barcode(self, barcode: str):
		component_dict = {}
		for end in ['r1', 'r2']:
			for component in self.barcode_dict[end].keys():
				# extract the start/end indicies of a given barcode component.
				# adjust if the indicies are from r2 under the assumption that
				# the sequence is created by appending the r1 seq to the r2 seq
				seq_start = self.barcode_dict[end][component]['index'][0] \
					if end == 'r1' \
					else self.barcode_dict[end][component]['index'][0]+self.max_r1
				seq_end = self.barcode_dict[end][component]['index'][1] \
					if end == 'r1' \
					else self.barcode_dict[end][component]['index'][1]+self.max_r1

				# get the component subsequence out of the barcode
				subseq = barcode[seq_start:seq_end]

				# add the component: best_match_dict key value pair to the
				# barcode_breakdown dict
				component_dict.setdefault("_".join([end, component]), subseq)

		return self.component_check(component_dict)

	def component_check(self, component_dict: dict) -> dict:
		"""Determine if the barcode passes (True) or fails (False) given the 
		edit distances between it and the expected components, and the allowable 
		edit distance between a given component and the actual value.

		Args:
			component_edit_dist_dict (dict): A dictionary where the keys are 
			barcode components and the values are the minimum edit distance 
			of a given barcode against the corresponding allowable components

		Returns:
			dict: A dict of structure {"pass": Boolean, True if the barcode 
			passes, "tf": Str, where the value is either "*" if unknown or a TF 
			string from the barcode_details} 
		"""
		component_check_dict = {}
		for k, v in self.barcode_dict[self.key_dict['components']].items():
			# if the value of a given component is a list, create a dict
			# from from the list where the keys and values are the same.
			target_dict = {x: x for x in v['map']} \
				if isinstance(v.get('map', None), list) \
				else v.get('map', None)
			if not isinstance(target_dict, dict):
				raise ValueError('Each component must have a map entry which ' +
								 'is either a list or a dictionary')
			# if this is a compound component (eg the tf for yeast),
			# construct the sequence from the components. Else, extract the
			# sequence from the input component dict for the given key
			query_seq = \
				"".join([component_dict[x] for x in v.get('components', None)])\
				if v.get('components', None) else component_dict[k]

			# get the match information for the given component
			component_check_dict[k] = \
				self.get_best_match(
				query_seq,
				target_dict,
				v.get('match_type', 'edit_distance'))
			if v.get('bam_tag', None):
				component_check_dict[k]['bam_tag'] = v.get('bam_tag')

		# figure out if the barcode passes based on edit distance allowances
		passing = True
		total_mismatches = 0
		for k, v in component_check_dict.items():
			# if the total_mismatches exceed the maximum number of mismatches 
			# in a given barcode, set the passing value to false and exit the 
			# loop
			if total_mismatches > self.max_mismatches:
				passing = False
				logging.debug('total_mismatches = {total_mismatches}; '
							  + 'max_mismatches = {self.max_mismatches}')
				break
			# else, for a given component, extract the mismatch tolerance
			match_allowance = \
				self.barcode_dict['components'][k]\
					.get('match_allowance', 0)
			# if the edit dist exceeds the match_allowance, set the barcode to
			# failing and break the loop
			if v['dist'] > match_allowance and \
					self.barcode_dict['components']\
						.get(k, {})\
						.get('require', True):
				passing = False
				break
			# if we reach this point, add the edit dist to the total_mismatch
			# note that if this is 0 it won't change anything, and move on to 
			# the next component
			else:
				total_mismatches = total_mismatches+v['dist']

		return {'passing': passing, 'details': component_check_dict}

	@staticmethod
	def get_best_match(query: str, component_dict: dict,
					   match_type: Literal['edit_distance', 'greedy'] = 'edit_distance') -> dict:
		"""Given a match method, return a dictionary describing the 
		best match between query and component_dict

		Args:
			query (str): _description_
			component_dict (dict): _description_
			match_type (Literal[&#39;edit_distance&#39;, 
			&#39;greedy&#39;], optional):_description_. Defaults to 'edit_distance'.

		Returns:
			dict: A dictionary of structure
			{'name': either the sequence match, or the name if map is a 
			named dictionary,
			'dist': edit dist between query and best match -- 
			always 0 or infinty if match is greedy depending on if exact 
			match is found or not. If not, return is 'name': '*', 'dist': inf}
		"""

		# if the match_type is edit_distance, then return a dict with the
		# keys query, with the query sequence, name, with the component 
		# to which the query best matched, and edit_dist, with the edit 
		# distance between query and the best match
		if match_type == 'edit_distance':
			# iterate over the component dict and align strings. dictionary 
			# is one where the key is the edit_distance and the value is a 
			# list of elements which have that edit distance compared to the 
			# query
			d = {}
			{d.setdefault(align(query, k)['editDistance'], []).append(v) \
				for k,v in component_dict.items()}
			# if the minimum edit distance is 0, then the first and only 
			# element in the list os the correct one. Else, select an element 
			# with the minimum element distance at random, if there are more 
			# than 1 possibility with an a given edit distance
			element_selector = 0 if min(d) == 0 \
				else random.randrange(0, len(d.get(min(d))), 1)
			return {'query': query, 'name': d[min(d)][element_selector], 'dist': min(d)}
		# if the match type is greedy, return the first exact match. same
		# return structure as above, where the matched value is the key and
		# the edit distance is 0. If none are found, value is "*" and the
		# edit distance is infinity
		elif match_type == 'greedy':
			for k, v in component_dict.items():
				if k in query:
					return {'name': v, 'dist': 0}
			return {'name': "*", 'dist': infinity}
		else:
			raise IOError(
				'%s is not a recognized match_type argument' % match_type)
