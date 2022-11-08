"""An object which parses a barcode string extracted from a read according 
to a barcode_details json file. Note that all sequences are cast to upper"""
import os
import json
import logging
from math import inf as infinity
from typing import Literal

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
        "components":"components",
        "insert_seqs": "insert_seqs",
        "match_allowance": "match_allowance",
    }
    
    _barcode_dict = {}
    _barcode_details_json = ""
    _barcode = ""
    
    # constructor --------------------------------------------------------------
    def __init__(self, barcode_details_json: str) -> None:
        """BarcodeParser Constructor

        Args:
            barcode_details_json (str): Path to the barcode details json file
        """
        # set the initialized barcode details
        # note that this also sets the match allowances
        self.barcode_details_json = barcode_details_json
    
    # getters/setters ----------------------------------------------------------
    @property
    def barcode_details_json(self):
        """path to the barcode details json file"""
        return self._barcode_details_json
    @barcode_details_json.setter
    def barcode_details_json(self, new_barcode_details_json):
        required_keys = {'r1', 'r2', 'batch'}
        
        # check that barcode json exists
        if not os.path.exists(new_barcode_details_json):
            raise FileNotFoundError(f"Invalid Path: {new_barcode_details_json}")
        # open json, read in as dict
        with open(new_barcode_details_json, 'r') as f1: #pylint:disable=W1514
            barcode_dict = json.load(f1)
        # check keys
        if not len(required_keys - set(barcode_dict)) == 0:
            raise KeyError(f'the following keys are required: {required_keys}')
        # update the barcode_dict attribute
        logging.info("Updating the barcode dict to reflect the new barcode details json...")
        self.barcode_dict = barcode_dict
        # if that works, update the barcode_details_json path
        logging.info("Success! Setting the barcode details json path")
        self._barcode_details_json = new_barcode_details_json
    
    @property
    def key_dict(self):
        """A dictionary which describes the defined fields of a valid barcode dict json file"""
        return self._key_dict
    
    @property
    def barcode_dict(self):
        """The barcode details json file verified and parsed into a python dictionary"""
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
        for k,v in self.barcode_dict['components'].items():
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
            l = len(self.barcode_dict[self.key_dict['insert_seqs']][0])
        except (KeyError, IndexError):
            l = 1
        return l
    
    @property
    def insert_seqs(self):
        """Getter for the insert seq sequence from the barcode details json. Returns upper case.

        Raises:
            AttributeError: Raised if the current barcode details json does 
            not have an insert seq key
        """
        if self.key_dict['insert_seqs'] in self.barcode_dict:
            return self.barcode_dict[self.key_dict['insert_seqs']]
        else:
            raise AttributeError(f'Current barcode details '\
                f'{self.barcode_details_json} does not have an ' \
                    f'insert seq component')
    @property
    def tagged_components(self):
        return {k:v['bam_tag'] for k,v \
            in self.barcode_dict['components'].items() \
                if v.get('bam_tag', None)}
                
    # private methods ----------------------------------------------------------
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
        for k,v in barcode_dict['components'].items():
            
            if not isinstance(v,dict):
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
            raise KeyError("The components do not fully describe the "+\
                "sequences extracted from the barcode")

        # return true if no errors are raised
        return True

    # public methods -----------------------------------------------------------                    
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
            passes, "tf": Str, where the value is eithe "*" if unknown or a TF 
            string from the barcode_details} 
        """
        component_check_dict = {}
        for k,v in self.barcode_dict[self.key_dict['components']].items(): 
            # if the value of a given component is a list, create a dict 
            # from from the list where the keys and values are the same.    
            target_dict = {x:x for x in v['map']} \
                if isinstance(v.get('map',None),list) \
                    else v.get('map', None)
            if not isinstance(target_dict, dict):
                raise ValueError('Each component must have a map entry which '+\
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
        
        passing = True
        for k,v in component_check_dict.items():
            match_allowance = self.barcode_dict[self.key_dict['match_allowance']].get(k,0)
            if v['dist'] > match_allowance and \
                self.barcode_dict['components'].get(k,{}).get('require', True):
                passing = False
 
        return {'passing': passing, 'details': component_check_dict}
    
    @staticmethod
    def get_best_match(query:str,component_dict:dict,
    match_type:Literal['edit_distance', 'greedy']='edit_distance') -> dict:
        """Given a match method, return a dictionary describing the 
        best match between query and component_dict

        Args:
            query (str): _description_
            component_dict (dict): _description_
            match_type (Literal[&#39;edit_distance&#39;, &#39;greedy&#39;], optional): _description_. Defaults to 'edit_distance'.

        Returns:
            dict: A dictionary of structure
            {'name': either the sequence match, or the name if map is a named dictionary,
            'dist': edit dist between query and best match -- 
            always 0 or infinty if match is greedy depending on if exact 
            match is found or not. If not, return is 'name': '*', 'dist': inf}
        """


        # if the match_type is edit_distance, then return a dict with the 
        # value of the component map as the key, and the edit distance 
        # as the value
        if match_type == 'edit_distance':
            d = {align(query,k)['editDistance']:v for k,v in component_dict.items()}
            return {'name':d[min(d)], 'dist': min(d)}
        # if the match type is greedy, return the first exact match. same 
        # return structure as above, where the matched value is the key and 
        # the edit distance is 0. If none are found, value is "*" and the 
        # edit distance is infinity
        elif match_type == 'greedy': 
            for k,v in component_dict.items():
                if k in query:
                    return {'name':v, 'dist':0}
            return {'name':"*", 'dist': infinity}
