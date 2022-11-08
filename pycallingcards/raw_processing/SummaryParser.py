#pylint:disable=W1203
# standard library
import os
import logging 

# outside library
import pandas as pd

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ['SummaryParser']

class SummaryParser():

    _query_string = "status == 0"

    _summary_columns = {'id':str, 'status':int, 'mapq':int, 'flag':int, 'chr':str, 
                       'strand':str, 'five_prime':str, 'insert_start':str, 
                       'insert_stop':str, 'insert_seq':str, 'depth': int}

    _grouping_fields = {'chr', 'insert_start', 'insert_stop', 'strand'}

    _qbed_col_order = \
        ['chr', 'start', 'end', 'depth', 'strand']
    
    _summary = None

    def __init__(self,summary:str or pd.DataFrame)->None:
        """_summary_

        Args:
            summary (str or pd.DataFrame): _description_
        """
        self.summary = summary
    
        
        # switcher = {
        #     'status': self._status_filter(*args, **kwargs),
        #     'mapq': self._mapq_filter(*args, **kwargs),
        #     'region': self._region_filter(*args, **kwargs),
        #     'default': self.filterError(method)
        # }

        # switcher.get(method, "default")
    
    # def filterError(self, method):
    #     raise NotImplementedError(f"No filter method matches {method}")

    # def _status_filter(self, query_string):
    #     self.filter_string = query_string


    # def _mapq_filter(self):
    #     raise NotImplementedError
    
    # def _region_filter(self):
    #     raise NotImplementedError
	
    @property
    def query_string(self):
        """_summary_"""
        return self._query_string
    @query_string.setter
    def query_string(self, query_string:str):
        self._query_string = query_string

    @property
    def summary(self):
        """_summary_"""
        return self._summary
    @summary.setter
    def summary(self, summary:str or pd.DataFrame):
        # check input
        if isinstance(summary, str):
            # check genome and index paths
            if not os.path.exists(summary):
                raise FileNotFoundError(f"Input file DNE: {summary}")
            summary = pd.read_csv(summary, dtype = self.summary_columns)
        elif isinstance(summary, pd.DataFrame):
            logging.info(f'passed a dataframe to SummaryParser')
        else:
            raise IOError(f'{summary} is not a data type recognized '+\
                'as a summary by SummaryParser')

        if 'depth' not in summary.columns:
            summary['depth'] = 1

        self._verify(summary)

        self._summary = summary
    
    @property
    def summary_columns(self):
        """_summary_"""
        return self._summary_columns
    @summary_columns.setter
    def summary_columns(self, col_list:list):
        self._summary_columns = col_list
    
    @property
    def grouping_fields(self):
        """_summary_"""
        return self._grouping_fields
    @grouping_fields.setter
    def grouping_fields(self, new_grouping_fields:dict):
        self.grouping_fields = new_grouping_fields
    
    @property
    def qbed_col_order(self):
        """_summary_"""
        return self._qbed_col_order
    @qbed_col_order.setter
    def qbed_col_order(self,new_col_order:list):
        self._qbed_col_order = new_col_order

    def _verify(self, summary:pd.DataFrame) -> None:
        """_summary_

        Args:
            summary (pd.DataFrame): _description_

        Raises:
            ValueError: _description_
        """
        if not len(set(self.summary_columns.keys()) - set(summary.columns)) == 0:
            raise ValueError(
                f"The expected summary columns are "\
                    f"{','.join(self.summary_columns)} in that order")

    def to_qbed(self) -> pd.DataFrame:
        """_summary_

        Args:
            annotation (str): _description_

        Raises:
            AttributeError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        local_grouping_fields = self.grouping_fields

        return self.summary\
            .query(self.query_string)\
            [['chr', 'insert_start', 'insert_stop', 'depth', 'strand']]\
            .groupby(list(local_grouping_fields))['depth']\
        .agg(['sum'])\
        .reset_index()\
        .rename(columns={'sum':'depth', 'insert_start':'start', 'insert_stop': 'end'})[self.qbed_col_order]
    
    def write_qbed(self,output_path:str)->None:
        """_summary_

        Args:
            output_path (str): _description_
        """
        if not output_path[-4:] in ['.tsv', 'txt']:
            logging.warning(f"output path {output_path} does not end with tsv or txt")
        self.to_qbed().to_csv(output_path,
                            sep = "\t",
                            header = None,
                            index = False)