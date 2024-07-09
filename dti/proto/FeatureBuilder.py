'''
This software project was created in 2023 by the U.S. Federal government.
See INTENT.md for information about what that means. See CONTRIBUTORS.md and
LICENSE.md for licensing, copyright, and attribution information.

Copyright 2023 U.S. Federal Government (in countries where recognized)
Copyright 2023 Michael Todd and Gilbert Peterson

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

'''
import pathlib
import os
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from .RuleExtractor import *
from .Utils import Timer, pair_start_stop, check_regex
from rich.traceback import install
from shutil import get_terminal_size

# from .RuleExtractor import RuleExtractor


class FeatureBuilder:
    """ This class builds feature vectors using a dti source and rules parsed from yara-like rule files using
    operations including field_contains(), field_is(), and field_compare().
    """
    DATA_DIR = './data/'
    CLR = '\x1b[1K\r'
    CLRN = '\x1b[1K\r\n'

    def __init__(self, data=None, rules=None, class_label=None) -> None:
        self.dti: pd.DataFrame = None
        self.feat_vec: pd.DataFrame = pd.DataFrame()
        self.ops_timer = {}
        self.val_rule_conf = set()
        self.fn = None
        self.rules = rules
        self.class_label = class_label
        # rich install custom console output
        install()
        # self.con: Console = Console(width=200)
        tqdm.pandas()

        
                 
        if data is not None:
            self.fn = self.DATA_DIR+data
            self.read_dti(self.fn)

        re = RuleExtractor(rules=rules)
        if re is not None:
            self.build(re.rule_list)

    def read_dti(self, resource: str, rows=None):
        """Load dti training data from a CSV file.

        Args:
            resource (str):  Path to the dti source file. 
            rows (_type_, optional): Number of rows to load. Defaults to None.

        Raises:
            FileNotFoundError: If resource is not found at location provided.
            RuntimeError: If reading dti file fails.
        """
        self.fn: str = resource
        _data_file: pathlib.Path = pathlib.Path(resource)

        try:
            if os.path.isfile(_data_file):
                print(f"> Loading DTI: {resource}", end=self.CLR)
                self.dti: pd.DataFrame = pd.read_csv(
                    _data_file, dtype=object, nrows=rows)
                print(f"- DTI source loaded from: {resource}", end=self.CLRN)
            else:
                raise FileNotFoundError("FILE NOT FOUND at "+str(_data_file))
        except IOError as e:
            raise RuntimeError("unable to load file "+str(_data_file)) from e

    def field_contains(self, field: str, value: str, type: str, sub_field: str) -> np.ndarray:
        """ Returns a vector the length of self.dti with a 1 if the field contains value.

        Args:
            field (str):  Name of the field being considered during calculation.
            value (str): Value to be checked for within the considered field.
            type (str): Type of value default is text.
            sub_field (str): If a sub_field is specificed int he message field this value is not None.

        Raises:
            RuntimeError: If check on self.dti fails.

               
        Raises:
            KeyError: If key is not found in the embedded JSON object.
            RuntimeWarning: If field contains ':' but is not defined in rules as a subfield. Check rule yara rule files for fields that should not be subfields.
            RuntimeError: If check on self.dti fails.

        Returns:
            np.ndarray: np.ndarry representing a feature vector value for the expert conditions checked.
        """
        try:
            _tmp_dti = pd.DataFrame()
            if sub_field is not None:
                
                _tmp_dti = pd.DataFrame()
                
                try:
                    _tmp_dti[sub_field] = self.dti.apply(
                        lambda x: json.loads(x['message'])[sub_field], axis=1).astype(str)
                except KeyError as e:
                    raise KeyError(f'{sub_field} is not a valid field in the loaded dti source. Check rules configuration for refrences to "message:{sub_field}", as they should not exist.')from e
                field = sub_field
            else:
                if ':' in field:
                    raise RuntimeWarning(
                    "field contains ':' but is not defined in rules as a subfield. Check rule yara rule files for fields that should not be subfields.")
                _tmp_dti[field] = self.dti[field]

            results: np.ndarray = np.zeros(len(_tmp_dti), dtype=np.int8)
            
            #DEBUG
            #print(f'DEBUG: value to check for {value}')
            #print('DEBUG: value to check for reg exp: '+ '\b'+value+'\b')
            #print(f'unescaped string {value.encode("utf-8").decode("unicode_escape")}')            
            #DEBUG
            try:
                # check if leading/trailing forward slash is present
                if value[0] == '/' and value[-1] == '/':
                    _value = value[1:-1]
                else:
                    _value = value
                    
                #print("Regex starting with: "+_value)
                if check_regex(_value):
                    print("  [OK] Pattern: "+_value)
                    results = _tmp_dti[field].astype(str).str.contains(_value, regex=True).values
                    _results = results & 1
                    _results_sum = sum(_results)
                    print(f'  ({_results_sum}) hits')
                    if _results_sum > 0 and _results_sum <=25:
                        print("  [!] WARNING: This feature has very few instances and may cause issues when training.")
                    return results & 1            
            except RuntimeError as e:
                raise RuntimeError(f'Failed to calculate results for a feature vector. Check regular expression for {_value}')from e

        except RuntimeError as e:
            
            raise RuntimeError(
                "failed to check if value is a substring of field.") from e

    def field_is(self, field: str, value: str, type: str, sub_field: str) -> np.ndarray:
        """ Returns a vector the length of self.dti with a 1 if the field is an exact match to value.

        Args:
            field (str): Name of the field being considered during calculation.
            value (str): Value to be checked for within the considered field.
            type (str): Type of value default is text.
            sub_field (str): If a sub_field is specificed int he message field this value is not None.

        Raises:
            RuntimeError: If check on self.dti fails.

        Returns:
            np.ndarray: ndarry representing a feature vector value for the expert conditions checked.
        """
        try:
            _tmp_dti = pd.DataFrame()
            
            #if value is numeric but was not presented as a float, add the decimal 
            #if value.isnumeric():
            #    if '.' not in value:
            #        value = value+".0"
            #        print(f"- Value updated to {value}")      
            
            if sub_field is not None:
                try:
                    _tmp_dti[sub_field] = self.dti.apply(
                        lambda x: json.loads(x['message'])[sub_field], axis=1).astype(str)
                except KeyError as e:
                    raise KeyError(f'{sub_field} is not a valid field in the loaded dti source. Check rules configuration for refrences to "message:{sub_field}", as they should not exist.')from e
                field = sub_field
            else:
                if ':' in field:
                    raise RuntimeWarning(
                    "field contains ':' but is not defined in rules as a subfield. Check rule yara rule files for fields that should not be subfields.")
                _tmp_dti[field] = self.dti[field]
            
            # in case we have nan values - set them all to 0
            _tmp_dti[field].fillna(0, inplace=True)
            #print(self.dti[field])
            results: np.ndarray = np.zeros(len(_tmp_dti), dtype=np.int8)
            results = _tmp_dti[field].astype(str).str.fullmatch(value)

            return results & 1
        except RuntimeError as e:
            raise RuntimeError(
                "failed to check if field is equal to value.") from e

    def field_between(self, field: str, val1: str, val2: str, type: str, sub_field: str):
        """ Returns a vector the length of self.dti with a 1 if the value in the specified field is between val1 and val2.

        Args:
            field (str): Name of the field being considered during calculation.
            val1 (str): Value to be checked for within the considered field.
            val2 (str): Value to be checked for within the considered field.
            type (str): Type of value default is text.
            sub_field (str): If a sub_field is specificed int he message field this value is not None.

        Raises:
            RuntimeError: If check on self.dti fails.

        Returns:
            np.ndarray: ndarry representing a feature vector value for the expert conditions checked.fs
        """
        try:
            _tmp_dti = pd.DataFrame()
            if sub_field is not None:
                try:
                    _tmp_dti[sub_field] = self.dti.apply(
                        lambda x: json.loads(x['message'])[sub_field], axis=1).astype(str)
                except KeyError as e:
                    raise KeyError(f'{sub_field} is not a valid field in the loaded dti source. Check rules configuration for refrences to "message:{sub_field}", as they should not exist.')from e

                field = sub_field
            else:
                if ':' in field:
                    raise RuntimeWarning(
                    "field contains ':' but is not defined in rules as a subfield. Check rule yara rule files for fields that should not be subfields.")
                _tmp_dti[field] = self.dti[field]

            results: np.ndarray = np.zeros(len(_tmp_dti), dtype=np.int8)
            results = _tmp_dti[field].between(val1, val2).values
            return results & 1
        except RuntimeError as e:
            raise RuntimeError(
                "failed to check if field value is between val1 and val2.") from e

    def field_compare(self, field: str, val1: str, type: str, operator: str, sub_field: str):
        """Returns a vector the length of self.dti with a 1 if the value in the specified field is between val1 and val2.

        Args:
            field (str): Name of the field being considered during calculation.
            val1 (str): Value to be checked for within the considered field.
            val2 (str): Value to be checked for within the considered field.
            type (str): Type of value default is text.
            sub_field (str): If a sub_field is specificed int he message field this value is not None.

        Raises:
            RuntimeError: If check on self.dti fails.

        Returns:
            np.ndarray: ndarry representing a feature vector value for the expert conditions checked.
        """
        try:
            _tmp_dti = pd.DataFrame()
            if sub_field is not None:
                try:
                    _tmp_dti[sub_field] = self.dti.apply(
                        lambda x: json.loads(x['message'])[sub_field], axis=1).astype(str)
                except KeyError as e:
                    raise KeyError(f'{sub_field} is not a valid field in the loaded dti source. Check rules configuration for refrences to "message:{sub_field}", as they should not exist.') from e
                field = sub_field
            else:
                if ':' in field:
                    raise RuntimeWarning(
                    "field contains ':' but is not defined in rules as a subfield. Check rule yara rule files for fields that should not be subfields.")
                _tmp_dti[field] = self.dti[field]

            results: np.ndarray = np.zeros(len(_tmp_dti), dtype=np.int8)
            # results = _tmp_dti[field].between(val1, val2).values
            match operator:
                case 'gt':
                    results = _tmp_dti[field].astype(
                        float).gt(float(val1)).values
                case 'lt':
                    results = _tmp_dti[field].astype(
                        float).lt(float(val1)).values
                case 'ge':
                    results = _tmp_dti[field].astype(
                        float).ge(float(val1)).values
                case 'le':
                    results = _tmp_dti[field].astype(
                        float).le(float(val1)).values
                case 'eq':
                    results = _tmp_dti[field].astype(
                        float).eq(float(val1)).values
                case 'ne':
                    results = _tmp_dti[field].astype(
                        float).ne(float(val1)).values
                case _:
                    raise ValueError(
                        f"Compare field operator {operator} not supported")
            return results & 1
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to check if field value is greater than val1.") from e

    def field_interval(self, field:str, start_pats: str, stop_pats:str, feature_name: str):
        
        #start_strings = [x.strip() for x in start_pats.split(',')]      
        #stop_strings = [x.strip() for x in stop_pats.split(',')]     
        #create shallow copy to do some work but not save it in self.dti
        data = self.dti.copy(deep=False)
        #setup results array
        #results = np.zeros(len(data), dtype=np.int8)
        #set default value for new feature
        data[feature_name+"_start"] = 0
        data[feature_name+"_stop"] = 0
        data[feature_name] = 0
        
        #create  start tqdm for progress bar       
        #start_tqdm = tqdm(enumerate(start_strings))
        #print('start strings ---->' + str(start_strings))
        # for p in start_tqdm:
        #     print(f'p: {p[0]}:{p[1]}')
        #     start_tqdm.set_description("- Start ")
        #     #results = _tmp_dti[field].str.contains(value.encode('utf-8').decode('unicode_escape'), regex=True).values
        #     #value.encode('utf-8').decode('unicode_escape'), regex=True
        #     #data[feature_name+p[1]] = [1 if x > 0 else 0 for x in data[field].str.find(p[1], 0)]
            
        #     data[feature_name+p[1]] = [1 if x > 0 else 0 for x in data[field].str.contains(p[1].encode('utf-8').decode('unicode_escape'), regex=True).values]
        #     print(data[feature_name+p[1]].values)
        #     data[feature_name+p[1]].to_csv('test_chrome_start.csv')
            
        #     data[feature_name+"_start"] = data[feature_name+"_start"] + data[feature_name+p[1]]
        #     data.drop(feature_name+p[1], axis=1, inplace=True)
        
        if start_pats[0] == '/' and start_pats[-1] == '/':
            _start_pats = start_pats[1:-1]
        else:
            _start_pats = start_pats
        
        
        if stop_pats[0] == '/' and stop_pats[-1] == '/':
            _stop_pats = stop_pats[1:-1]
        else:
            _stop_pats = stop_pats
       
        #print(f'SELF.DTI! \n{self.dti}')


        #print("Regex starting with: "+_start_pats)
        if check_regex(_start_pats) and check_regex(_stop_pats):
            
            data[feature_name+"_start"] = data[field].astype(str).str.contains(_start_pats, regex=True).values * 1
            print(data)
            data[feature_name+"_stop"] = data[field].astype(str).str.contains(_stop_pats, regex=True).values * 1
            
            print(f"  o Start hits: ({data[feature_name+'_start'].sum()}) {_start_pats}")
            print(f"  o Stop hits: ({data[feature_name+'_stop'].sum()}) {_stop_pats}")

            # Find the indices of rows where start patterns are (1) and stop patterns (1)
            start_indices = data.index[data[feature_name+"_start"] == 1].tolist()
            data.drop(feature_name+"_start", axis=1, inplace=True)
            end_indices = data.index[data[feature_name+"_stop"] == 1].tolist()
            data.drop(feature_name+"_stop", axis=1, inplace=True)
            print(f'start indexes: {start_indices}')
            print(f'stop indexes: {end_indices}')
        
            paired_starts, paired_stops = pair_start_stop(start_indices, end_indices)
            #print(f'paried starts: {paired_starts} len of starts: {len(paired_starts)}')
            #print(f'paired stops: {paired_stops} len of stops: {len(paired_stops)}')
            # zip start and stop indicies
            for start, end in zip(paired_starts, paired_stops):
                data.loc[start:end, feature_name] = 1
            
            return data[feature_name]

    def build(self, rules_list):
        """Builds feature vector using rules from RuleGenerator by testing conditions
        specified for each feature in rule_list.

        Args:
            rules_list (_type_): Rule list parsed from RuleExtractor. 

        Raises:
            RuntimeError: If unable to parse subfield.
            RuntimeError: If failed evaluating ek_list.
            RuntimeError: If unable to parse contains subfield.
            RuntimeError: If unable to parse field_contains ek_con.
            RuntimeError: If unable to parse between subfield.
            RuntimeError: If field between operation fails.
            RuntimeError: If unable to parse subfield for 'gt', 'lt', 'ge', 'le', 'eq', 'ne'.
            RuntimeError: If failed parsing ek conditions from rule list.
        """
        print("\n[START] -> Feature Builder")
        # now test rules and create feature vector with a 1 if all condtions are true and 0 if false.
        for idx, mf in enumerate(rules_list):
            
            ek_list = []
            for ek_con in mf:
                #print("working on "+ str(mf))    
                try:
                    operator = ek_con['condition_terms'][1]
                    if operator is 'interval':
                        print(f'found interval rule!!\n {ek_con}')
                        exit()
                        
                    # process conditions
                    if operator == "is":
                        assert len(
                            ek_con['condition_terms']) == 3, "'is' operator requires two operands, for 3 total terms."

                        fld_dict = next(
                            (item for item in ek_con['strings'] if item['name'] == "$FIELD"), None)
                        assert fld_dict is not None, "$FIELD not set in rule file."
                        # now check if there is a subfield present
                        try:
                            fld_dict['sub_field'] = None
                            if ":" in fld_dict['value']:
                                fld_dict['sub_field'] = fld_dict['value'].split(':')[
                                    1]
                        except RuntimeError as e:
                            raise RuntimeError(
                                "unable to parse subfield") from e

                        # get values from dict
                        val_dict = next(
                            (item for item in ek_con['strings'] if item['name'] == "$VALUE"), None)
                        assert val_dict is not None, "$VALUE not set in rule file."

                        # parse all values from value string (delimiter is ',')
                        val_list = [x.strip() for x in val_dict['value'].split(',')]
                        #print("val list in 'in' check:", val_list)
                        try:
                            print(
                                f"> Evaluating {ek_con['rule_name']}", end=self.CLR)
                            with Timer() as t:
                                if len(val_list) > 1:
                                    # this provides the OR functionality to check for more than one condition witin a field
                                    _tmp_resluts_list = []
                                    for val in val_list:
                                        _tmp_resluts_list.append(self.field_is(
                                        fld_dict['value'], val, val_dict['type'], fld_dict['sub_field']))
                                    
                                    # sum all evaluated results for field contains
                                    _sum_list = np.sum(_tmp_resluts_list, axis = 0)
                                    
                                    # append the final evaluated condition, where anything >0 hits is true.
                                    ek_list.append((_sum_list >= 1).astype(int))     
                                else:
                                    ek_list.append(self.field_is(
                                    fld_dict['value'], val_dict['value'], val_dict['type'], fld_dict['sub_field']))
                                
                                self.dti[ek_con['rule_name']] = ek_list[-1]
                            self.ops_timer[ek_con['rule_name'] + "_is"] = round(t.interval, 2)
                        except RuntimeError as e:
                            raise RuntimeError(
                                "failed evaluting ek_list") from e

                    elif operator == "contains":
                        
                        assert len(
                            ek_con['condition_terms']) == 3, "'contains' operator requires two operands, for 3 total terms."
                        fld_dict = next(
                            (item for item in ek_con['strings'] if item['name'] == "$FIELD"), None)
                        assert fld_dict is not None, "$FIELD not set in rule file."
                        # now check if there is a subfield present
                        try:
                            fld_dict['sub_field'] = None
                            if ":" in fld_dict['value']:
                                fld_dict['sub_field'] = fld_dict['value'].split(':')[
                                    1]
                        except RuntimeError as e:
                            raise RuntimeError(
                                "unable to parse subfield") from e

                        # get dict with values 
                        val_dict = next(
                            (item for item in ek_con['strings'] if item['name'] == "$VALUE"), None)
                        assert val_dict is not None, "$VALUE not set in rule file."

                        # get values from dict
                        val_list = [x.strip() for x in val_dict['value'].split(',')]
                         
                        try:
                            print(
                                f"> Evaluating {ek_con['rule_name']}", end=self.CLR)
                            with Timer() as t:
                                
                                if len(val_list) > 1:
                                    # this provides the OR functionality to check for more than one condition witin a field
                                    tmp_resluts_list = []
                                    for val in val_list:
                                        # this evaluates the condition to see if the field contains the value.
                                        tmp_resluts_list.append(self.field_contains(
                                        fld_dict['value'], val, val_dict['type'], fld_dict['sub_field']))
                                        
                                    # sum all evaluated results for field contains
                                    sum_list = np.sum(tmp_resluts_list, axis = 0)
                                    
                                    # append the final evaluated condition, where anything >0 hits is true.
                                    ek_list.append((sum_list >= 1).astype(int))                              
                                else:
                                    # this evaluates the condition to see if the field contains the value.
                                    ek_list.append(self.field_contains(
                                        fld_dict['value'], val_dict['value'], val_dict['type'], fld_dict['sub_field']))
                                
                                
                                # this adds the evaluated rule results to the ek_list. 
                                self.dti[ek_con['rule_name']] = ek_list[-1]
                            self.ops_timer[ek_con['rule_name'] +
                                           "_contains"] = round(t.interval, 2)
                        except RuntimeError as e:
                            raise RuntimeError(
                                "unable to parse field_contains ek_con") from e

                    elif operator == "between":
                        assert len(
                            ek_con['condition_terms']) == 5, "'contains' operator requires 3 operands, 2 seprated by the 'and' operator, for 5 total terms."

                        fld_dict = next(
                            (item for item in ek_con['strings'] if item['name'] == "$FIELD"), None)
                        assert fld_dict is not None, "$FIELD not set in rule file."

                        # now check if there is a subfield present
                        try:
                            fld_dict['sub_field'] = None
                            if ":" in fld_dict['value']:
                                fld_dict['sub_field'] = fld_dict['value'].split(':')[
                                    1]
                        except RuntimeError as e:
                            raise RuntimeError(
                                "unable to parse subfield") from e

                        start_dict = next(
                            (item for item in ek_con['strings'] if item['name'] == "$START"), None)
                        assert start_dict is not None, "$VALUE not set in rule file."
                        stop_dict = next(
                            (item for item in ek_con['strings'] if item['name'] == "$STOP"), None)
                        assert stop_dict is not None, "$VALUE not set in rule file."

                        try:
                            print(
                                f"> Evaluating {ek_con['rule_name']}", end=self.CLR)
                            # with Timer() as t:
                            ek_list.append(self.field_between(fld_dict['value'], start_dict['value'],
                                                              stop_dict['value'], start_dict['type'],
                                                              fld_dict['sub_field']))
                            self.dti[ek_con['rule_name']] = ek_list[-1]
                        except RuntimeError as e:
                            raise RuntimeError(
                                'field "between" operation failed') from e

                    elif operator in ['gt', 'lt', 'ge', 'le', 'eq', 'ne']:
                        assert len(
                            ek_con['condition_terms']) == 3, "'compare' operators requires two operands, for 3 total terms."

                        fld_dict = next(
                            (item for item in ek_con['strings'] if item['name'] == "$FIELD"), None)
                        assert fld_dict is not None, "$FIELD not set in rule file."
                        # now check if there is a subfield present
                        try:
                            fld_dict['sub_field'] = None
                            if ":" in fld_dict['value']:
                                fld_dict['sub_field'] = fld_dict['value'].split(':')[
                                    1]
                        except RuntimeError as e:
                            raise RuntimeError(
                                "unable to parse subfield") from e

                        val_dict = next(
                            (item for item in ek_con['strings'] if item['name'] == "$VALUE"), None)
                        assert val_dict is not None, "$VALUE not set in rule file."

                        try:
                            with Timer() as t:
                                ek_list.append(self.field_compare(
                                    fld_dict['value'], val_dict['value'], val_dict['type'], operator, fld_dict['sub_field']))
                                self.dti[ek_con['rule_name']] = ek_list[-1]
                        finally:
                            self.ops_timer[ek_con['rule_name'] +
                                           "_"+operator] = round(t.interval, 2)
                    
                    elif operator == "interval":
                        #print(f'interval rule found \n ')
                        
                        assert len(
                            ek_con['condition_terms']) == 5, "'interval' operators requires FIELD, \
                                                              START and STOP operands, for 5 total terms."

                        fld_dict = next(
                            (item for item in ek_con['strings'] if item['name'] == "$FIELD"), None)
                        assert fld_dict is not None, "$FIELD not set in rule file."
                        
                        start_dict = next(
                            (item for item in ek_con['strings'] if item['name'] == "$START"), None)
                        assert start_dict is not None, "$VALUE not set in rule file."
                        
                        stop_dict = next(
                            (item for item in ek_con['strings'] if item['name'] == "$STOP"), None)
                        assert stop_dict is not None, "$VALUE not set in rule file."
                        
                        try:
                            with Timer() as t:
                                ek_list.append(self.field_interval(fld_dict['value'], start_dict['value'], stop_dict['value'], ek_con['rule_name']))
                                self.dti[ek_con['rule_name']] = ek_list[-1]
                        finally:
                            self.ops_timer[ek_con['rule_name'] +
                                           "_"+operator] = round(t.interval, 2)
                    else:
                        raise ValueError(
                            f"conditional term '{ek_con['condition_terms'][1]}' -- not currently supported.")

                except RuntimeError as e:
                    raise RuntimeError(
                        "failed parsing ek conditions from rule list") from e

            print(f"  - Built feature: {ek_con['tags'][0]} ")
            print(u'\u2500' * get_terminal_size()[0])

            # self.con.print(np.array(ek_list))
            self.feat_vec[ek_con['tags'][0]] = np.logical_and.reduce(
                np.array(ek_list)) & 1
            self.val_rule_conf.add(ek_con['tags'][0])
        
        # assign a label to feature vectors
        self.feat_vec[self.class_label] = self.dti[self.class_label]

        # cleanup: if any columns are empty, remove them.
        tmp_headers_before = set(self.feat_vec.columns)
        _empty_features = self.feat_vec.loc[:, (self.feat_vec != 0).any(axis=0)].columns
        tmp_headers_after = set(_empty_features)
        if tmp_headers_before != tmp_headers_after:
            print(f"\n  [!] Empty features: {tmp_headers_before.difference(tmp_headers_after)}\n")
                
        print("[DONE] <- Feature Builder\n", end=self.CLRN)

    def get_feature_vector(self):
        """Get method for FeatureBuilder object feat_vec object. 

        Returns:
            pandas.DataFrame: A DataFrame with feature vector value which conatins 1 if true and 0 if false. 
        """
        return self.feat_vec
    
    def get_dti(self) -> pd.DataFrame:
        """Get method for FeatureBuilder object dti object. 

        Returns:
            pandas.DataFrame: A DataFrame with dti source data. 
        """
        return self.dti
     
    def to_csv(self, file_name=None, drop=None):
        """Helper function wrapper for pandas to_csv funciton.

        Args:
            file_name (string, optional): Filename to save the csv to. Defaults to None.
            drop (list, optional): List of columns to drop from the dataframe before saving to csv. Defaults to None.

        Raises:
            IOError: _description_
        """
        try:
            if drop is not None:
                self.feat_vec.drop(drop, axis=1, inplace=True)
                print(f'[NOTE] Dropped features: {drop} during build.\n')
            self.feat_vec.to_csv(self.DATA_DIR+file_name,  index=False) 
        except IOError as e:
            raise IOError(f"Failed to write to csv: {file_name}.")       
        
       