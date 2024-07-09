"""
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
"""
import os.path
import pathlib
import sys
import glob
import logging
import json
import numpy as np
import pandas as pd
from functools import partial
from tqdm.auto import tqdm
from rich.traceback import install
from .FeatureBuilder import *
from .Utils import *
# import swifter
# from swifter import set_defaults

class DTI:
    """Read a raw data source and configuration file and convert to Digital Trace Framework format."""
    __version__ = "0.0.30"
    DATA_DIR = "./data/"
    CLR = '\r'
    CLRN = '\r\n'
    
    def __init__(self, input: str, class_label: str, config:dict, fields = None) -> pd.DataFrame:
        self.raw_source_file = self.DATA_DIR+input
        self.src_filename = None
        self.raw_file_list = None
        self.input_type = None
        self.config = config
        self.ignored_fields = []
        self.ignored_fields_cnt = 0
        self.map_cnt = 0
        self.mappings = {}
        self.fill_values = {}
        self.source_df = None
        self.concat_list = list()
        self.message_fields = list()
        self.source_length = 0
        self.input_size = 0
        self.metafunctions = []
        self.dti_output = None
        self.chk_json = False
        self.all_headers = None
        self.config_done = False
        self.dti_fields = None
        self.class_label = class_label
        
        if fields is not None:
            self.dti_fields = fields
            self.dti_length = len(self.dti_fields)
            # splash
            splash()
            print(f'- DTI fields set to {self.dti_fields}')
        else:
            print('Must set dti field values!')
            exit()

        # Swifter settings
        # set_defaults(
        #     npartitions=None,
        #     dask_threshold=1,
        #     scheduler="processes",
        #     progress_bar=True,
        #     progress_bar_desc=None,
        #     allow_dask_on_strings=False,
        #     force_parallel=False,
        # )

        logging.basicConfig(
            format="%(asctime)s %(message)s",
            filename="dti.log",
            level=logging.DEBUG,
            force=True,
        )

        # Configure rich for cleaner debugging messages
        install()

        # Enable tqdm for pandas apply
        tqdm.pandas()

        # basic environments checks
        self._setup()
        
        # compile
        self.compile(config=self.config)
        

    def _setup(self):
        """Create directories, detect file types, and get list of source files.

        Raises:
            FileExistsError: If data directory can not be created.
            TypeError: If source file is not csv or json.
            RuntimeError: If setup failed.
            IOError: If can not get working directory.
            FileExistsError: If source file is not found.
            FileNotFoundError: If raw source files are not found.
        """
        print('\n[START] -> dti')
        _data_dir = pathlib.Path(self.DATA_DIR)
        _src_file = pathlib.Path(self.raw_source_file)

        try:
            if not os.path.exists(_data_dir):
                os.makedirs(_data_dir)
        except IOError as e:
            raise FileExistsError(
                f"Setup failed - can not make directory {self.DATA_DIR}") from e

        try:
            _ftype = file_type_detector(str(_src_file))
            if _ftype != 'unknown':
                self.set_type(_ftype)
                self.set_src_filename(str(_src_file))
            else:
                raise TypeError("File is unknown and not valid json or csv.")

        except RuntimeError as e:
            raise RuntimeError('Setup failed - file type not valid.') from e

        try:
            cwd = os.getcwd()
        except IOError as e:
            raise IOError("Failed to get current working directory") from e

        try:
            if len(glob.glob(str(_src_file))) < 1:
                raise FileExistsError(
                    f"Currently in -> {cwd}, but source file '{str(_src_file)}' not found. \n DTI looks for data starting in ./data/"
                )
            else:
                self.raw_file_list = glob.glob(str(_src_file))
                assert len(
                    self.raw_file_list) > 0, "Raw files not found during setup."
        except IOError as e:
            raise FileNotFoundError(
                f"Setup failed to check source file: {_src_file.resolve()}.") from e

    def chk_config(self, config: dict):
        """Taken in the path to a config file (.tcfg) and compare it with the provided source.

        Args:
            config (dict): Configuration dictionary with required settings to build a DTI structure.

        Raises:
            RuntimeError: If configuration process failed.
            RuntimeError: If failed to set ignored_fields. 
            RuntimeError: If failed checking mappings.
        """
        logging.info("STARTING new dti session.")
        try:
            self.config = config
            if config is not None:
                if check_dti_config(config, self.dti_fields):
                    if 'map' in config:
                        self.set_mappings(config['map'])    
                    if 'concat-map' in config:
                        self.set_concat_list(config['concat-map']['concat_msg'])    
                    if 'fill' in config:
                        self.set_fill_values(config['fill'])    
                    
        except RuntimeError as e:
            logging.exception("Configuration failed, check settings.")
            raise RuntimeError('Configuration failed, check settings.') from e

        # Determin ignore fields
        try:
            # config passed so far, read in source file
            self.get_input_headers(self.raw_file_list)

            # get all column headers from source
            _src_cols_set = set(self.all_headers)

            # create a set of all items in the source file.
            _keep_cols_set = set(self.concat_list)
            for k, val in self.mappings.items():
                _keep_cols_set.add(val)

            # set subtraction
            _src_drop_set = _src_cols_set - _keep_cols_set

            self.ignored_fields = list(_src_drop_set)
            self.ignored_fields_cnt = len(self.ignored_fields)

        except RuntimeError as e:
            raise RuntimeError("failed to set ignored_fields.") from e

        # Log number of mappings for debugging
        try:
            self.map_cnt = (
                len(self.concat_list) + len(self.mappings) +
                len(self.fill_values)
            )
            self.source_length = len(self.all_headers)
            logging.info(
                f'INFO: Concat list length: {str(len(self.concat_list))}\nINFO: mappings length: {str(len(self.mappings))}\nINFO: fill_values length: {str(len(self.fill_values))}')
            logging.info(
                f'INFO: Mapped from config: {str(self.map_cnt)}\nINFO: Fields in source(s): {str(self.source_length)}\nINFO: Ignored in config: {str(self.ignored_fields_cnt)}')
        except RuntimeError as e:
            logging.error(
                "ERROR: Failed to check mappings in source and config files"
            )
            raise RuntimeError(
                "Failed checking mappings, check log files for deatils.") from e
        self.config_done = True

    def compile(self, config:dict):
        """Takes a source file and returns an np array containing all fields from the source.
           Writes dti_formatted csv files.
         
        Raises:
            IOError: If failed creating output, in data directory.
            RuntimeError: If dti is not possible with given config and input files.
            IOError: Failed creating output dti.
            RuntimeError: Failed creating mapping values. 
            RuntimeError: Failed concatenation fields to messages.
            RuntimeError: Failed renaming fields in target dti.
            RuntimeError: Failed dropping ignored fields.
            IOError: Failed reading source file to DataFrame.
        """
        # check the configuration is valid.
        self.chk_config(config=config)
        
        src_path = pathlib.Path(self.raw_source_file)
        output_path = pathlib.Path(
            self.DATA_DIR, f'dti_{src_path.name.replace(".json", ".csv").replace("*","")}')
        self.dti_output = str(output_path)
        try:
            with Timer() as t:
                try:
                    # Create output directory if not already present
                    if not os.path.exists(self.DATA_DIR):
                        os.makedirs(self.DATA_DIR)
                except IOError as e:
                    logging.error(
                        f"ERROR: Failed to create target output directory. \n {self.DATA_DIR}")
                    logging.info(
                        "INFO: Exiting -> Output could not be created. Possibly check permissions of local directory.")
                    raise IOError(
                        f"Failed creating output, in {self.DATA_DIR}, check logs for details.")

                # Get current source file to start
                try:
                    if self.source_df is None:

                        self.read_to_df()
                    _src_raw = self.source_df

                    if not self.dti_possible(_src_raw):
                        logging.error(
                            f"ERROR: Mappings in configuration do not match source. \n Mappings: {str(list(self.mappings.keys()))}")
                        raise RuntimeError(
                            "DTI is not possible with given config and input files.")
                except IOError as e:
                    logging.error(
                        f"ERROR: Failed to read source file to DataFrame. \n {str(src_path)}")
                    logging.info(
                        "INFO: Exiting -> Can not read source file to dataframe, check if source is a valid csv.")
                    raise IOError(
                        "Failed reading source file to DataFrame check logs for details. \n {str(output_path)}") from e
               
                # first drop all ignore fields
                try:
                    if len(self.ignored_fields) == 0:
                        _src_reduced = _src_raw
                    else:
                        _src_reduced = _src_raw.drop(
                            self.ignored_fields, axis=1)
                
                except RuntimeError as e:
                    logging.error("ERROR: Failed dropping ignored fields.")
                    raise RuntimeError(
                        "Failed dropping ignored fields, check logs for details. \n {str(output_path)}") from e
                    
                    
                # next rename all fields that are direct mappings
                try:
                    if 'map' in self.config:
                        # self.conf_path(f'mappings: {self.mappings} \n {_src_reduced}')
                        maps = dict(
                            [(value, key)
                                for key, value in self.mappings.items()]
                        )
                        _src_reduced.rename(columns=maps, inplace=True)
                
                except RuntimeError as e:
                    logging.error("ERROR: Failed renaming fields.")
                    raise RuntimeError(
                        "Failed renaming fields in target DTI. \n {str(output_path)}") from e    

                # concatenate message fields into json object
                try:
                    if 'concat-map' in self.config:
                        # get index of all columns being concatentated
                        idxs = [
                            _src_reduced.columns.get_loc(c)
                            for c in self.concat_list
                            if c in _src_reduced
                        ]
                        #  concat fields to message field
                        # _src_reduced["message"] = _src_reduced.iloc[
                        #     :, idxs
                        # ].swifter.apply(lambda x: x.to_json(), axis=1)
                        print('> Building DTI.', end=self.CLR)
                        _src_reduced["concat_msg"] = _src_reduced.iloc[
                            :, idxs
                        ].apply(lambda x: x.to_json(), axis=1)

                        # drop fields that were concatenated, but not if it is 'message'
                        #if "message" in self.concat_list:
                        #    self.concat_list.remove("message")
                        _src_reduced.drop(self.concat_list, axis=1, inplace=True)
                
                except RuntimeError as e:
                        logging.error(
                            "ERROR: Failed concatenating message fields.")
                        raise RuntimeError(
                            f"Failed concatenation fields to messages, check logs for details.\n {str(output_path)}") from e
                        
                        
                # create value labels if they exist
                try:
                    if 'fill' in self.config:
                        for key, value in self.fill_values.items():
                            zeroes = np.zeros(len(_src_reduced))
                            dframe = pd.DataFrame(
                                {key: value, "count": zeroes}
                            )
                            _src_reduced[key] = dframe[key]
                        
                except RuntimeError as e:
                    logging.error(
                        "ERROR: mapping value labels.")
                    raise RuntimeError(
                        f"Failed creating mapping values, check logs for details. \n {str(output_path)}") from e

                # output dti source file
                try:
                    _src_reduced.sort_values(
                        by=["datetime"], inplace=True)

                    if self.dti_valid(
                        _src_reduced, chk_json=self.chk_json
                    ):
                        print(
                            f"> Writing to csv ({str(self.input_size)}M)", end=self.CLR)
                        _src_reduced.to_csv(
                            output_path, index=False)
                        self.source_df = _src_reduced
                        print(
                            f"- DTI written to: {str(output_path)}", end=self.CLRN
                        )
                    else:
                        print(
                            f"[FAIL] DTI is not valid. \n Columns for output dti: {_src_reduced.columns.values}", end=self.CLRN
                        )
                        exit()

                except IOError as e:
                    logging.error(
                        f"ERROR: Failed to write dti to csv at {str(output_path)}")
                    raise IOError(
                        f"Failed creating output dti, check logs for details. \n {str(output_path)}") from e
            t.interval

        finally:
            print("[DONE] <- dti (%.03f sec.)\n" % t.interval, end=self.CLRN)
            
        return _src_reduced   

    def build_features(self, data, rules, drop=None):
        """Build the compiled feature vector file using the 'dti_' source file. Creates a file in the data directory begining
        with 'fv_dti_' followed by the source file. 

        Args:
            data (str): Path to file under data direcotry. 
            rules (str): Path to directory with EK rules.
            drop (list, optional): List of feature names to be dropped before saving the 'fv_dti_' file. Defaults to None.

        Raises:
            RuntimeError: If call to FeatureBuilder fails to complete.
        """
        try:
            fb=FeatureBuilder(data=data, rules=rules, class_label=self.class_label)
            fb.to_csv(f'fv_{data}', drop)
        except RuntimeError as e:
            raise RuntimeError('failed to complete build_features call.')
    

    def get_input_headers(self, raw_files_list) -> None:
        """Takes pathlib.Path object and references src_filename to find all header indexes from
        all input files as derived from SOURCE_FILE configuration option. Works with kleene star notation
        to process all files identified in the source directory.

        Args:
           raw_files_list: List where raw source and configuration files are located.

        Raises:
            Exception: If source file is not found in pathlib.Path object location.
            Exception: If no files are found using configuration options.
        """
        assert len(raw_files_list) > 0, ' no files found in raw files list.'
        print('> Finding headers.', end=self.CLR)
        if len(raw_files_list) > 0:
            try:
                if self.input_type == "json":
                    cols_set = set()
                    for file in raw_files_list:
                        with open(file, 'r', encoding='utf-8') as _file:
                            for line in _file:
                                for key in json.loads(line):
                                    cols_set.add(key)

                    self.all_headers = list(cols_set)

            except IOError as e:
                logging.error(
                    f"JSON ERROR: Failed to find source file {str(raw_files_list)}")
                logging.info(
                    f"JSON INFO: Exiting -> Check source file {str(raw_files_list)}")
                raise IOError(
                    f"Failed to get headers from source file(s), check logs for details. {str(raw_files_list)}") from e

            try:
                if self.input_type == "csv":
                    cols_set = set()

                    for fn in raw_files_list:
                        with open(fn, 'r', encoding='utf-8') as f:
                            header_list = list(
                                f.readline().strip("\n").split(","))
                            for col in header_list:
                                cols_set.add(col)
                    self.all_headers = list(cols_set)
            except IOError as e:
                logging.error(
                    f"ERROR: Failed to find source files {str(raw_files_list)}")
                logging.info(
                    f"INFO: Exiting -> Check source files {str(raw_files_list)}")
                raise IOError(
                    f"Failed to get headers from source file(s), check logs for details. \n{str(raw_files_list)}") from e

            print(f'- {len(self.all_headers)} Valid headers found:',
                  end=self.CLRN)
            for i in range(0, len(self.all_headers), 8):
                print(self.all_headers[i:i + 8])

    def read_to_df(self) -> None:
        """Reads in source files and returns a concatenated DataFrame containing all fields from all sources.

        Raises:
            FileNotFoundError: If failed to find source file(s) from raw_file_list.
        """
        try:

            if len(self.raw_file_list) > 0:
                if self.source_df is None:
                    print("> Reading sources", end=self.CLR)

                    if self.input_type == "json":
                        self.source_df = pd.concat(
                            map(partial(pd.read_json, lines=True),
                                self.raw_file_list,
                                )
                        )

                    if self.input_type == "csv":
                        self.source_df = pd.concat(
                            map(pd.read_csv, self.raw_file_list)
                        )

                        logging.info(
                            "INFO: Loading current sources to DataFrame.")
                        logging.info(
                            f"Starting DF headers - {str(self.source_df.columns.values)}")

                    self.input_size = round(
                        ((sys.getsizeof(self.source_df) / 1024) / 1000), 2
                    )

                    print(f"- Loaded: {str(self.input_size)}M ", end=self.CLRN)

                    for file in self.raw_file_list:
                        print(f'  o {file}', end=self.CLRN)

        except IOError as e:
            logging.error(
                f"ERROR: Failed to find source file {str(self.raw_file_list)}")
            logging.info(
                f"INFO: Exiting -> Check source file {str(self.raw_file_list)}")

            raise FileNotFoundError(
                f"Failed to find source file, check logs for details. Source: {str(self.raw_file_list)} \n Current working dir: {os.getcwd()}") from e

    def dti_valid(self, dataframe: pd.DataFrame, chk_json) -> bool:
        """Takes a DataFrame and checks if all dti fields are present. This function
        also checks 25% of the dataframe to make sure the message field is parsable JSON.

                Args:
                    dataframe (pandas.DataFrame) DataFrame to be checked.
                    chk_json (bool) if message field should be checked e.g, is it a valid JSON object.

                Returns:
                    True if valid dti format.
        """
        try:
            #print(f'DATAFRAME values {dataframe}')
            assert set(self.dti_fields) == set(dataframe.columns), f'fields of output dataframe are note equal.\n self.dti_fields = {self.dti_fields}\n dataframe.columns = {dataframe.columns}'
            if set(self.dti_fields) == set(dataframe.columns):
                logging.info(
                    f"INFO: dti field check match complete : {str(self.dti_fields)}")
                if chk_json:
                    # same columns so, now test contents of message
                    idxs = np.random.randint(
                        0, dataframe.shape[0], int(
                            0.25 * dataframe.shape[0])
                    )

                    for i in idxs:
                        if not json.loads(dataframe._get_value(i, "message")):
                            raise RuntimeWarning(
                                "Embedded JSON object is not valid.")
                    logging.info(
                        "INFO: -> 25% Random message field check passed as JSON parsable. "
                    )
                return True

        except RuntimeError as e:
            logging.error(
                "ERROR: Failed validity check, make sure all dti field are only mapped one time."
            )
            logging.info("INFO: Exiting -> Check source and config files.")
            raise RuntimeError("dti check failed valid check. ") from e

        return False

    def dti_possible(self, raw_df: pd.DataFrame) -> bool:
        """Compares input from raw source and configuration and returns True if
        columns names and mappings are present in raw source.

        Args:
            raw_df (pd.DataFrame): DataFrame to be checked.

        Raises:
            RuntimeError: Configuration is net yet compiled.

        Returns:
            bool: True if valid dti format.
        """
        try:
            if self.config_done:
                # check for any headers added through metafunctions and add them here too.
                _raw_src_set = set(self.all_headers)
                _raw_src_set.update(set(self.metafunctions))

                if set(list(self.mappings.values())).issubset(_raw_src_set):
                    # all config in single-map are present in raw source df

                    if set(list(self.concat_list)).issubset(_raw_src_set):
                        # all concat fields are in raw source

                        if len(set(list(self.fill_values.keys()))) == 0 or (
                            not set(list(self.fill_values.keys())).issubset(
                                set(list(self.mappings.keys()))
                            )
                        ):
                            # all field-values are not in single-map
                            logging.info(
                                "INFO: dti generation is possible with current settings."
                            )
                            return True
                        else:
                            print(
                                "[field-values] found in [single-map] settings--check config.")
                            logging.error(
                                "ERROR: dti generation is NOT possible with current settings, check configuration parameters. "
                            )
                            logging.info(
                                "INFO: field-values: "
                                + str(list(self.fill_values.keys()))
                            )
                            print(
                                f'INFO: field-values: {str(list(self.fill_values.keys()))}')
                            logging.info(
                                "INFO: single-map: " +
                                str(list(self.mappings.keys()))
                            )
                            print(
                                f'INFO: single-map: : {str(list(self.mappings.keys()))}')
                            return False
                    else:
                        print(
                            "[message-map] fields not found in raw source--check config and source.",
                            style="red",
                        )
                        logging.error(
                            "ERROR: dti generation is NOT possible with current settings, check configuration parameters."
                        )
                        logging.info("INFO: message-map: " +
                                     str(list(self.concat_list)))
                        logging.info("INFO: source fields: " +
                                     str(raw_df.columns))
                        return False

                else:
                    print(
                        "[single-map] fields not found in raw source--check config and source.")
                    logging.error(
                        "ERROR: dti generation is NOT possible with current settings, check configuration parameters. "
                    )
                    logging.info(
                        "INFO: single-map: " +
                        str(list(self.mappings.values()))
                    )
                    logging.info("INFO: source fields: " + str(raw_df.columns))
                    return False

        except RuntimeError as e:
            logging.error(
                "ERROR: Configuration is net yet compiled, run dti.compile(config_dict).")
            logging.info("INFO: Exiting, check configuration file.")
            raise RuntimeError(
                "Configuration is net yet compiled, run dti.compile(config_dict).") from e

    def concat(self, dfs: list, sort_by_field: str = None):
        """Concatenate two dti data sources together. 

        Args:
            dfs (list): A list of pandas.DataFrames with dti data.
            sort_by_field (str, optional): Field name of the column to sort by. Defaults to None.

        Raises:
            KeyError: If value to sort by is not found in source_df.
            RuntimeError: If concatentation failed.
        """
        try:
            # combining fields into embedded JSON object
            assert isinstance(dfs, list), "dfs must be a list."
            assert all(isinstance(df, pd.DataFrame)
                    for df in dfs), "Concat requires a list of pd.DataFrames."

            # all dataframes have same headers as the current dti
            headers = set(list(self.source_df.columns))
            assert all(headers == set(list(df.columns))
                    for df in dfs), "Unable to concat DataFrames with different headers."

            lst = [self.source_df]
            lst.extend(dfs)
            self.source_df = pd.concat(lst)
            if sort_by_field is not None:
                if sort_by_field in list(headers):
                    self.source_df.sort_values(by=[sort_by_field], inplace=True)
                else:
                    raise KeyError(
                        f'Sort failed "{sort_by_field}" not found in fields: \n {str(list(headers))}\n -> source_df was still concatenated, but not sorted.')
        
        except RuntimeError as e:
            raise RuntimeError(f'Failed to concatentate dti sources. Ensure both have the same headers.')
            
    def get_source_df(self):
        """Get method for source_df.

        Returns:
            pandas.DataFrame: The currently loaded source_df. 
        """
        return self.source_df

    def csv_to_dti_source(self, abs_path):
        """Load dti source from a CSV file.

        Args:
            abs_path (str): Absolute path to the dti source file.

        Raises:
            FileExistsError: If file does not exist at absolute path provided.
            FileNotFoundError: If file is not found.
        """
        print(f"- Reading dti source: {abs_path}", end=self.CLRN)
        try:
            file = pathlib.Path(abs_path)

            if os.path.isfile(file):
                self.source_df = pd.read_csv(file)
            else:
                raise FileExistsError("file not found at " + str(file))

        except IOError as e:
            raise FileNotFoundError("unable to load file " + str(file)) from e

        self.input_size = round(
            ((sys.getsizeof(self.source_df) / 1024) / 1000), 2
        )
        print(
            f"- dti loaded from {str(file)}: {str(self.input_size)}M", end=self.CLRN
        )

    def dti_source_to_csv(self, out_path: str = None):
        """This function saves the curreent dti source file to CSV. This function is for external use
        only and is not used during internal operations. 

        Args:
            out_path (str, optional): The path to save the dti output file. Defaults to None.

        Raises:
            FileNotFoundError: If following save to_csv, the saved file could not be verified.
            IOError: If write to csv file fails.
        """
        assert self.source_df is not None, "source_df is not loaded and cannot be saved."
        if out_path is not None:
            assert os.path.isdir(os.path.dirname(out_path)
                                 ), 'target directory does not exist!'
            output_path = pathlib.Path(out_path)
        else:
            src_path = pathlib.Path(self.raw_source_file)
            output_path = pathlib.Path(
                self.DATA_DIR, f'dti_{src_path.name.replace(".json", ".csv").replace("*","")}')

        print(f"> Writing to csv ({str(output_path)})", end=self.CLR)
        try:
            self.source_df.to_csv(str(output_path), index=False)
            if not os.path.isfile(str(output_path)):
                raise FileNotFoundError(
                    f"following save to_csv, could not verify {str(output_path)}")

        except IOError as e:
            raise IOError("unable to write source_df to csv.") from e

        print(f"- dti CSV saved to: {str(output_path)}", end=self.CLRN)

    def set_type(self, type: str):
        """Set input file type (csv or json).

        Args:
            type (str): Either json or csv, as determined by type checking utility function file_type_detector.

        Raises:
            TypeError: If invalid file type - e.g., not csv or json. This is detected using a utility function.
            RuntimeError: If failed to set input_type.
        """
        try:
            if not type in ['csv', 'json']:
                raise TypeError("invalid file type")
            self.input_type = type
        except RuntimeError as e:
            raise RuntimeError('failed to set input_type.') from e

    def set_mappings(self, mappings_dict):
        """Set mappings.

        Args:
            mappings_dict (dict): Dictionary containing mappings settings.

        Raises:
            TypeError: If mappings_dict is not a dict.
            RuntimeError: If failed to set mappings.
        """
        try:
            if not isinstance(mappings_dict, dict):
                raise TypeError("invalid mappings type, must be a dict.")

            self.mappings = mappings_dict

        except RuntimeError as e:
            raise RuntimeError('failed to set mappings.') from e

    def set_concat_list(self, concat_list):
        """Set concat_list.

        Args:
            concat_list (list): List of field names to concatentation into the 'message' field.

        Raises:
            TypeError: If config['concat-map']['message'] is not a list.
            RuntimeError: If failed to set concat_list.
        """
        try:
            if not isinstance(concat_list, list):
                raise TypeError(
                    f"config['concat-map']['message'] must be a list.")

            self.concat_list = concat_list

        except RuntimeError as e:
            raise RuntimeError('Failed to set concat_list.') from e

    def set_fill_values(self, fill_dict):
        """Set fill_values.

        Args:
            fill_dict (dict): Dictionary with fill_value settings.

        Raises:
            TypeError: If fill_dict is not a dict.
            RuntimeError: If failed to set fill_values. 
        """
        try:
            if not isinstance(fill_dict, dict):
                raise TypeError("config['fill'] must be a dict.")

            self.fill_values = fill_dict

        except RuntimeError as e:
            raise RuntimeError('Failed to set fill_values.') from e

    def set_src_filename(self, src_file: str):
        """Set src_file name.

        Args:
            file (str): File name of the dti source file.

        Raises:
            FileNotFoundError: If file name is not found in the ./data directory.
            RuntimeError: If failed to set src_filename
        """
        try:
            if not isinstance(src_file, str):
                raise TypeError(f"source file name must be a string")

            self.src_filename = src_file

        except RuntimeError as e:
            raise RuntimeError('failed to set src_filename.') from e


def check_dti_config(config: dict, fields=None):
    """Check configuration for a dti structure.

    Args:
        config (dict): Dictionary with 'map', 'concat-map' and 'fil-map' settings. 
        dti_fields (list, optional): list of fields to define a new dti structure. Defaults to None.

    Raises:
        ValueError: If all dti fields are not present.
        ValueError: If fields were used more than once in the configuration.
        ValueError: If map and fill have mapped same keys more than once.

    Returns:
        bool: True if dti configuration is valid.
    """
    assert isinstance(fields, list)
    assert all(isinstance(elem, str) for elem in fields)
    req_fields = set(fields)
        
    config_maps = set()
    # check that all dti fields are present
    if 'map' in config:
        config_maps.update(set(dict(nested_dict_iter(config['map'])).keys()))
    if 'concat-map' in config:
        config_maps.update(set(dict(nested_dict_iter(config['concat-map'])).keys()))
    if 'fill' in config:
        config_maps.update(set(dict(nested_dict_iter(config['fill'])).keys()))

    if not req_fields == config_maps:
        missing = req_fields - config_maps
        no_match = config_maps - req_fields
                
        logging.exception(
            f'[ConfigError] Required dti keys were not found: \n Missing keys: {str(list(missing))} \n Mismatched keys: {str(list(no_match))}')
        raise ValueError(
            f'[ConfigError]  Required dti keys were not found: \n Missing keys: {str(list(missing))} \n Mismatched keys: {str(list(no_match))}\n config_maps:{config_maps} \n req_fields:{req_fields}')

    # check if any fields are used twice.
    direct_maps = set()
    concat_maps = set()
    fill_maps = set()
    
    if 'map' in config:
        direct_maps.update(set(flatten_extend(
            dict(nested_dict_iter(config['map'])).values())))
    if 'concat-map' in config:
        concat_maps.update(set(flatten_extend(
        dict(nested_dict_iter(config['concat-map'])).values())))
    if 'fill' in config:    
        fill_maps.update(set(flatten_extend(
            dict(nested_dict_iter(config['fill'])).values())))
    # print(f'len of intersection of all sets: {str(len(direct_maps & concat_maps & fill_maps))}')

    # assert len(direct_maps & concat_maps) + len(direct_maps & fill_maps) + len(concat_maps & fill_maps)==0, "one or more fields are used twice."
    if not len(direct_maps & concat_maps) + len(direct_maps & fill_maps) + len(concat_maps & fill_maps) == 0:
        # find the overlapping values
        elements = set()
        elements.update(direct_maps & concat_maps)
        elements.update(direct_maps & fill_maps)
        elements.update(concat_maps & fill_maps)

        logging.exception(
            f'[ConfigError] The following fields are mapped more than once: \n{str(list(elements))}')
        raise ValueError(
            f'[ConfigError] The following fields are mapped more than once: \n{str(list(elements))}')

    # check if map and fill have mapped same keys more than once
    direct_maps_keys = set()
    fill_maps_keys = set()
    if 'map' in config:
        direct_maps_keys.update(set(
            dict(nested_dict_iter(config['map'])).keys()))
    if 'fill' in config:
        fill_maps_keys.update(set(dict(nested_dict_iter(config['fill'])).keys()))
        
    if not len(direct_maps_keys & fill_maps_keys) == 0:
        elements = set()
        elements.update(direct_maps_keys & fill_maps_keys)
        logging.exception(
            f'[ConfigError] The dti keys are mapped more than once: \n{str(list(elements))}')
        raise ValueError(
            f'[ConfigError] The dti keys are mapped more than once: \n{str(list(elements))}')

    return True
