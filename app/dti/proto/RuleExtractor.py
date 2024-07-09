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

This module relies on the plyara project located at 
https://github.com/plyara/plyara.git.
'''
import plyara
import os
import pathlib
from rich.traceback import install

class RuleExtractor:
    """_summary_

    Raises:
        Exception: 
    """
    CLR = '\x1b[1K\r'
    CLRN = '\x1b[1K\r\n'

    def __init__(self, rules: str = None) -> None:
        self.rule_list = []
        self.parser = None
        self.rules_dir = None
        self.name = "rule_extractor"
        install()
        if rules is not None:
            self.rules_dir = rules
            self.parse_rules(self.rules_dir)

    def parse_rules(self, rules_dir: str) -> None:
        """Load model rules from yara-like formatted file located at rules.

        Args:
            rules (str): path object for the rules files.
        """
        print("[START] -> Rule Extractor")
        print(f"- Loading rules from: {rules_dir}")
        try:
            _rule_file = pathlib.Path(rules_dir)
            if self.rules_dir is None:
                self.rules_dir = rules_dir

            if os.path.isfile(_rule_file):
                if str(_rule_file).lower().endswith('.ekr'):
                    self.parser = plyara.Plyara()
                    with open(_rule_file, 'r') as fh:
                        self.rule_list.append(
                            self.parser.parse_string(fh.read()))
            elif os.path.isdir(_rule_file):
                _rule_file_list = [f for f in os.listdir(
                    _rule_file) if os.path.isfile(os.path.join(_rule_file, f))]
                for yara_rule_file in _rule_file_list:
                    if yara_rule_file.lower().endswith('.ekr'):
                        self.parser = plyara.Plyara()
                        with open(os.path.join(_rule_file, yara_rule_file), 'r') as fh:
                            self.rule_list.append(
                                self.parser.parse_string(fh.read()))
                            print(f"- Loaded {yara_rule_file}")
            print("- Loaded %2d rules." % len(self.rule_list))
            print(f"[DONE] <- Rule Extractor")
                        
        except RuntimeError as e:
            raise RuntimeError(
                f"Rules files not found at {rules_dir}.") from e

    def to_string(self):
        """Prints yara configuration loaded from file to console.
        """
        print(self.rule_list)
