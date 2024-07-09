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

NOTE: Must have streamlit installed to run this script.
"""
import sys, pathlib
from dti.proto import *
from streamlit.web import cli as stcli

# Define field list of all fields included in your data source
fields = ["datetime", "timestamp_desc", "source", "source_long", "message", "parser", "display_name", "rel"]

# Map fields from data source(s) to fields in the training data set.
config = {}
config['map'] = {'datetime': 'datetime', 'timestamp_desc':'timestamp_desc', 'source_long':'source_long',
                 'display_name':'display_name','message':'message','parser':'parser','source':'source',
                 'rel':'tag'}

# Model location and class label
MOD_NAME    = "s1_100K_nu_15_5_100000_6000_15_20240416_191136"
MOD_DIR     = pathlib.Path('./models', MOD_NAME).as_posix()
MOD_PATH    = pathlib.Path(MOD_DIR,MOD_NAME+"_model.pkl").as_posix()
RULE_DIR    = pathlib.Path(MOD_DIR, "ek_rules").as_posix()
CLASS       = "rel"

# Training data (in ./data directory)
TRN_DATA = "s1_win10_train_2024-02-19_labeled.csv"
DTI_TRN = "dti_" + TRN_DATA
FV_TRN = "fv_dti_" + TRN_DATA

# Test data (in ./data directory)
TEST_DATA = "s1_win10_test_2024-02-21_labeled.csv"
DTI_UNSEEN = "dti_" + TEST_DATA
FV_UNSEEN = "fv_dti_" + TEST_DATA

# Define the fields for the output summary report previewed in streamlit
SUM_FILEDS = ["datetime", "source_long", "message", "tags"]

#######################################################################################

# Encode training data into feature vectors using the ek rules
train_dti = DTI(input=TRN_DATA, fields=fields, config=config, class_label=CLASS)
train_dti.build_features(data=DTI_TRN, rules=RULE_DIR)

# Encode unseen data into feature vectors using the ek rules
unseen_dti = DTI(input=TEST_DATA, fields=fields, config=config, class_label=CLASS)
unseen_dti.build_features(data=DTI_UNSEEN, rules=RULE_DIR)

# Create DTILCS object
dtilcs = DTILCS(name="s1_predict", rules=RULE_DIR, class_label=CLASS)
results = dtilcs.pred_from_model(MOD_PATH, FV_TRN, FV_UNSEEN, CLASS, MOD_DIR, SUM_FILEDS)

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "open_report.py", str(results)]
    sys.exit(stcli.main())