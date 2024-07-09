# dti-proto
### This prototype package provides a platform to produce Michigan Style Learning Classifier System binary classifiers for use with temporal metadata.  
The EK Rule framework provides a user-friendly interface for extracting interval and characteristic abstractions used to train a learning machine. 

**Requires**
+ numpy >= 1.23.3
+ pandas >= 1.4.4
+ pyarrow >= 1.0.0
+ swifter >= 1.4.0
+ tqdm >= 4.65.0
+ rich >= 13.5.3
+ plyara >= 2.1.1
+ scikit-ExSTraCS >= 1.1.1
+ skrebate >= 0.62
+ scikit-learn >= 1.2.2
+ matplotlib >= 3.6.0
+ streamlit >= 1.36.0
+ imblearn >= 0.0

## How to install
1) Install dti maually (requires setuptools).
```
python3 -m pip install --upgrade pip
pip install setuptools, wheel
```
- Clone dti
```
git clone https://github.com/TheCodeheadMT/2023-dti-proto.git
```
- Open IDE of choice at the root of the 2023-dti-proto folder (developed using vscode)
- All required packages will be installed during this process
- Open terminal/console within IDE and run the following commands:
```
python.exe setup.py bdist_wheel sdist
pip install .
```

2) Use pipy (NOT CURRENTLY IMPLEMENTED)
```
pip install dti 
```
# Note: All data must be under the ./data directory. Datasets are provided in ./data/datasets.zip, unzip before running provided examples.
Sub directories under ./data are accepted as well. e.g., data in ./data/dataset1/data1.csv -> input="dataset1/data1.csv" or input="dataset1/data*.csv" if multiple files are included.
# How to run DTI 
```
from dti import *

if __name__ == "__main__":
    
    # Define field list of all fields included in your data source.
    fields = ["datetime", "timestamp_desc", "source", "source_long", "message", "parser", "display_name", "rel"]
    
    # Map fields from data source(s) to fields in the training data set
    config = {}
    config['map'] = {'datetime': 'datetime',
                    'timestamp_desc':'timestamp_desc',
                    'source_long':'source_long',
                    'display_name':'display_name',
                    'message':'message',
                    'parser':'parser',
                    'source':'source',
                    'rel':'tag'}

    # set rule directory
    rule_dir = './rules/win10_atomics'
    
    # Check config to ensure it is valid and all fields are present
    if check_dti_config(config=config, fields=fields):
        
        ################# CREATE TRAINING DATASET DTS #################
        # Create aggregated dataset using fields and config (creates dti_s1_win10_train_2024-02-19_labeled.csv)
        train_dti = DTI(input='s1_win10_train_2024-02-19_labeled.csv', fields=fields, config=config, class_label='rel')
        
        # Create feature vectors for training using EK rules (creates fv_dti_s1_win10_train_2024-02-19_labeled.csv)
        train_dti.build_features(data='dti_s1_win10_train_2024-02-19_labeled.csv',rules=rule_dir)

        ################# CREATE TEST DATASET DTS #################
        # Same for test data if required (creates dti_s1_win10_test_2024-02-21_labeled.csv)
        test_dti = DTI(input="s1_win10_test_2024-02-21_labeled.csv", fields=fields, config=config, class_label='rel')

        # Create feature vectors for training using EK rules (creates fv_dti_s1_win10_test_2024-02-21_labeled.csv)
        test_dti.build_features(data='dti_s1_win10_test_2024-02-21_labeled.csv',rules=rule_dir)
        
 
        ################# BUILD AND TEST MODELS #################
        # Create a DTILCS using labeled data
        dti = DTILCS(name="s1_100K_model_metrics", class_label="rel", rules=rule_dir)
        
        # Load training data from above
        dti.load_train_data('fv_dti_s1_win10_train_2024-02-19_labeled.csv')
        
        # If testing load test data from above
        dti.load_test_data('fv_dti_s1_win10_test_2024-02-21_labeled.csv')
        
        # Prepare training dataset using oversampling with replacement of the minority class
        dti.prep_dataset(random_state=42)
        
        # Create ExSTraCS object 
        dti.compile(iters=100000, N=6000, nu=15)
        
        # Fit model using loaded training data
        dti.fit()
        
        # Save model to 'models' directory
        dti.save_model()
        
        # Check performance using cross validation
        dti.cross_validate(k=5) #long running
        
        # If testing load test data and specifiy output (reports, plots)
        dti.test_model(
            train_data="fv_dti_s1_win10_train_2024-02-19_labeled.csv",
            class_label='rel',
            predict=True,
            model_metrics=True,
            acc_score=True,
            class_rpt=True,
            roc_prc=True,
            predict_proba=False
        )
          
        # Save false negatives and false posistives in models folder.
        dti.do_analysis()

        
        # PREFORM TESTING ON A MODEL TO CHECK FOR BEST PARAMS:
        dti_test = DTILCSTest()
        # set test parameters and compile
        params = {}
        params['test_name'] = "s1_F1_vs_iters"
        params['training_data_file'] = "fv_dti_s1_win10_train_2024-02-19_labeled.csv"
        params['test_data_file'] = "fv_dti_s1_win10_test_2024-02-21_labeled.csv"
        params['class_label'] = "rel"
        params['rules'] = rule_dir
        params['tests'] = 5
        params['cv_fold'] = 2
        params['iters'] = [40000] 
        params['n_vals'] = [6000]
        params['nu_vals'] = [15]
        params['random_state'] = 42
        dti_test.compile(params=params)

        # start test
        dti_test.start()

    

```
# How to create EK rules.
### See EK rules in ../rules/win10_atomics for examples [here](./rules/win10_atomics).
```
#(interval) edge_running.ekr
rule edge_running : edge_running
{
    strings:
        $FIELD = "message"
        $START = /\[MSEDGE.EXE\] was executed/
        $STOP =  /MSEDGE.EXE.*USN_REASON_CLOSE/
    condition:
        $FIELD interval $START and $STOP
} 


#(characteristic) user_typed_clicked.ekr
rule usr_typed_clicked : usr_typed_clicked
{
	strings:
		$FIELD = "message"
		$VALUE = /User typed.*\]|User clicked.*\]/
	condition:
		$FIELD contains $VALUE
} 



```

## Sample Input
See ./data/ a sample [input](./data/). (must be unzipped)  


## Reference
Under review for publication.  


## License

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

The views expressed in this work are those of the authors, and do not reflect 
the official policy or position of the United States Air Force, Department of 
Defense, or the U.S. Government.