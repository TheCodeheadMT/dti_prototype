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

from dti.proto import *

if __name__ == "__main__":
    
    # Define field list of all fields included in your data source.
    fields = ["datetime", "timestamp_desc", "source", "source_long", "message", "parser", "display_name", "rel"]
    
    # Map fields from data source(s) to fields in the training data set.
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
    
    # check config to ensure it is valid and all fields are present.
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
        dti = DTILCS(name="s1_100K", class_label="rel", rules=rule_dir)
        
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
        
        # If model already created, use load model instead of fit
        #dti.load_model("./models/s1_100K_20240709_103602_model.pkl")
                
        # Save model to 'models' directory
        dti.save_model()
        
        # Check performance using cross validation
        dti.cross_validate(k=2) #long running so set to 2
        
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
