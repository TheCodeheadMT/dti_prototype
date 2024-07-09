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

This package is an extention of the scikit-ExSTraCS project located at 
https://github.com/UrbsLab/scikit-ExSTraCS.git.
"""
import math
import os
import pathlib
import numpy as np
from app.dti.proto.Utils import Timer
from app.dti.proto.DTILCS import DTILCS

class DTILCSTest:
    """Digital Trace Inspector Learning Classifier System Test Suite is a custom test parameter testings suite to provide
    an interface for users of DTILCS to build many models and adjust hyperparameters. This suite also creats reports 
    showing results from each trial.
    """
    def __int__(self, iters=[10000], n_vals=[6000], nu_vals=[5])->None:
        self.dti = None
        self.tests = 5
        self.cv_fold = 3
        self.n_vals = n_vals
        self.nu_vals = nu_vals
        self.iters = iters
        self.training_data_file = None
        self.test_data_file = None
        self.cv = []
        self.accuracy = []
        self.bal_accuracy = []
        self.error = []
        self.true_neg = []
        self.false_pos = []
        self.false_neg = []
        self.true_pos = []
        self.total_obs = []
        self.fit_times = []
        self.class_label = None
        self.rules_dir = None

    def compile(
        self,
        training_data_file=None,
        test_data_file=None,
        test_name=None,
        do_ek_scores=False,
        report_dir="./reports/",
        tests=10,
        cv_fold=3,
        iters=[10000],
        n_vals=[6000],
        nu_vals=[5],
        params=None,
    ):
        """Compile test settings before executing.

        Args:
            training_data_file (_type_, optional): Feature vector used to train the model. Defaults to None.
            test_data_file (_type_, optional): Feature vector to test. Defaults to None.
            class_label (str, optional): Class label used in feature vector. Defaults to "relevant".
            test_name (str, optional): Title of the test for saving results. Defaults to "Test".
            do_ek_scores (bool, optional): Fit option, if true use skrebate.ReliefF to calculate feature importance and speedup training. Defaults to False.
            report_dir (str, optional): Output directory for results. Defaults to "./reports/".
            tests (int, optional): Number of tests to run at each setting level. Defaults to 10.
            cv_fold (int, optional): Cross validation fold value. Defaults to 3.
            iters (list, optional): Training iterations (epochs) through training dataset. Defaults to [10000].
            n_vals (list, optional): N-value hyper parameter -- maximum micro classifier population size (sum of classifier numerosities). Defaults to [6000].
            nu_vals (list, optional): Power parameter used to determine the importance of high accuracy when calculating fitness. Defaults to [5].
            params (_type_, optional): Configuration dict to configure model with a batch of settings. Defaults to None.

        Raises:
            ValueError: If number of tests not an int >= 1
            ValueError: If number of cv folds not an int >= 2
            ValueError: If number of iters not positve int >= 1
            ValueError: If number of n_vals not positve int >= 1
            ValueError: If nu_vals not positve int >= 1
            ValueError: If training_data_file not a str
            ValueError: If test_data_file not a str
            ValueError: If test_name value not a str
            ValueError: If test_name not a str
            ValueError: If do_ek_scores not a bool
            ValueError: If report_dir not a str
        """
        self.training_data_file = training_data_file
        self.test_data_file = test_data_file
        self.do_ek_scores = do_ek_scores
        self.report_dir = report_dir
        self.tests = tests
        self.cv_fold = cv_fold
        self.n_vals = n_vals
        self.nu_vals = nu_vals
        self.iters = iters
        self.test_name = test_name
        self.random_state = None


        if params is not None:
            for key, val in params.items():
                match key:
                    case "tests":
                        if type(val) is int and val >= 1:
                            self.tests = val
                        else:
                            raise ValueError("number of tests must be an int >=1")
                    case "cv_fold":
                        if type(val) is int and val >= 2:
                            self.cv_fold = val
                        else:
                            raise ValueError("number of cv folds must be an int >=2")
                    case "iters":
                        if all(elem >= 1 for elem in val):
                            self.iters = val
                        else:
                            raise ValueError(
                                "iters must be positve int >= 1, default is 5000"
                            )
                    case "n_vals":
                        if all(elem >= 1 and isinstance(elem, int) for elem in val):
                            self.n_vals = val
                        else:
                            raise ValueError(
                                "n_vals must be a list of positve int >= 1, default is 6000"
                            )
                    case "nu_vals":
                        if all(elem >= 1 and isinstance(elem, int) for elem in val):
                            self.nu_vals = val
                        else:
                            raise ValueError(
                                "nu_vals must be a list of positve int >= 1, default is 5"
                            )
                    case "training_data_file":
                        if val is not None:
                            if os.path.isfile(pathlib.Path('data',val)):
                                self.training_data_file = val
                            else:
                                raise FileNotFoundError(f"file not found at {str(pathlib.Path('data',val))}")
                            
                        else:
                            raise ValueError("training_data_file must be str and a file in the ./data directory")
                        
                    case "test_data_file":
                        #assert (type(val) is str, "test_data_file value must be a str")
                        if val is not None:
                            if os.path.isfile(pathlib.Path('data',val)):
                                self.test_data_file = val
                            else:
                                raise FileNotFoundError(f"file not found at {str(pathlib.Path('data',val))}")
                        else:
                            raise ValueError("test_data_file must be str and a file in the ./data directory")
                        
                    case "class_label":
                        if val is not None and isinstance(val, str):
                            self.class_label = val
                        else:
                            raise ValueError("class_label must be str")
                        
                    case "test_name":
                        if val is not None and isinstance(val,str):
                            self.test_name = val
                        else:
                            raise ValueError("test_name must be str")
                        
                    case "do_ek_scores":
                        if val is not None and isinstance(val,bool):
                            self.do_ek_scores = val
                        else:
                            raise ValueError("do_ek_scores must be bool")
                        
                    case "random_state":
                        if val is not None and isinstance(val, int) and val >= 0:
                            self.random_state = val
                        else:
                            raise ValueError("number of tests must be an int >=1")
                        
                    case "rules":
                        if val is not None and isinstance(val,str):
                            self.rules_dir = val
                        else:
                            raise ValueError("rules directory must be a valid string")
                    case _:
                        print(f"param: {key} not supported.")

        try:
            
            if self.test_name is not None and self.class_label is not None and self.rules_dir is not None:
                # create DTI object for this test instance.
                self.dti = DTILCS(self.test_name, class_label=self.class_label, rules=self.rules_dir)
            else:
                raise ValueError("missing argument for DTILCS object, be sure name, label, and rules are provided")
        except RuntimeError as e:
            raise RuntimeError("failed to create dti object to start tests") from e    

        
    def start(self):
        """Start trial using current dit, test data, training data and config settings.

        Raises:
            IOError: If report directory can not be created.
            ValueError: Not all test configuration values are set.
            RuntimeError: If model fails to fit.
        """
        try:
            if not os.path.exists(self.report_dir):
                os.makedirs(self.report_dir)
                
        except IOError as e:
            raise IOError("failed to create report directory.") from e

        if any(
            elem is None
            for elem in list([self.dti, self.test_data_file, self.training_data_file])
        ):
            raise ValueError("test configuration values not all set")

        print(f"Running {str(self.tests)} test per setting level.")
        for itr in self.iters:
            for n in self.n_vals:
                for nu in self.nu_vals:
                    self.cv = []
                    self.accuracy = []
                    self.bal_accuracy = []
                    self.error = []
                    self.true_neg = []
                    self.false_pos = []
                    self.false_neg = []
                    self.true_pos = []
                    self.total_obs = []
                    self.fit_times = []
                    self.mccs = []
                    self.f1s = []

                    for i in range(self.tests):
                        try:
                            print(
                                f"\nTrial : {str(i)+'_'+str(itr)+'_'+str(n)+'_'+str(nu)}, with iter:{str(itr)}, N:{str(n)}, nu:{str(nu)}"
                            )
                            self.dti = DTILCS(self.dti.name, class_label=self.class_label)
                            self.dti.load_train_data(self.training_data_file)
                            self.dti.load_test_data(self.test_data_file)
                            #self.x_trn = X_over
                            #self.y_trn = y_over
                            self.dti.prep_dataset(
                                do_ek=self.do_ek_scores,
                                random_state=self.random_state,
                            )
                            self.dti.compile(trial=i, iters=itr, N=n, nu=nu)
                            
                        except RuntimeError as e:
                            raise RuntimeError("Error while setting up training and test data for trail ") from e
                        
                        try:
                            with Timer() as t:
                                try:
                                    self.dti.fit()
                                    self.dti.save_model()
                                    #self.dti.do_analysis()
                                except RuntimeError as e:
                                    raise RuntimeError ("Failed to fit model check configuraiton.") from e
                        finally:
                            self.fit_times.append(round(t.interval, 2))

                        #self.cv.append(self.dti.cross_validate(k=self.cv_fold))
                        #CV turned off!
                        self.cv.append(0)

                        acc, bal_acc, err, cm = self.dti.test_model(
                            train_data=self.training_data_file,
                            class_label=self.class_label,
                        )

                        tn, fp, fn, tp = cm.ravel()
                        self.total_obs.append(np.sum((tn, fp, fn, tp)))
                        self.true_neg.append(tn)
                        self.false_pos.append(fp)
                        self.false_neg.append(fn)
                        self.true_pos.append(tp)
                        self.accuracy.append(acc)
                        self.bal_accuracy.append(bal_acc)
                        self.error.append(err)
                        N = tn + tp + fn + fp
                        S = (tp + fn) / N
                        P = (tp + fp) / N
                        _mcc= ((tp/N) - (S*P)) / math.sqrt(P*S*(1-S)*(1-P))
                        rec = tp / (tp+fp)
                        prc = tp / (tp+fn)
                        _f1 = 2 * prc * rec / (prc + rec)
                        self.mccs.append(_mcc)
                        self.f1s.append(_f1)

                    rpt_name = pathlib.Path(
                        "./reports",
                        f'{self.test_name}_{str(self.tests)}_tests_{"iter-" + str(itr) + "_nu-" + str(n) + "_n-" + str(nu)}.csv',
                    )
                    with open(rpt_name, "w", encoding="utf-8") as report:
                        report.write(f"iter,N,nu,accuracy,bal_accuracy,error,cv{str(self.cv_fold)},true_neg,total_obs,false_pos,false_neg,true_pos,fit_time,mcc,f1\n")
                        for i in range(self.tests):
                            report.write(
                                   f"{str(itr)},\
                                    {str(n)},\
                                    {str(nu)},\
                                    {str(self.accuracy[i])},\
                                    {str(self.bal_accuracy[i])},\
                                    {str(self.error[i])},\
                                    {str(self.cv[i])},\
                                    {str(self.total_obs[i])},\
                                    {str(self.true_neg[i])},\
                                    {str(self.false_pos[i])},\
                                    {str(self.false_neg[i])},\
                                    {str(self.true_pos[i])},\
                                    {str(self.fit_times[i])},\
                                    {str(self.mccs[i])},\
                                    {str(self.f1s[i])}\n")
                                
                    sum_rpt_name = pathlib.Path(
                        "./reports",
                        f'{self.test_name}_{str(self.tests)}_tests_{"iter-" + str(itr) + "_nu-" + str(n) + "_n-" + str(nu)}_summary.csv',
                    )
                    with open(sum_rpt_name, "w", encoding="utf-8") as sum_report:
                        sum_report.write(
                            f"iter,N,nu,accuracy,bal_accuracy,error,cv{str(self.cv_fold)},true_neg,total_obs,false_pos,false_neg,true_pos,fit_time,mcc,f1\n"
                        )
                        sum_report.write(
                            f"{str(itr)},\
                              {str(n)},\
                              {str(nu)},\
                              {str(np.mean(self.accuracy))},\
                              {str(np.mean(self.bal_accuracy))},\
                              {str(np.mean(self.error))},\
                              {str(round(np.mean(self.cv), 4))},\
                              {str(np.mean(self.total_obs))},\
                              {str(np.mean(self.true_neg))},\
                              {str(np.mean(self.false_pos))},\
                              {str(np.mean(self.false_neg))},\
                              {str(np.mean(self.true_pos))},\
                              {str(round(np.mean(self.fit_times), 4))},\
                              {str(round(np.mean(self.mccs), 4))},\
                              {str(round(np.mean(self.f1s), 4))}"
                        )
