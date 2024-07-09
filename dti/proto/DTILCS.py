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
import pathlib
import os
import time
import pickle
import pandas as pd
import numpy as np
from skExSTraCS import ExSTraCS
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

from sklearn.metrics import (
    roc_curve,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics import RocCurveDisplay as rcd
from sklearn.metrics import PrecisionRecallDisplay as pcd
from skrebate import ReliefF
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

# from rich.console import Console
from rich.traceback import install

# from rich import inspect
from .Utils import (
    cumulativeFreq,
    movingAvg,
    plot_confusion_matrix,
    Timer,
    analyze_model,
    copy_ek_rules,
    report
)

class DTILCS:
    __version__="0.0.30"
    DATA_DIR = "./data/"
    MOD_DIR = "./models/"
    FIGS_DIR = None
    WRK_DIR = None
    FIG_W = 6.8
    FIG_H = 6.8
    FIG_DPI = 1000
    CLR = '\x1b[1K\r'
    CLRN = '\x1b[1K\r\n'

    def __init__(self, name: str, class_label=None, rules=None, headless=False) -> None:
        self.name = name
        self.fit_mod_name = None
        self.class_label = class_label
        self.cur_mod = None
        self.fit_mod = None
        self.x_trn = None
        self.x_val = None
        self.y_trn = None
        self.y_val = None
        self.trn_data = None
        self.hdrs = None
        self.trn_fn = None
        self.tst_fn = None
        self.tst_hdrs = None
        self.tst_data = None
        self.EK_scores = None
        self.TEST_DATA_FILE = None
        self.headless = headless
        self.trl_name = None
        self.pred = None
        self.pred_probs = None
        self.yhat = None
        self.pred_fn = None
        self.rules = rules
        self.report_path = None

        # run setup
        self._setup()

        # rich install custom console output
        install()
        # self.con = Console(width=250)

    def _reset(self) -> None:
        # reset state on all object variables
        self.fit_mod_name = None
        self.class_label = None
        self.cur_mod = None
        self.fit_mod = None
        self.x_trn = None
        self.x_val = None
        self.y_trn = None
        self.y_val = None
        self.trn_data = None
        self.hdrs = None
        self.tst_fn = None
        self.tst_hdrs = None
        self.tst_data = None
        self.EK_scores = None
        self.TEST_DATA_FILE = None


    def _setup(self) -> None:
        # Create working directories and check environment
        try:
            if not os.path.exists(self.MOD_DIR):
                os.makedirs(self.MOD_DIR)

            if not os.path.exists(self.DATA_DIR):
                os.makedirs(self.DATA_DIR)

        except OSError as e:
            raise IsADirectoryError(
                "Setup failed. Check permissions in local directory."
            ) from e
        plt.style.use('fast')
        font = {'family' : 'arial',
        'size'   : 16}
        plt.rc('font', **font)

    def load_train_data(self, trn_data_fn: str = None, headers_only=False) -> None:
        """Load DTI training data from a CSV file.

        Args:
            trn_data_fn (str, optional): Path to the DTI source file. Defaults to None.
            headers_only (bool, optional): True if only headers are being loaded, e.g., full data is not required. Defaults to False.

        Raises:
            FileExistsError: If data directory does not exist and it can not be created.
            FileExistsError: If the trn_data in DATA_DIR can not be loaded.
        """
        _data_file = pathlib.Path(self.DATA_DIR, trn_data_fn)
        _data_file_reduced = pathlib.Path(self.DATA_DIR, trn_data_fn, "_reduced.csv")
        try:
            if os.path.isfile(_data_file):
                if headers_only:
                    self.trn_data = pd.read_csv(_data_file, nrows=5)
                else:
                    # print(f"> Load train data:\t{str(_data_file)}")
                    print(f"> Load train data:\t{str(_data_file)}", end="\r")
                    self.trn_data = pd.read_csv(_data_file)
                    self.trn_fn = trn_data_fn

                # remove any uef that might have been appended
                # self.trn_dat.rename(lambda x: x  if any(k in x for k in keys) else x, axis=1)

                print(
                    f"- Loaded train data: \t{str(_data_file)} ({len(self.trn_data)} rows)  "
                )

            else:
                raise FileExistsError(
                    "file not found at " + str(_data_file)
                )

        except OSError as e:
            raise FileExistsError(
                "unable to load file " + str(_data_file)
            ) from e

    def load_test_data(self, tst_data_fn: str = None) -> None:
        """Load DTI test data from a CSV file.

        Args:
            tst_data_fn (str): Filename of test data rule set to load. This file must be in the
            DATA_DIR.

        Raises:
            FileExistsError: If data directory does not exist and it can not be created.
            FileExistsError: If the tst_data_fn in DATA_DIR can not be loaded into a DataFrame.
        """
        _test_file = pathlib.Path(self.DATA_DIR, tst_data_fn)
        try:
            if os.path.isfile(_test_file):
                print(f"> Load test data:\t{str(_test_file)}", end="\r")
                self.tst_data = pd.read_csv(_test_file)
                self.tst_fn = tst_data_fn
                print(
                    f"- Loaded test data: \t{str(_test_file)} ({len(self.tst_data)} rows)")
            else:
                raise FileExistsError(
                    "load_train_data: FILE NOT FOUND at " + str(_test_file)
                )

        except OSError as e:
            raise FileExistsError(
                "load_train_data: Unable to load file " + str(_test_file)
            ) from e

    def prep_dataset(
        self, val_size=0, random_state=None, do_ek=False, reduce_features=True, oversample_minority=True
    ) -> None:
        """Split data into train and validation split.

        Args:
            class_label (str): phenotype/class either 1 or 0.
            val_size (float, optional): float value between 0-1. Defaults to .30.
            random_state (int, optional): int value. Defaults to None.

        Raises:
            KeyError: If self.trn_data is not loaded.
            RuntimeError: If class label not found self.trn_data.
            RuntimeError: If sklearn.model_selection.train_test_split or skrebate.ReliefF functions fail
            to execute correctly.
            ValueError: If self.trn_data not set. Use load_train_data(filename.csv).
        """
        # check if a data source has be set
        if self.trn_data is not None:
            # verify the class label being used is in the training data
            if self.class_label not in self.trn_data:
                raise KeyError("class label not found in dataset -> " + self.class_label)

            try:
                # if any features are completely empty, drop them before proceeding.
                # tmp_headers_before = set(self.trn_data.columns)
                # self.trn_data = self.trn_data.loc[:, (self.trn_data != 0).any(axis=0)]
                # tmp_headers_after = set(self.trn_data.columns)
                # if tmp_headers_before != tmp_headers_after:
                #     print(f"\n  [!] Removed empty features: {tmp_headers_before.difference(tmp_headers_after)}")

                if reduce_features and self.tst_data is not None:
                    print(f'Reducing feature set to match train and test data provided.')
                    # Drop all rows that are not found in train or test datasets.
                    try:
                        cols_in_trn = self.trn_data.loc[:, ~self.trn_data.any()].columns.values
                        cols_in_tst = self.tst_data.loc[:, ~self.trn_data.any()].columns.values
                        drop_cols = set(cols_in_trn) & set(cols_in_tst)
                        self.trn_data.drop(drop_cols, axis=1, inplace=True)
                        self.tst_data.drop(drop_cols, axis=1, inplace=True)
                    except RuntimeError as drop_dataframes_error:
                        raise RuntimeError(
                            "failed to drop empty columsn from training and test data sets before training"
                        ) from drop_dataframes_error

                # get data values for to split
                _data_features = self.trn_data.drop(self.class_label, axis=1)
                # get header names (meta-feature names) while still a df
                self.hdrs = _data_features.columns
                print(f'\n  [:)] Training with [{len(self.hdrs)}] features: {str(self.hdrs)}\n')
                # get np array of training data
                _data_features = _data_features.values
                # get np array of phenotype label (class)
                _data_labels = self.trn_data[self.class_label].values
                # create a balanced dataset using oversampling
                if oversample_minority:
                    oversample = RandomOverSampler(sampling_strategy='minority')
                    # fit and apply the transform
                    Xs, ys = oversample.fit_resample(_data_features, _data_labels)
                else:
                    Xs = _data_features
                    ys = _data_labels

                print("> Preparing data")

                if val_size == 0:
                    self.x_trn = Xs
                    self.y_trn = ys
                    print(f'  o Class disribution: {Counter(self.y_trn)}')
                else:
                    (
                        self.x_trn,
                        self.x_val,
                        self.y_trn,
                        self.y_val,
                    ) = train_test_split(
                        Xs,
                        ys,
                        test_size=val_size,
                        random_state=random_state
                    )
                    print(f'  o Class disribution in training data: {Counter(self.y_trn)}')
                    print(f'  o Class disribution in validation data: {Counter(self.y_val)}')

                print("> Split training data", end="\r")

            except RuntimeError as e:
                raise RuntimeError("failed to split data.") from e

            try:
                if do_ek:
                    with Timer() as t: 
                        # set EK scores for use during fit
                        print("> Find feature importance", end="\r")
                        relieff = ReliefF()
                        relieff.fit(self.x_trn, self.y_trn)
                        self.EK_scores = relieff.feature_importances_

                    print(f"[DONE] Found feature importance ({'%.03f'% t.interval} sec.)")    
                else:
                    self.EK_scores = None

            except RuntimeError as e:
                raise RuntimeError(
                    "failed to create EK_scores to identify feature importance."
                ) from e

        else:
            raise ValueError(
                "self.trn_data not set, use load_train_data(filename.csv)."
            )

    def compile(
        self, trial=None, iters=10000, N=6000, nu=5, random_state=None
    ) -> ExSTraCS:
        """Initialize the ExSTraCS LCS and sets hyper parameters.

        Args:
            iters (int, optional): The number of training cycles to run. Defaults to 10000.
            N (int, optional):  Maximum micro classifier population size (sum of classifier numerosities). Defaults to 6000.
            nu (int, optional): Power parameter used to determine the importance of high accuracy when calculating fitness. Defaults to 5.

        Returns:
            ExSTraCS: ExSTraCS 2.0 object (Extended Supervised Tracking and Classifying System)
        """
        try:

            if trial is not None:
                self.trl_name = (
                    str(trial) + "_" + str(iters) + "_" + str(N) + "_" + str(nu)
                )
                if random_state is not None:
                    self.trl_name = self.trl_name + "_seed-" + str(random_state)

            # Initialize ExSTraCS Model
            if self.EK_scores is not None:
                self.cur_mod = ExSTraCS(
                    learning_iterations=iters,
                    N=N,
                    nu=nu,
                    track_accuracy_while_fit=True,
                    expert_knowledge=self.EK_scores,
                )
            else:
                self.cur_mod = ExSTraCS(
                    learning_iterations=iters, N=N, nu=nu, track_accuracy_while_fit=True
                )
        except RuntimeError as e:
            raise RuntimeError("compile: Failed to create ExSTraCS model.") from e

    def fit(self, exp_trking_data: bool = True) -> None:
        """Fit the current model using paramters set by compile.

        Args:
            exp_trking_data (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: If self.cur_mod not defined.
            RuntimeError: If model is not yet compiled.
        """
        if self.cur_mod is not None:
            if self.x_trn is not None and self.y_trn is not None:
                print("> Fit model          ", end="\r")
                # Fit the model using training data and labels

                with Timer() as t:
                    self.fit_mod = self.cur_mod.fit(self.x_trn, self.y_trn)
                _build_tm = "%.03f sec." % t.interval

                timehack = "_" + time.strftime("%Y%m%d_%H%M%S")

                try:
                    # create work dir to save files and figures in
                    if self.trl_name is not None:
                        self.fit_mod_name = self.name + "_" + self.trl_name + timehack
                    else:
                        self.fit_mod_name = self.name + timehack

                    # create model named directory
                    _mod_dir = pathlib.Path(self.MOD_DIR, self.fit_mod_name)
                    os.makedirs(_mod_dir)

                except IOError as e:
                    raise IOError(f"failed creating {str(_mod_dir)}") from e

                # set working directory
                self._set_wrk_dir(_mod_dir, dir=True)

                # save the iteration tracking to disk
                if exp_trking_data:
                    self.fit_mod.export_iteration_tracking_data(
                        str(
                            pathlib.Path(
                                self.WRK_DIR, self.fit_mod_name + "_iterData.csv"
                            )
                        )
                    )

                # export final rule population
                self.fit_mod.export_final_rule_population(
                    self.hdrs,
                    self.class_label,
                    filename=str(
                        pathlib.Path(
                            self.WRK_DIR, self.fit_mod_name + "_finalRulePop.csv"
                        )
                    ),
                    DCAL=False,
                )
                print(f"- Model fit in {_build_tm}")

            else:
                raise ValueError("x_trn and y_trn not set: run load_test_data first.")
        else:
            raise RuntimeError("cur_mod not defined: compile model before fitting.")

    def cross_validate(self, k=5) -> float:
        """Cross validate the current model using sklearn cross_val_score.

        Args:
            k (int, optional): Cross validation folds. Defaults to 5.

        Raises:
            RuntimeError: if cross validate fails
        """
        print(f"> {str(k)}-fold Cross Validation", end='\r')
        try:
            # shuffling data before cross validation
            _formatted = np.insert(self.x_trn, self.x_trn.shape[1], self.y_trn, 1)
            np.random.shuffle(_formatted)
            _data_features = np.delete(_formatted, -1, axis=1)
            _data_labels = _formatted[:, -1]

            # Defaults to 5 fold CV
            try:
                with Timer() as t:
                    try:
                        cv_scores = cross_val_score(estimator=self.cur_mod, X=_data_features, y=_data_labels, cv=k)
                        # cv_scores = cross_val_score(self.fit_mod, _data_features, _data_labels, cv=k)
                        print(f'    o CV Scores: {cv_scores}')
                        cv_score = round(np.mean(cv_scores),2,)         
                    except RuntimeError as e:
                        raise RuntimeError("Cross validation failed.") from e    
            finally:     
                print(f"- {str(k)}-fold Cross Validation score:\t{str(cv_score)} ({'%.03f'% t.interval} sec.)")

        except RuntimeError as e:
            raise RuntimeError(
                "cross validate failed - check x and y data shapes."
            ) from e

        return cv_score

    def save_model(self) -> None:
        """Save a model to disk using pickle file.

        Raises:
            IOError: If self.fit_mod does not exist.
            RuntimeError: If picket.dump fails to write the model to disk.
        """
        if self.fit_mod is not None:
            try:
                assert (
                    self.WRK_DIR is not None
                ), "can not save without work directory being established."
                outfile = pathlib.Path(self.WRK_DIR, self.fit_mod_name + "_model.pkl")
                pickle.dump(self.fit_mod, open(outfile, "wb"))
                print(f"- Saved model -> {str(outfile)}")

            except IOError as e:
                raise IOError(
                    f"pickel.dump: Failed to call picket.dump to {str(outfile)}"
                ) from e
        else:
            raise RuntimeError(
                "save_model: fit_mod not defined--compile and fit model before savinging."
            )

    def load_model(self, model_path: str = None) -> None:
        """Loads existing model from a pkl file and sets the fit_mod and fit_mod_name values.

        Args:
            model_name (str): file name of the model, normally in the format 'mod1_20231010_150556_model.pkl'.

        Raises:
            FileNotFoundError: If model file does not exist.
            IOError: if pickel.load fails to load model.

        """
        try:
            # print(f'provided to load model: {model_path}')
            _mod_path = pathlib.Path(model_path)
            if os.path.isfile(_mod_path):
                # load model from pkl file
                print("> Load model", end="\r")
                self.fit_mod = pickle.load(open(_mod_path, "rb"))

                # set working directory to loaded models
                self._set_wrk_dir(_mod_path)

                print(f"- Loaded model:\t\t{str(_mod_path)}   ")
            else:
                print(f"[FAIL] Load model\t\t{str(_mod_path)}")
                raise FileNotFoundError(f"model file not found: {str(_mod_path)}")

        except IOError as e:
            raise IOError(
                f"pickel.load failed to load model model_name {str(_mod_path)}"
            ) from e

    def predict(self, unseen_data: pd.DataFrame, class_label=None) -> np.ndarray:
        """Given a pandas DataFrame of observations, predict the phenotype (class) as relevant or not.

        Args:
            unseen_data (pd.DataFrame): _description_
            class_label (_type_, optional): _description_. Defaults to None.

        Raises:
            RuntimeError: If model.predict fails.

        Returns:
            np.ndarray: array of int values representing phenotype label (0 not relevant, 1 relevant)
        """
        try:
            if class_label is not None:
                # data was provided a label
                xs = unseen_data.drop(class_label, axis=1).values
            else:
                xs = unseen_data.values

            # get predcition for each unseen observation
            self.pred = self.fit_mod.predict(xs)

            return self.pred

        except RuntimeError as e:
            raise RuntimeError(
                "predict failed, if unseen data is labled, specific class_label='class'"
            ) from e

    def test_model(
        self,
        predict=True,
        acc_score=True,
        class_rpt=True,
        model_metrics=False,
        roc_prc=False,
        predict_proba=False,
        threshold=0.5,
        train_data=None,
        class_label=None,
    ):
        """Test a fit or loaded model and produce various reports and plots. By default, only predict, accuracy score,
        and classification report are produced if no options are specified.

        Args:
            predict (bool, optional): Predict using fit model x and y data. Defaults to True.
            acc_score (bool, optional): Calculate accuracy score using fit model x and y data. Defaults to True.
            class_rpt (bool, optional): Create a classification report using fit model x and y data.. Defaults to True.
            model_metrics (bool, optional): Output plots of model metrics. Defaults to False.
            roc_prc (bool, optional): Create ROC/PRC plots using fit model x and y data. Defaults to False.
            predict_proba (bool, optional): Predict probabilities using fit model x and y data. Defaults to False.
            threshold (float, optional): Set threshold. Defaults to 0.5.
            train_data (_type_, optional): x-data to be tested with fit or loaded model. Defaults to None.
            class_label (_type_, optional): Class label as string, the column name. Defaults to None.

        Raises:
            ValueError: If class label not found in data training and test datasets.
            ValueError: If test_model test data hdrs do not match training hdrs.
            RuntimeError: If failed to drop class label.   
            RuntimeError: If model predict failed.
            RuntimeError: If predict test failed.
            IOError: If failed to write test data to csv.
            RuntimeError: If model predict proba failed.
            RuntimeError: If failed to create roc/prc curves.
            RuntimeError: If failed to plot model metrics.
            RuntimeError: If failed to plot model stats.
            RuntimeError: If failed to create classification report.
            IOError: If failed to write classification report.
            IOError: If failed to write confusion matrix.
            IOError: If failed to write confusion matrix (normlalized).
            IOError: If failed to write ROC display from pred.
            IOError: If failed to write PRC threshold. 
            RuntimeError: If failed to create plots.
            RuntimeError: If test data not loaded.

        Returns:
             np.ndarray: Array of predicted values, e.g., y-hat.
        """
        assert (
            self.WRK_DIR is not None
            and self.fit_mod_name is not None
            and self.FIGS_DIR is not None
        ), "directories are not setup!"

        acc, bal_acc, err, model_cm = None, None, None, None

        if self.tst_data is not None:
            if self.fit_mod is not None:
                if self.trn_data is None:
                    assert (
                        train_data is not None
                    ), "Training data not loaded, must provide a path to training data."
                    self.load_train_data(train_data, headers_only=True)

                assert (
                    self.trn_data is not None and self.tst_data is not None
                ), "training or test data not loaded."
                self.hdrs = self.trn_data.columns.tolist()
                _tst_hdrs = self.tst_data.columns.tolist()

                if class_label in self.hdrs and class_label in _tst_hdrs:
                    self.class_label = class_label
                else:
                    raise ValueError(
                        f"Class label {class_label} not found in data training and test datasets."
                    )

                try:
                    # get header names (meta-feature names) while still a df
                    self.tst_hdrs = self.tst_data.columns.tolist()
                    # get test data values
                    if set(self.hdrs) != set(self.tst_hdrs):
                        raise ValueError(
                            "test_model: test data hdrs do not match training hdrs."
                        )
                    _test_ys = self.tst_data[self.class_label].values
                    _tst_data_features = self.tst_data.drop(
                        self.class_label, axis=1
                    ).values
                except RuntimeError as e:
                    raise RuntimeError(f"Failed to drop class label: {self.class_label}") from e

                # get prediction for each unseen observation
                if predict:
                    try:
                        with Timer() as t:
                            try:
                                print("> Predict test", end="\r")
                                self.yhat = self.fit_mod.predict(_tst_data_features)
                                self.tst_data["pred"] = self.yhat
                                self.pred_fn = pathlib.Path(
                                    self.WRK_DIR,
                                    self.fit_mod_name + "__pred_on_" + self.tst_fn,
                                )

                            except RuntimeError as e:
                                raise RuntimeError("Model predict failed.") from e
                        try:    
                            self.tst_data.to_csv(self.pred_fn, index=False)    
                            print(f"- Predicted test saved to -> {str(self.pred_fn)} ({'%.03f'% t.interval} sec.)")
                        except IOError as e:
                            raise IOError("Failed to write predicted values to file.") from e

                    except RuntimeError as e:
                        raise RuntimeError("predict test failed.") from e

                # get prediction probablities (default False)
                if predict_proba:
                    try:
                        with Timer() as t:

                            print(
                                "> Predict proba",
                                end="\r",
                            )
                            self.pred_prob = self.fit_mod.predict_proba(
                                _tst_data_features
                            )
                            self.tst_data["pred_not_rel"] = self.pred_prob[:, 0]
                            self.tst_data["pred_rel"] = self.pred_prob[:, 1]
                            fn = pathlib.Path(
                                self.WRK_DIR,
                                self.fit_mod_name
                                + "__pred_proba_on_"
                                + self.tst_fn,
                            )
                            try:
                                self.tst_data.to_csv(fn, index=False)
                            except IOError as e:
                                raise IOError("failed to write test data to csv.") from e

                        print(f"- Predicted proba saved to -> {str(fn)} ({'%.03f'% t.interval} sec.)")    

                    except RuntimeError as e:
                        raise RuntimeError("model predict proba failed") from e

                if roc_prc:
                    print("> Plot ROC/PRC", end="\r")
                    try:
                        with Timer() as t:
                            self._roc_prc_curves(Xs=_tst_data_features, ys=_test_ys)

                        print(f"- ROC/PRC plot saved to -> {self.FIGS_DIR} ({'%.03f'% t.interval} sec.)")

                    except RuntimeError as e:
                        raise RuntimeError("failed to create roc/prc curves") from e

                if model_metrics:
                    print("> Plot model metrics", end="\r")
                    try:
                        with Timer() as t:

                            self._plot_model_stats()
                        print(f"- Model metrics saved to -> {self.FIGS_DIR} ({'%.03f'% t.interval} sec.) ")        

                    except RuntimeError as e:
                        raise RuntimeError("failed to plot model metrics") from e

                if acc_score:
                    if acc_score and not predict:
                        print(
                            "NOTE: Skipping accuracy score--must set predict=True to produce accuracy score."
                        )

                    else:
                        print("> Check accuracy", end="\r")
                        try:
                            with Timer() as t:
                                acc = accuracy_score(
                                    self.tst_data[self.class_label].values,
                                    self.tst_data["pred"].values,
                                )
                                class_names = ["Not Relevant", "Relevant"]

                                y_truth = self.tst_data[self.class_label].values
                                y_pred = self.tst_data["pred"].values

                                if threshold != 0.5:
                                    y_pred[y_pred > threshold] = 1
                                    y_pred[y_pred < threshold] = 0

                                # accuracy scores
                                bal_acc = balanced_accuracy_score(y_truth, y_pred)
                                acc, bal_acc, err = (
                                    round(acc, 2),
                                    round(bal_acc, 2),
                                    round(1 - acc, 2),
                                )
                            print(f"- Accuracy: {str(acc)} | Balanced Accuracy: {str(bal_acc)} | Error: {str(err)} ({'%.03f'% t.interval} sec.) ")

                        except RuntimeError as e:
                            print(f"[FAIL] Check accuracy scores using {self.fit_mod_name}")
                            raise RuntimeError("failed to plot model stats") from e    

                # classification report to figures
                model_cm = None
                if class_rpt:

                    try: 
                        # classification report
                        cr = classification_report(
                            y_truth, y_pred, target_names=class_names
                        )
                    except RuntimeError as e:
                        raise RuntimeError ("failed to create classification report.") from e

                    try:
                        _cr_file = pathlib.Path(self.WRK_DIR,self.fit_mod_name + "__class_report_.txt")
                        with open(_cr_file, "w", encoding="utf-8") as text_file:
                            text_file.write(cr)

                    except IOError as e:
                        raise IOError ("failed to write classification report.") from e        

                    try:
                        # confusion matrix
                        model_cm = confusion_matrix(y_true=y_truth, y_pred=y_pred)
                        plt.figure(dpi=self.FIG_DPI).set_size_inches(
                            self.FIG_W, self.FIG_H
                        )
                        plot_confusion_matrix(
                            model_cm,
                            classes=class_names,
                            #title="Confusion matrix - (" + str(threshold) + ")",
                            title="Confusion matrix",
                        )
                        plt.savefig(
                            pathlib.Path(
                                self.FIGS_DIR, self.fit_mod_name + "_fig_cm.svg"
                            )
                        )
                        plt.close()
                        if not os.path.isfile(pathlib.Path(self.FIGS_DIR, self.fit_mod_name + "_fig_cm.svg")):
                            raise IOError("failed to write confusion matrix.")

                        # confusion matrix normalized
                        plt.figure(dpi=self.FIG_DPI).set_size_inches(
                            self.FIG_W, self.FIG_H
                        )
                        plot_confusion_matrix(
                            model_cm,
                            normalize=True,
                            classes=class_names,
                            title="Confusion matrix (norm)",
                        )
                        plt.savefig(
                            pathlib.Path(
                                self.FIGS_DIR,
                                self.fit_mod_name + "_fig_cm_norm.svg",
                            )
                        )
                        plt.close()
                        if not os.path.isfile(pathlib.Path(self.FIGS_DIR, self.fit_mod_name + "_fig_cm_norm.svg")):
                            raise IOError("failed to write confusion matrix (normlalized).")

                        # ROC plot
                        plt.figure(dpi=self.FIG_DPI).set_size_inches(
                            self.FIG_W, self.FIG_H
                        )
                        rcd.from_predictions(y_truth, y_pred, name="ROC")
                        plt.savefig(
                            pathlib.Path(
                                self.FIGS_DIR,
                                self.fit_mod_name + "_fig_ROC_Disp_frm_pred.svg"))
                        plt.close()
                        if not os.path.isfile(pathlib.Path(self.FIGS_DIR, self.fit_mod_name + "_fig_ROC_Disp_frm_pred.svg")):
                            raise IOError("failed to write ROC display from pred.")

                        # PRC threshold
                        plt.figure(dpi=self.FIG_DPI).set_size_inches(
                            self.FIG_W, self.FIG_H
                        )
                        # plt.figure.suptitle("2-class Precision-Recall curve")
                        # plt_ = precision_recall_threshold(
                        #     self.fit_mod,
                        #     _tst_data_features,
                        #     y_truth,
                        #     y_preds=self.yhat,
                        #     thold=threshold,
                        #     title="",
                        # )
                        pcd.from_predictions(y_truth, y_pred, name="MLCS")

                        plt.savefig(
                            pathlib.Path(
                                self.FIGS_DIR,
                                self.fit_mod_name + "_prc_threshold.svg",
                            )
                        )
                        plt.close()
                        if not os.path.isfile(pathlib.Path(self.FIGS_DIR, self.fit_mod_name + "_prc_threshold.svg")):
                            raise IOError("failed to write PRC threshold.")

                        print(f"- Confusion matrix and plotts in f-> {self.FIGS_DIR}"
                                )
                    except RuntimeError as e:
                        raise RuntimeError("Error creating plots.") from e
        else:
            raise RuntimeError(
                "test data not loaded - load test data with load_test_data(tst_data_filename.csv)"
            )
        return acc, bal_acc, err, model_cm

    # PLOTS FOR INTERPRETING THE MODEL referenced from
    # https://github.com/UrbsLab/scikit-ExSTraCS/blob/master/scikit-ExSTraCS%20User%20Guide.ipynb
    def _roc_prc_curves(self, Xs=None, ys=None, verbose=False) -> None:
        """Create ROC and PRC curves based on current model and training data.

        Args:
            Xs (ndarray, optional): x-values for provided data. Defaults to None.
            ys (array, optional): y-values for provided data. Defaults to None.
            verbose (bool, optional): Print metrics to console. Defaults to False.

        Raises:
            IOError: If failed to write ROC curve.
            IOError: If failed to write PRC curve.
            RuntimeError: If failed to create plots for ROC/PRC curve.
            RuntimeError: If is not fit or loaded before producing ROC/PRC plot.
        """
        if not any(elem is None for elem in [Xs, ys, self.fit_mod]):
            if self.pred_probs is None:
                self.pred_probs = self.fit_mod.predict_proba(Xs)

            try:
                # no skill pred
                ns_probs = [0 for _ in range(len(ys))]
                # keep probabilities for the positive outcome only
                pos_probs = self.pred_probs[:, 1]

                # calculate scores
                ns_auc = roc_auc_score(ys, ns_probs)
                lcs_auc = roc_auc_score(ys, pos_probs)

                # summarize scores
                # print("-- LCS AUC=%.3f" % (lcs_auc), end="")
                # print(", No Skill AUC=%.3f" % (ns_auc), end="")

                # calculate roc curves
                ns_fpr, ns_tpr, _ = roc_curve(ys, ns_probs)
                fpr, tpr, _ = roc_curve(ys, pos_probs)

                # plot the roc curve for the model
                plt.figure(dpi=self.FIG_DPI).set_size_inches(self.FIG_W, self.FIG_H)
                plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
                plt.plot(fpr, tpr, marker=".", label="LCS")

                # axis labels
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")

                # show the legend
                plt.legend()

                # save figure
                plt.savefig(
                    pathlib.Path(self.FIGS_DIR, self.fit_mod_name + "_fig_ROC.svg")
                )
                plt.close()

                if not os.path.isfile(pathlib.Path(self.FIGS_DIR, self.fit_mod_name + "_fig_ROC.svg")):
                    raise IOError("failed to write ROC curve.")

                ### Now precision recall curve
                # predict class values
                precision, recall, _ = precision_recall_curve(ys, pos_probs)
                f1, lcs_auc = f1_score(ys, self.yhat), auc(recall, precision)

                # summarize scores
                print("-- F1=%.3f, AUC=%.3f" % (f1, lcs_auc))

                # plot the precision-recall curves
                # no_skill = len(ys[ys==1]) / len(ys)
                # plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
                plt.plot(recall, precision, marker=".", label="LCS")
                # axis labels
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                # show the legend
                plt.legend()
                plt.savefig(
                    pathlib.Path(self.FIGS_DIR, self.fit_mod_name + "_fig_PRC.svg")
                )
                plt.close()
                if not os.path.isfile(pathlib.Path(self.FIGS_DIR, self.fit_mod_name + "_fig_PRC.svg")):
                    raise IOError("failed to write PRC curve.")

            except RuntimeError as e:
                raise RuntimeError("Failed to create plots for ROC/PRC curve.") from e

            if verbose:
                print("\n" + "PRC AUC: \t" + str(round(auc(recall, precision), 3)))
                print("ROC AUC:\t" + str(round(auc(fpr, tpr), 3)))

                print(
                    "Final Training Accuracy: "
                    + str(self.fit_mod.get_final_training_accuracy())
                )
                print(
                    "Final Instance Coverage: "
                    + str(self.fit_mod.get_final_instance_coverage())
                )
                print(
                    "Final Attribute Specificity List: "
                    + str(self.fit_mod.get_final_attribute_specificity_list())
                )
                print(
                    "Final Attribute Accuracy List: "
                    + str(self.fit_mod.get_final_attribute_accuracy_list())
                )
                print("Final Attribute Tracking Sums:")
                print(self.fit_mod.get_final_attribute_tracking_sums())
                print("Final Attribute Cooccurences:")
                print(self.fit_mod.get_final_attribute_coocurrences(self.hdrs.values))
        else:
            raise RuntimeError("model must be fit or loaded before producing ROC/PRC plot.")

    def _plot_model_stats(self, iter_file = None) -> None:
        """Create plots for the fit_mod if training metadata is available.

        Raises:
            IOError: If failed to create plots for model metrics.
            FileNotFoundError If iterData file is not found.
            RuntimeError: If model not fit.
        """
        if iter_file is not None:
            _iter_data_file = pathlib.Path(iter_file)    
        else:
            #_mod_name = "".join(self.fit_mod_name.rsplit("_model", 1))
            _iter_data_file = pathlib.Path(
                self.WRK_DIR, "".join(self.fit_mod_name.rsplit("_model", 1)) + "_iterData.csv"
            )
            
        if not os.path.isfile(_iter_data_file):
            raise FileNotFoundError(f"iterData not found at {str(_iter_data_file)}!")

        if self.fit_mod is not None:
            data_tracking = pd.read_csv(_iter_data_file)

            iterations = data_tracking["Iteration"].values
            accuracy = data_tracking["Accuracy (approx)"].values
            generality = data_tracking["Average Population Generality"].values
            macro_pop = data_tracking["Macropopulation Size"].values
            micro_pop = data_tracking["Micropopulation Size"].values
            m_size = data_tracking["Match Set Size"].values
            c_size = data_tracking["Correct Set Size"].values
            experience = data_tracking[
                "Average Iteration Age of Correct Set Classifiers"
            ].values
            subsumption = data_tracking["# Classifiers Subsumed in Iteration"].values
            crossover = data_tracking[
                "# Crossover Operations Performed in Iteration"
            ].values
            mutation = data_tracking[
                "# Mutation Operations Performed in Iteration"
            ].values
            covering = data_tracking[
                "# Covering Operations Performed in Iteration"
            ].values
            deletion = data_tracking[
                "# Deletion Operations Performed in Iteration"
            ].values
            rc = data_tracking["# Rules Removed via Rule Compaction"].values

            g_time = data_tracking["Total Global Time"].values
            m_time = data_tracking["Total Matching Time"].values
            cross_time = data_tracking["Total Crossover Time"].values
            cov_time = data_tracking["Total Covering Time"].values
            mut_time = data_tracking["Total Mutation Time"].values
            at_time = data_tracking["Total Attribute Tracking Time"].values
            init_time = data_tracking["Total Model Initialization Time"].values
            rc_time = data_tracking["Total Rule Compaction Time"].values
            del_time = data_tracking["Total Deletion Time"].values
            sub_time = data_tracking["Total Subsumption Time"].values
            sel_time = data_tracking["Total Selection Time"].values
            eval_time = data_tracking["Total Evaluation Time"].values

            _met_dir = pathlib.Path()
            try:
                _out = pathlib.Path(self.FIGS_DIR, self.fit_mod_name + "_metrics_acc-gen_vs_iter.svg")
                plt.figure(dpi=self.FIG_DPI).set_size_inches(self.FIG_W, self.FIG_H)
                plt.plot(iterations, accuracy, label="approx accuracy")
                plt.plot(iterations, generality, label="avg generality")
                plt.xlabel("Iteration")
                plt.ylabel("Accuracy/Generality")
                plt.legend()
                plt.savefig(_out)
                plt.close()

                if not os.path.isfile(_out):
                    raise IOError("failed to write _metrics_acc-gen_vs_iter")

                plt.figure(dpi=self.FIG_DPI).set_size_inches(self.FIG_W, self.FIG_H)
                plt.plot(iterations, macro_pop, label="MacroPop Size")
                plt.plot(iterations, micro_pop, label="MicroPop Size")
                plt.xlabel("Iteration")
                plt.ylabel("Macro/MicroPop Size")
                plt.legend()
                _out = pathlib.Path(self.FIGS_DIR, self.fit_mod_name + "_metrics_macro_pop-micro_vs_iter.svg")
                plt.savefig(_out)
                plt.close()
                if not os.path.isfile(_out):
                    raise IOError("failed to write _metrics_macro_pop-micro_vs_iter")

                plt.figure(dpi=self.FIG_DPI).set_size_inches(self.FIG_W, self.FIG_H)
                plt.plot(iterations, m_size, label="[M] size")
                plt.plot(iterations, movingAvg(m_size), label="[M] size movingAvg")
                plt.plot(iterations, c_size, label="[C] size")
                plt.plot(iterations, movingAvg(c_size), label="[C] size movingAvg")
                plt.xlabel("Iteration")
                plt.ylabel("[M]/[C] size per iteration")
                plt.legend()
                _out = pathlib.Path(self.FIGS_DIR, self.fit_mod_name + "_metrics_match-correct_vs_iter.svg")
                plt.savefig(_out)
                plt.close()

                if not os.path.isfile(_out):
                    raise IOError("failed to write _metrics_match-correct_vs_iter")

                plt.figure(dpi=self.FIG_DPI).set_size_inches(self.FIG_W, self.FIG_H)
                plt.plot(iterations, experience)
                plt.ylabel("Average [C] Classifier Age")
                plt.xlabel("Iteration")
                _out = pathlib.Path(self.FIGS_DIR, self.fit_mod_name + "_metrics_class-age_vs_iter.svg")
                plt.savefig(_out)
                plt.close()

                if not os.path.isfile(_out):
                    raise IOError("failed to write _metrics_class-age_vs_iter")

                plt.figure(dpi=self.FIG_DPI).set_size_inches(self.FIG_W, self.FIG_H)
                plt.plot(
                    iterations, cumulativeFreq(subsumption), label="Subsumption Count"
                )
                plt.plot(iterations, cumulativeFreq(crossover), label="Crossover Count")
                plt.plot(iterations, cumulativeFreq(mutation), label="Mutation Count")
                plt.plot(iterations, cumulativeFreq(deletion), label="Deletion Count")
                plt.plot(iterations, cumulativeFreq(covering), label="Covering Count")
                plt.plot(iterations, cumulativeFreq(rc), label="RC Count")
                plt.xlabel("Iteration")
                plt.ylabel("Cumulative Operations Count Over Iterations")
                plt.legend()
                _out = pathlib.Path(self.FIGS_DIR, self.fit_mod_name + "_metrics_cum-op-cnt_vs_iter.svg")
                plt.savefig(_out)
                plt.close()

                if not os.path.isfile(_out):
                    raise IOError("failed to write _metrics_cum-op-cnt_vs_iter")

                plt.figure(dpi=self.FIG_DPI).set_size_inches(self.FIG_W, self.FIG_H)
                plt.plot(iterations, init_time, label="Init Time")
                plt.plot(iterations, m_time + init_time, label="Matching Time")
                plt.plot(iterations, cov_time + m_time + init_time, label="Covering Time")
                plt.plot(
                    iterations,
                    sel_time + cov_time + m_time + init_time,
                    label="Selection Time",
                )
                plt.plot(
                    iterations,
                    cross_time + sel_time + cov_time + m_time + init_time,
                    label="Crossover Time",
                )
                plt.plot(
                    iterations,
                    mut_time + cross_time + sel_time + cov_time + m_time + init_time,
                    label="Mutation Time",
                )
                plt.plot(
                    iterations,
                    sub_time
                    + mut_time
                    + cross_time
                    + sel_time
                    + cov_time
                    + m_time
                    + init_time,
                    label="Subsumption Time",
                )
                plt.plot(
                    iterations,
                    at_time
                    + sub_time
                    + mut_time
                    + cross_time
                    + sel_time
                    + cov_time
                    + m_time
                    + init_time,
                    label="AT Time",
                )
                plt.plot(
                    iterations,
                    del_time
                    + at_time
                    + sub_time
                    + mut_time
                    + cross_time
                    + sel_time
                    + cov_time
                    + m_time
                    + init_time,
                    label="Deletion Time",
                )
                plt.plot(
                    iterations,
                    rc_time
                    + del_time
                    + at_time
                    + sub_time
                    + mut_time
                    + cross_time
                    + sel_time
                    + cov_time
                    + m_time
                    + init_time,
                    label="RC Time",
                )
                plt.plot(
                    iterations,
                    eval_time
                    + rc_time
                    + del_time
                    + at_time
                    + sub_time
                    + mut_time
                    + cross_time
                    + sel_time
                    + cov_time
                    + m_time
                    + init_time,
                    label="Evaluation Time",
                )
                plt.plot(iterations, g_time, label="Total Time")
                plt.xlabel("Iteration")
                plt.ylabel("Cumulative Time (Stacked)")
                plt.legend()
                _out = pathlib.Path(self.FIGS_DIR, self.fit_mod_name + "_metrics_cum-time_vs_iter.svg")
                plt.savefig(_out)
                plt.close()

                if not os.path.isfile(_out):
                    raise IOError("failed to write _metrics_cum-time_vs_iter")

            except RuntimeError as e:
                raise IOError("plot model metrics failed.") from e
        else:
            raise RuntimeError("model must be fit or loaded before plotting model stats.")

    def _set_wrk_dir(self, mod_path: pathlib.Path, dir=False):
        """Internal method to set working directory.

        Args:
            mod_path (pathlib.Path): path object to the loaded model.
            dir (bool, optional): boolean object representing if the path is a directory. Defaults to False.

        Raises:
            IOError: If failed to set working directory. 
        """
        try:
            if dir:
                # its only the work dir
                assert os.path.isdir(
                    mod_path
                ), f" model directory not found at {mod_path}"
                _mod_dir = str(mod_path)
                _fig_dir = str(pathlib.Path(mod_path, "figures"))

            else:
                assert os.path.isfile(mod_path), f" model file not found at {mod_path}"
                _mod_dir = str(mod_path.parent)
                _mod_file = str(mod_path.name)
                _mod_name = _mod_file.replace(".pkl", "")
                _fig_dir = str(pathlib.Path(_mod_dir, "figures"))
                self.fit_mod_name = _mod_name
                self.name = _mod_name

            if not os.path.exists(_fig_dir):
                os.makedirs(_fig_dir)

            self.WRK_DIR = _mod_dir
            self.FIGS_DIR = _fig_dir

        except IOError as e:
            raise IOError("failed to set working directory.") from e

    def do_analysis(self):
        """ Method to create reports used to do analysis, including FP/FN reports, and FULL and SUMMARY final reports.
        """
        assert self.pred_fn is not None, "prediction file not created"
        assert self.fit_mod_name is not None, "model must be fit and loaded before doing analysis"
        
        # get file name from feature vector training file name
        dti_file = f"data/{self.trn_fn.replace('fv_','')}"
        
        # get path to model prediction on test data file
        mod_pred_file = str(self.pred_fn)
        assert os.path.isfile(mod_pred_file), "model prediction file not found"

        # get dti file to match with predicted, so content is shown during review
        dti_df = pd.read_csv(dti_file)
        mod_pred_df = pd.read_csv(mod_pred_file)
        
        # create FP and FN reports in model directory
        analyze_model(dti_df,mod_pred_df,'rel',self.WRK_DIR)

        # copy EK rules in model dir.
        copy_ek_rules(self.WRK_DIR, self.rules)

        # produce final prediction report
        self.report_path = str(report(dti_df,mod_pred_df, self.WRK_DIR))

    def pred_from_model(self, model_path=None, trn_data=None, tst_data=None, class_label = None, out_dir='.', sum_flds = None) -> pathlib:
        """Predict on unseen data using an existing model located at model_path.  

        Args:
            model_path (str): path to the model being used to predict.
            trn_data (str): path to training data used to train the model.
            tst_data (str): path to unsee data.
            class_label (str): class label.
            out_dir (str, optional): path to output directory. Defaults to '.'.

        Raises:
            RuntimeError: If failed to drop empty columsn from training and test data sets before training.
            ValueError: If class label is not found in data training and unseen datasets.
            ValueError: If unseen data headers do not match training headers.
            RuntimeError: If failed to drop class label.
            RuntimeError: If model predict failed.
            IOError: If failed to write predicted values to file.

        Returns:
            pathlib: _description_
        """
        # load model
        self.load_model(model_path)

        # set class label
        self.class_label = class_label

        # load data
        self.load_train_data(trn_data_fn=trn_data)
        self.load_test_data(tst_data_fn=tst_data)

        # Drop all features that are not present in both datasets.
        try:
            cols_in_trn = self.trn_data.loc[:, ~self.trn_data.any()].columns.values
            cols_in_tst = self.tst_data.loc[:, ~self.trn_data.any()].columns.values
            drop_cols = set(cols_in_trn) & set(cols_in_tst)
            self.trn_data.drop(drop_cols, axis=1, inplace=True)
            self.tst_data.drop(drop_cols, axis=1, inplace=True)
        except RuntimeError as drop_dataframes_error:
            raise RuntimeError(
                "failed to drop empty columsn from training and test data sets before training"
            ) from drop_dataframes_error

        # get prediction
        if self.tst_data is not None and self.trn_data is not None:
            if self.fit_mod is not None:
                self.hdrs = self.trn_data.columns.tolist()
                _tst_hdrs = self.tst_data.columns.tolist()

            if self.class_label not in self.hdrs or self.class_label not in _tst_hdrs:
                raise ValueError(
                    f"Class label {class_label} not found in data training and test datasets.\n Acutal labels are not required for only predicting, however, the column must exist in the test data."
                )

            try:
                # get header names while still a df
                self.tst_hdrs = self.tst_data.columns.tolist()
                # get test data values
                if set(self.hdrs) != set(self.tst_hdrs):
                    raise ValueError(
                        "test_model: test data hdrs do not match training hdrs."
                    )
                # _test_ys = self.tst_data[self.class_label].values
                _tst_data_features = self.tst_data.drop(self.class_label, axis=1).values
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to drop class label: {self.class_label}"
                ) from e

            # get prediction for each unseen observation
            with Timer() as t:
                try:
                    print("> Predict test", end="\r")
                    self.yhat = self.fit_mod.predict(_tst_data_features)
                    self.tst_data["pred"] = self.yhat
                    self.pred_fn = pathlib.Path(
                        out_dir,
                        self.fit_mod_name + "__pred_on_" + self.tst_fn,
                    )

                except RuntimeError as e:
                    raise RuntimeError("Model predict failed.") from e

            try:
                self.tst_data.to_csv(self.pred_fn, index=False)
                print(
                    f"- Predicted test saved to -> {str(self.pred_fn)} ({'%.03f'% t.interval} sec.)"
                )
            except IOError as e:
                raise IOError("Failed to write predicted values to file.") from e

            train_fn = str(self.pred_fn).split("_fv_")[1]
            dti_file = f"data/{train_fn}"
            # mod_pred_file = str(self.pred_fn)
            # Analyze model
            dti_df = pd.read_csv(dti_file)
            print("Loaded dti file: " + dti_file)

            # produce final prediction report
            # report_path = report(dti_df,self.tst_data, out_dir)
            # reset index
            dti_df.reset_index()
            self.tst_data.reset_index()

            # combine dti file with model pred
            result = pd.merge(dti_df, self.tst_data, left_index=True, right_index=True)
            # get mask from features and prediction
            mask = self.tst_data == 1

            features = mask.columns.values
            for index, col in mask.items():
                mask.loc[mask[index] == True, index] = index
                mask.loc[mask[index] == False, index] = "#"
            self.tst_data["tags"] = mask[features].agg(",".join, axis=1)
            self.tst_data["tags"] = self.tst_data["tags"].str.replace("#,", "")
            self.tst_data["tags"] = self.tst_data["tags"].str.replace(",#", "")
            self.tst_data["tags"] = self.tst_data["tags"].str.replace("#", "")

            dti_df["tags"] = self.tst_data["tags"]
            dti_df["pred"] = result["pred"]

            dti_df.sort_values("datetime", ascending=True, inplace=True)
            # full report
            full_report = pathlib.Path(
                out_dir, f"{self.name}_FULL_dti_predict_report.csv"
            )
            dti_df.to_csv(full_report, index=False)

            # summary report, only predicted relevant
            dti_df = dti_df[dti_df["pred"] == 1]
            
            # if summary report columns is specified apply the columns
            try:
                if sum_flds:
                    assert isinstance(sum_flds, list)
                    dti_df = dti_df[sum_flds]
            except KeyError as e:
                raise KeyError('failed to select summary fields specified.')
                
            # drop class label and pred
            if self.class_label in dti_df:
                dti_df.drop([self.class_label], axis=1, inplace=True)
                
            summ_out = pathlib.Path(out_dir, f"{self.name}_SUMMARY_report.csv")
            dti_df.to_csv(summ_out, index=False)

            # "- Loaded train data: \t{str(_data_file)} ({len(self.trn_data)} rows)  "
            print(f"Summary report ready for review at -> {str(summ_out)}")

        return summ_out
