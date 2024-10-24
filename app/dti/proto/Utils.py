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
import time, os
import json, csv, shutil
import numpy as np
import itertools, time
from collections import abc
from rich.console import Console
from rich.panel import Panel
import pandas as pd
import pathlib
import streamlit as st
from sklearn.metrics import (
    precision_recall_curve,
)
from sklearn.metrics import RocCurveDisplay as rcd
import matplotlib.pyplot as plt

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

def nested_dict_iter(nested: dict) -> dict:
    for key, value in nested.items():
        if isinstance(value, abc.Mapping):
            for inner_key, inner_value in nested_dict_iter(value):
                yield inner_key, inner_value
        else:
            yield key, value


def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        if isinstance(row, list):
            flat_list.extend(row)
        else:
            flat_list.append(row)
    return flat_list


def file_type_detector(file_path):
    # Check the file extension first
    if file_path.endswith('.json'):
        return 'json'
    elif file_path.endswith('.csv'):
        return 'csv'
    else:
        # If the extension is not definitive, check the content
        try:
            with open(file_path, 'r') as file:
                # Try parsing as JSON
                json.load(file)
                return 'json'
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        try:
            with open(file_path, 'r') as file:
                # Try reading as CSV
                reader = csv.reader(file)
                rows = list(reader)
                if rows:
                    return 'csv'
        except UnicodeDecodeError:
            pass

    return 'unknown'


def pair_start_stop(starts, stops):
    paired_starts = []
    paired_stops = []
    stop_iter = iter(stops)

    try:
        current_stop = next(stop_iter)
        if current_stop is None:
            return paired_starts, paired_stops
        
        for start in starts:
            while current_stop < start:
                current_stop = next(stop_iter)
            paired_starts.append(start)
            paired_stops.append(current_stop)
    except StopIteration:
        #print("Ran out of stop indexes. Last start index was:", start)
        return paired_starts, paired_stops

    return paired_starts, paired_stops


# cumulative frequencey utility used to sum a series
def cumulativeFreq(freq) -> None:
    a = []
    c = []
    for i in freq:
        a.append(i + sum(c))
        c.append(i)
    return np.array(a)


# moving average utility
def movingAvg(a, threshold=300) -> None:
    weights = np.repeat(1.0, threshold) / threshold
    conv = np.convolve(a, weights, "valid")
    return np.append(
        conv,
        np.full(threshold - 1, conv[conv.size - 1]),
    )


def splash() -> None:
   
    print("") 
    print("     _ _   _")
    print("    | | | (_)")
    print("  __| | |_ _ ")
    print(" / _` | __| |")
    print("| (_| | |_| |")
    print(" \__,_|\__|_|")
    print("\n")          

def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function is from https://sklearn.org/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


## Helper for precision recall curve
def precision_recall_threshold(model, X, y, thold=0.5, title="", y_preds=None) -> plt:
    try:
        if y_preds is None:
            probs_y = model.predict(X)
        else:
            probs_y = y_preds
    except:
        raise Exception("failed to get predictions during precision_recall_thrshopld call.")
    try:
        # using threshold
        probs_y[probs_y > thold] = 1
        probs_y[probs_y < thold] = 0
        precision, recall, thresholds = precision_recall_curve(y, probs_y)
    except:
        raise Exception("failed to create recision recall curve.")
    
    try:
        plt.title(title + " Precision-Recall vs Threshold Chart")
        plt.plot(thresholds, precision[:-1], "b--", label="Precision")
        plt.plot(thresholds, recall[:-1], "r--", label="Recall")
        plt.ylabel("Precision, Recall")
        plt.xlabel("Threshold")
        plt.legend(loc="upper left")
        plt.ylim([0, 1])
        plt.tight_layout()
        return plt
    except:
        raise Exception("failed to plot precision recall vs threshold plt.")
    
def check_regex(pattern):
        import re
        # pattern is a string containing the regex pattern
        try:
            re.compile(pattern)
        except re.error:
            print("Non valid regex pattern - remember to not end the regex with a double backslash '\\'!")
            print(f'---> Bad pattern {pattern}')
            return False
        return True
    
def test_regex(pattern, field, dataframe):
    # check if leading/trailing forward slash is present
    if pattern[0] == '/' and pattern[-1] == '/':
        _value = pattern[1:-1]
    else:
        _value = pattern

    # print("Regex starting with: "+_value)
    if check_regex(_value):
        print("  [OK] Pattern: "+_value)
        results = dataframe[field].str.contains(_value, regex=True).values
        # dataframe.loc[results]
        results_tbl = dataframe.loc[results][[field, 'rel']]
        print(results_tbl)

        print(f"Sum of rel field: {results_tbl['rel'].sum()}")
        # print(test_df[test_df['rel']>0]['message'])
        
def analyze_model(dti_df, mod_pred_df, class_name, target_dir):
    """Create false positive and false negative reports. 

    Args:
        dti_df (DataFrame): The dti file used to create feature vectors, this file is automatically generated and found in the data directory. 
        mod_pred_df (DataFrame): The prediction on test data loaded from predict output.
        class_name (str): Class name (column name).
        target_dir (str): Output directory for the reports.

    Raises:
        RuntimeError: If failed to create false positive and false negative reports.
    """
    try:
        # parse model name using split of full file path
        model_name = target_dir.split("\\")[1]

        # remove class label
        dti_df.drop(class_name, axis=1, inplace=True)

        # reset index
        dti_df.reset_index()
        mod_pred_df.reset_index()

        # combine dti file with model pred
        result = pd.merge(dti_df, mod_pred_df,
                          left_index=True, right_index=True)

        # FP check
        FPOS = result.loc[(result[class_name] == 0) & (result['pred'] == 1)]
        FPOS.to_csv(f'{target_dir}/{model_name}_false_positive_report.csv')

        # FN check
        FNEG = result.loc[(result[class_name] == 1) & (result['pred'] == 0)]
        FNEG.to_csv(f'{target_dir}/{model_name}_false_negative_report.csv')

    except RuntimeError as e:
        raise RuntimeError(f"Failed to create FP and FN reports.") from e
    

def copy_ek_rules(target_dir, rule_dir):
    try:
        # copy ek rules used to build the datasets into the model output directory
        shutil.copytree(rule_dir, target_dir+'/ek_rules')

    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to colpy atomics to model directory.") from e

def report(dti_df, mod_pred_df, target_dir):
    try:
        # parse model name using split of full file path
        model_name = target_dir.split("\\")[1]

        # remove class label
        #dti_df.drop(class_name, axis=1, inplace=True)

        # reset index
        dti_df.reset_index()
        mod_pred_df.reset_index()

        # combine dti file with model pred
        result = pd.merge(dti_df, mod_pred_df,
                          left_index=True, right_index=True)
        # get mask from features and prediction
        mask = mod_pred_df == 1
        
        features = mask.columns.values
        for index, col in mask.items():
            mask.loc[mask[index]==True, index] = index
            mask.loc[mask[index]==False,index] = "#"
        mod_pred_df['tags'] = mask[features].agg(','.join, axis=1)
        mod_pred_df['tags'] = mod_pred_df['tags'].str.replace('#,','')
        mod_pred_df['tags'] = mod_pred_df['tags'].str.replace(',#','')
        mod_pred_df['tags'] = mod_pred_df['tags'].str.replace('#','')
        
        dti_df['tags'] = mod_pred_df['tags']
        dti_df['pred'] = result['pred']

        dti_df.sort_values('datetime', ascending=True, inplace=True)
        
        # Full report
        dti_df.to_csv(f'{target_dir}/{model_name}_FULL_dti_predict_report.csv')
        
        # Summary report
        dti_df = dti_df[dti_df['pred'] == 1]
        summ_out = pathlib.Path(target_dir, model_name+"_SUMMARY_dti_predict_report.csv" )
        dti_df.to_csv(summ_out)
    
        #_data_file = pathlib.Path(self.DATA_DIR, trn_data_fn)
        #"- Loaded train data: \t{str(_data_file)} ({len(self.trn_data)} rows)  "
        print(f'Summary report ready for review at -> {str(summ_out)}')

        return summ_out

    except RuntimeError as e:
        raise RuntimeError(f"Failed to create prediction report.") from e