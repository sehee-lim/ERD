import pandas as pd
import argparse
import re

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


pd.set_option('display.max_rows', None)

parser = argparse.ArgumentParser()

parser.add_argument('--folder_name', type = str)
parser.add_argument('--subset', action = 'store_true')
parser.add_argument('--subset_idx', type = str, default = 231126)



def update_dominant_distortion(row):
    # treat secondary distortion as a valid ground truth value
    if row["Distortion Type"] == row["Secondary Distortion"]:
        row["Dominant Distortion"] = row["Secondary Distortion"]
    return row

if __name__ == '__main__':
    args = parser.parse_args()
    
    # load data 
    if args.subset:
        res = pd.read_csv(f'output/{args.folder_name}/SubsetIDX-{args.subset_idx}-exp_result.csv', encoding = 'latin1')
    else:
        res = pd.read_csv(f'output/{args.folder_name}/FullDataset-exp_result.csv', encoding = "latin1")
        

    # convert string values to lowercase
    res["Dominant Distortion"] = res["Dominant Distortion"].str.lower()
    res["Secondary Distortion"] = res["Secondary Distortion"].str.lower()
    res["Distortion Type"] = res["Distortion Type"].str.lower()
    res['Distortion Type'] = res['Distortion Type'].apply(lambda x: re.sub(r'\d+\.\s+', '', x) if isinstance(x, str) else x)
    
    # display predictions
    print()
    print(">> PREDICTION <<")
    print(res["Distortion Assessment"].value_counts())
    print(res["Distortion Type"].value_counts())

    # display true values
    print()
    print(">> TRUE <<")
    print(res["Dominant Distortion"].value_counts())

    # evaluation for binary classification
    print()
    print(">> EVALUATION <<")
    true_assessment = res['Dominant Distortion'].apply(lambda x: True if x != 'no distortion' else False)
    pred_assessment = res["Distortion Assessment"] 
    f1_binary = f1_score(true_assessment, res["Distortion Assessment"])
    f1_binary_weighted = f1_score(true_assessment, res["Distortion Assessment"], average = "weighted")
    print("Distortion Assessment:", f1_binary)
    # print("Distortion Assessment(weighted):", f1_binary_weighted)

    # evaluation for multi-class classification
    true_classification = res["Dominant Distortion"]
    res.loc[~pred_assessment, "Distortion Type"] = "no distortion"
    f1_multi = f1_score(true_classification, res["Distortion Type"], average = "weighted")
    f1_multi2 = f1_score(true_classification[true_assessment & pred_assessment], res["Distortion Type"][true_assessment & pred_assessment], average = "weighted")
    res = res.apply(update_dominant_distortion, axis = 1)
    true_classification = res["Dominant Distortion"]
    f1_multi3 = f1_score(true_classification[true_assessment & pred_assessment], res["Distortion Type"][true_assessment & pred_assessment], average = "weighted")
    # print("Distortion Classificaion:", f1_multi)
    # print("Distortion Classificaion(excluding No distortion):", f1_multi2)
    # print("Distortion Classificaion(+Secondary Distortion):", f1_multi3)
    print("Distortion Classificaion:", f1_multi3)
    
    # display the confusion matrix
    print()
    print(">> CONFUSION MATRIX <<")
    confusion_matrix = confusion_matrix(true_assessment, res["Distortion Assessment"])
    print(confusion_matrix)
    print("No Distortion Accuracy:", confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1]))
    accuracy = accuracy_score(true_assessment, pred_assessment)
    precision = precision_score(true_assessment, pred_assessment, zero_division = 0)
    recall = recall_score(true_assessment, pred_assessment, zero_division = 0)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    
    
     
    