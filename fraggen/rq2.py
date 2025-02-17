import json
import os
import pandas as pd
import csv

APPS = ['addressbook', 'petclinic', 'claroline', 'dimeshift', 'pagekit', 'phoenix', 'ppma', 'mantisbt']
OUTPUT_CSV = True
setting = "within-apps" # we only do within-apps for Fraggen
baseline = 'fraggen'
feature = 'hybrid'

DISTINCT_CLASS = 0
NEAR_DUP_CLASS = 1

base_path = '/Users/kasun/Documents/uni/semester-4/thesis/Baseline-NDD'
filename = f'{base_path}/results/rq2-ALT-{setting}-{baseline}.csv'

def get_prediction(df_filtered, stateA, stateB):
    row = df_filtered[
        ((df_filtered['state1'] == stateA) & (df_filtered['state2'] == stateB)) |
        ((df_filtered['state1'] == stateB) & (df_filtered['state2'] == stateA))
        ]

    if not row.empty:
        hybrid_val = row.iloc[0]['hybrid']  # take the first match
        if hybrid_val in [0, 1]:
            return NEAR_DUP_CLASS
        elif hybrid_val == 2:
            return DISTINCT_CLASS
    else:
        print(f" missing hybrid value: {stateA}, {stateB}")
        return DISTINCT_CLASS

def fraggen_model_quality():

    with open("test-output/combinedEntries.json", "r", encoding="utf-8") as f:
        all_entries = json.load(f)

    # Name the columns according to your JSON structure
    columns = [
        "appname",  # index 0
        "col1",  # index 1
        "state1",  # index 2
        "state2",  # index 3
        "col4",
        "col5",
        "col6",
        "col7",
        "col8",
        "col9",
        "col10",
        "col11",
        "col12",
        "col13",
        "y_actual",  # index 14
        "col15",
        "hybrid",  # index 16
        "col17"
    ]
    df_json = pd.DataFrame(all_entries, columns=columns)

    df_filtered = df_json[['state1', 'state2', 'appname', 'y_actual', 'hybrid']].copy()

    df_filtered['y_actual'] = df_filtered['y_actual'].astype(int)
    df_filtered['hybrid'] = df_filtered['hybrid'].astype(int)

    if OUTPUT_CSV:
        if not os.path.exists(filename):
            header = ['Setting', 'App', 'Baseline', 'Feature', 'F1', 'Precision', 'Recall']
            with open(filename, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)

    for app in APPS:
        print(f'\n=== Coverage for App={app}  ===')
        cluster_file_name = f'{base_path}/output/{app}.json'

        model = []  # list of states that are included in model
        covered_bins = []
        number_of_bins = 0
        total_number_of_states = 0
        not_detected_near_duplicate_pairs = []
        all_comparison_pairs = []

        with open(cluster_file_name, 'r') as f:
            data = json.load(f)
            for bin in data:
                bin_index = 0  # index of the current state in the bin
                number_of_bins += 1
                bin_covered = False
                for state in data[bin]:
                    total_number_of_states += 1
                    if model == []:  # if model is empty, add the first state:
                        model.append(state)
                        bin_covered = True
                    else:
                        is_distinct = True
                        if bin_index == 0:  # if the first state in the bin, i.e. the 'key' => compare with all existing states in the model
                            bin_index += 1
                            for ms in model:  # for each state already in the model
                                pred = get_prediction(df_filtered, ms, state)
                                if pred == NEAR_DUP_CLASS:
                                    is_distinct = False
                                    break

                            if is_distinct:
                                model.append(state)
                                bin_covered = True
                        else:
                            pred = get_prediction(df_filtered, data[bin][bin_index - 1], state)
                            all_comparison_pairs.append((data[bin][bin_index - 1], state))
                            if pred == DISTINCT_CLASS:
                                model.append(state)
                                not_detected_near_duplicate_pairs.append((data[bin][bin_index - 1], state))
                            bin_index += 1

                if bin_covered:
                    covered_bins.append(bin)

        unique_states_in_model = len(covered_bins)
        precision = unique_states_in_model / len(model) if len(model) > 0 else 0
        recall = len(covered_bins) / number_of_bins if number_of_bins > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        print(f"App: {app}, Baseline: {baseline}, Feature: {feature}")
        print(f"Covered bins: {len(covered_bins)}")
        print(f"Number of bins: {number_of_bins}")
        print(f"Total number of states: {total_number_of_states}")
        print(f"Number of states in model: {len(model)}")
        print(f"Unique states in model: {unique_states_in_model}")
        print(f"Number of State-Pairs not detected as near-duplicates: {len(not_detected_near_duplicate_pairs)}")
        # print(f"State-Pairs not detected as near-duplicates: {not_detected_near_duplicate_pairs}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")

        if OUTPUT_CSV:
            with open(filename, 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow([setting, app, baseline, feature, f1_score, precision, recall])


if __name__ == "__main__":
    fraggen_model_quality()
