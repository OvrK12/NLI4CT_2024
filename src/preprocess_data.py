import os
from datasets import load_dataset

SAVE_LOCATION = "./data/"
DATASET_LOCATION = "bigbio/sem_eval_2024_task_2"
DATASET_SOURCE = "sem_eval_2024_task_2_source"
CT_SOURCE = "sem_eval_2024_task_2_ct"

def load_and_split_dataset():
    """Loads and returns the dataset (train, validation and test split)

    Returns:
        train, val and test split of the dataset
    """
    dataset = load_dataset(DATASET_LOCATION, name=DATASET_SOURCE)
    train_data, val_data, test_data = dataset['train'], dataset['validation'], dataset['test']
    return train_data, val_data, test_data

def load_clinical_trial_reports():
    """Loads the clinical trial reports from the dataset and returns it as a dictionary for better access time

    Returns:
        The clinical trial reports as a dict mapping CT_Id to the sections
        (i.e. "eligibility criteria", "intervention", "results" and "adverse events") of the respective report
    """
    ct_reports = load_dataset(DATASET_LOCATION, name=CT_SOURCE)['train']
    # we use a dict instead of a dataset object for better access time
    ct_reports_dict = {row['clinical_trial_id']: dict(row) for row in ct_reports}
    return ct_reports_dict

def create_statement_column(row, ct_reports_dict):
    """Creates a new "statement" column based on the input row and the corresponding CT report.
    This statement column will be the input to our model from which it predicts the label.

    Args:
        row: input row. Based on the existing columns the new statement column is formed
        ct_reports_dict: CT reports from which the evidence is extracted to form the statement

    Returns:
        row with the additional statement column
    """ 
    # "Adverse Events" column from ct dataset does not match the column in the HF dataset -> do preprocessing
    section_id = row['section_id'].lower()
    section_id = "adverse_events" if section_id == "adverse events" else section_id

    # create the evidence from primary_id and secondary_id of the CT report
    evidence = "Primary trial evidence: "
    primary_report = ct_reports_dict[row['primary_id']] 
    evidence += "".join(primary_report[section_id])

    if row['type'] == "Comparison":
        evidence += " Secondary trial evidence: "
        secondary_report = ct_reports_dict.get(row['secondary_id'])
        if secondary_report:
            evidence += "".join(secondary_report[section_id])

    # final format for the statement column
    row['statement'] = f"{evidence} \n Question: Does this imply that {row['statement']}? Entailment or Contradiction?"
    return row

def preprocess_dataset(dataset, ct_reports_dict):
    """Preprocess a dataset so it can be used as input for the HF models

    Args:
        dataset: the dataset to be preprocessed
        ct_reports_dict: CT reports that are used for preprocessing

    Returns:
        The preprocessed dataset
    """
    # create a statement column from the existing the columns which is used as the input for the HF model
    dataset = dataset.map(lambda row: create_statement_column(row, ct_reports_dict))
    # drop all columns that are not needed for the model. Only input is the statement column which predicts the label
    #dataset = dataset.remove_columns(["type", "section_id", "primary_id", "secondary_id", "id"])
    # HF expects a "text" column as input -> rename statement column
    dataset = dataset.rename_column("statement", "text")
    return dataset

def save_dataset_to_json(dataset, save_location, filename):
    """Saves the preprocessed dataset to a json file so it can be easily used as input for the HF models

    Args:
        dataset: the dataset to be saved
        save_location: save location (folder) of the dataset
        filename: file name under which the dataset is saved
    """
    dataset.to_json(os.path.join(save_location, filename))

def main():
    print(os.path.join(SAVE_LOCATION, "test"))
    train_data, val_data, test_data = load_and_split_dataset()
    ct_reports_dict = load_clinical_trial_reports()

    preprocessed_train_set = preprocess_dataset(train_data, ct_reports_dict)
    preprocessed_val_set = preprocess_dataset(val_data, ct_reports_dict)
    preprocessed_test_set = preprocess_dataset(test_data, ct_reports_dict)

    save_dataset_to_json(preprocessed_train_set, SAVE_LOCATION, "train.json")
    save_dataset_to_json(preprocessed_val_set, SAVE_LOCATION, "val.json")
    save_dataset_to_json(preprocessed_test_set, SAVE_LOCATION, "test.json")

if __name__ == "__main__":
    main()
