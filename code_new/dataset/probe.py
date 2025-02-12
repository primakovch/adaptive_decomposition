import random

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer
import json


def honesty_function_dataset(
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        user_tag: str = "",
        n_train: int = 512,
        assistant_tag: str = "",
        seed: int = 0,
):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df["label"] == 1][
        "statement"
    ].values.tolist()  # list of true statements
    false_statements = df[df["label"] == 0][
        "statement"
    ].values.tolist()  # list of false statements

    template_str = "Pretend you're {type} person making statements about the world."
    honest_statements = []
    untruthful_statements = []

    # Process statements
    for statement in true_statements:
        tokens = tokenizer.tokenize(statement)  # tokenize statement

        for idx in range(1, len(tokens) - 5):
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(
                truncated_tokens
            )  # convert tokens to string
            honest_statements.append(
                f"""{user_tag} {template_str.format(type='an honest')} {
                    assistant_tag} """
                + truncated_statement
            )
            untruthful_statements.append(
                f"""{user_tag} {template_str.format(type='an untruthful')} {
                    assistant_tag} """
                + truncated_statement
            )

    # Create training data
    ntrain = n_train
    combined_data = [
        [honest, untruthful]
        for honest, untruthful in zip(honest_statements, untruthful_statements)
    ]  # combine honest and untruthful statements
    train_data = combined_data[
        :ntrain
    ]  # split into train and test data

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append(
            [s == true_s for s in d]
        )

    train_data = np.concatenate(
        train_data
    ).tolist()

    # Create test data
    reshaped_data = np.array(
        [
            [honest, untruthful]
            for honest, untruthful in zip(
                honest_statements[:-1], untruthful_statements[1:]
            )
        ]
    ).flatten()
    eval_data = reshaped_data[
        ntrain: ntrain * 2
    ].tolist()

    test_data = reshaped_data[
        -300:-1
    ].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Eval data: {len(eval_data)}")
    print(f"Test data: {len(test_data)}")
    return {
        "train": {"data": train_data, "labels": train_labels},
        "eval": {"data": eval_data, "labels": [[1, 0]] * len(eval_data)},
        "test": {"data": test_data, "labels": [[1, 0]] * len(test_data)},
    }


# --------------------------------------------------------------------------------------------------------------------------------------


# Creates a dataset that will help extract the feature representing complexity of the claims 
# def complexity_function_datast(data_path: str, sample_size: int = 300 , num_class: int = 3, seed: int = 0, test = False):

#     with open(data_path, "r") as fp:
#         data = pd.DataFrame(json.load(fp))
#     fp.close()

#     random.seed(seed)
#     dataset = {}

#     if test == False:

#         simple_claims = random.sample(data[data["taxonomy_label"] == "temporal"].claim.values.tolist(), sample_size//num_class)
#         intermediate_claims = random.sample(data[data["taxonomy_label"].isin(["statistical", "interval"])].claim.values.tolist(), sample_size//num_class)
#         complex_claims = random.sample(data[data["taxonomy_label"] == "comparison"].claim.values.tolist(), sample_size//num_class)

#         dataset["simple_claims"] = simple_claims
#         dataset["intermediate_claims"] = intermediate_claims
#         dataset["complex_claims"] = complex_claims
    
#     elif test == True:
#         dataset["claims"] = data.claim.values.tolist()
#         dataset["labels"] = []
#         for taxonomy in data.taxonomy_label:
#             if taxonomy == "temporal":
#                 dataset["labels"].append(0)
#             elif taxonomy == "statistical" or taxonomy == "interval":
#                 dataset["labels"].append(1)
#             else:
#                 dataset["labels"].append(2)

    

#     return dataset


def complexity_function_datast(data_path: str, sample_size: int = 300 , num_class: int = 2, seed: int = 0, test = False, has_label = False):

    with open(data_path, "r") as fp:
        data = pd.DataFrame(json.load(fp))
    fp.close()

    random.seed(seed)
    dataset = {}

    if test == False:

        simple_claims = random.sample(data[data["complexity"] == 0].claim.values.tolist(), sample_size//num_class)
        # intermediate_claims = random.sample(data[data["complexity"] == 1].claim.values.tolist(), sample_size//num_class)
        complex_claims = random.sample(data[data["complexity"] >= 2].claim.values.tolist(), sample_size//num_class)

        dataset["simple_claims"] = simple_claims
        # dataset["intermediate_claims"] = intermediate_claims
        dataset["complex_claims"] = complex_claims
    
    elif test == True:
        dataset["claims"] = data.claim.values.tolist()
        if has_label == True:
            dataset["labels"] = data.complexity.values.tolist()
    

    return dataset
    