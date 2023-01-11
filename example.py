import multiprocessing
from datalabs import load_dataset
from datalabs.operations.aggregate.text_classification import get_features_dataset_level


def run_example(dataset_name: str, sub_dataset: str, feature_func):

    dataset = load_dataset(dataset_name, sub_dataset)
    all_splits = list(dataset["train"]._info.splits.keys())

    for split_name in all_splits:
        dataset[split_name] = dataset[split_name].apply(
            feature_func,
            num_proc=multiprocessing.cpu_count(),
            mode="local",
        )
        # print(dataset[split_name])

    dataset = load_dataset(dataset_name, sub_dataset)
    for split_name in all_splits:
        dataset[split_name] = dataset[split_name].apply(
            get_features_dataset_level, mode="local", prefix="avg" + "_" + split_name
        )

    return dataset


# Example1: single document
# dataset_name = "cnn_dailymail"
# sub_dataset = "3.0.0"
# from get_summ_features import summarization_multi_document
# feature_func = summarization_single_document
# dataset = run_example(dataset_name, sub_dataset, feature_func)

# Sample-level features: dataset["train"][0]
# print(dataset["train"][0])
# # Dataset-level features: dataset["train"]._stat
# print(dataset["train"]._stat)
# print(dataset["validation"]._stat)
# print(dataset["test"]._stat)


# Example2: multiple documents
# dataset_name = "cochrane"
# sub_dataset = "cochrane"
# feature_func = summarization_multi_document
# run_example(dataset_name, sub_dataset, feature_func)
