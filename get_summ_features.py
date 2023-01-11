import numpy

from datalabs.operations.featurize import nlp_featurizing
from datalabs.operations.featurize.plugins.summarization.sum_attribute import (
    SUMAttribute,
)
from datalabs.operations.featurize.summarization import get_oracle_summary


def get_summary_features(text: str, summary: str):
    summary_attribute = SUMAttribute()
    attribute_info = summary_attribute.cal_attributes_each(text, summary)
    return {
        "density": attribute_info["attr_density"],
        "coverage": attribute_info["attr_coverage"],
        "compression": attribute_info["attr_compression"],
        "repetition": attribute_info["attr_repetition"],
        "novelty": attribute_info["attr_novelty"],
        "copy_len": attribute_info["attr_copy_len"],
    }


def basic_features(text: str):
    return {"length": len(text.split(" "))}


def get_oracle(source: str, reference: str):
    """
    oracle_info =
        {
        "source":src,
        "reference":ref,
        "oracle_summary":oracle,
        "oracle_labels":labels,
        "oracle_score":max_score
        }
    """

    sample = {
        "text": source,
        "summary": reference,
    }
    oracle_info = get_oracle_summary.func(sample)

    index_of_oracles = [i for i, e in enumerate(oracle_info["oracle_labels"]) if e != 0]
    oracle_position = numpy.mean(index_of_oracles)

    return {
        "oracle_position": oracle_position,
        "oracle_score": oracle_info["oracle_score"],
    }


@nlp_featurizing(name="summarization_single_document")
def summarization_single_document(sample: dict):

    text = sample["text"]
    summary = sample["summary"]

    res_info_general_all = {}

    res_info_general = basic_features(text)
    for k, v in res_info_general.items():
        res_info_general_all["text" + "_" + k] = v

    res_info_general = basic_features(summary)
    for k, v in res_info_general.items():
        res_info_general_all["summary" + "_" + k] = v

    # get task-dependent features
    task_dependent_features = get_summary_features(text, summary)
    # get oracle features
    # oracle_features = get_oracle(text, summary)

    # update the res_info_general_all
    res_info_general_all.update(task_dependent_features)
    # res_info_general_all.update(oracle_features)

    return res_info_general_all


@nlp_featurizing(name="summarization_multi_document")
def summarization_multi_document(sample: dict):

    texts = sample["texts"]
    summary = sample["summary"]
    text = " ".join(texts)

    res_info_general_all = {}

    res_info_general = basic_features(text)
    for k, v in res_info_general.items():
        res_info_general_all["texts" + "_" + k] = v

    res_info_general = basic_features(summary)
    for k, v in res_info_general.items():
        res_info_general_all["summary" + "_" + k] = v

    # get task-dependent features
    task_dependent_features = get_summary_features(text, summary)
    # get oracle features
    # oracle_features = get_oracle(text, summary)

    # update the res_info_general_all
    res_info_general_all.update(task_dependent_features)
    # res_info_general_all.update(oracle_features)

    return res_info_general_all
