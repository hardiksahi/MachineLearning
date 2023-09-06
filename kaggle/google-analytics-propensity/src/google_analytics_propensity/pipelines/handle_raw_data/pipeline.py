"""
This is a boilerplate pipeline 'handle_raw_data'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import normalize_dataset


def template_pipeline(**kwargs) -> Pipeline:
    node1 = node(
        func=normalize_dataset,
        inputs=["raw_data", "params:raw_data"],
        outputs="intermediate_data",
        name="normalize_dataset",
    )
    return pipeline([node1])


def create_pipeline(**kwargs) -> Pipeline:
    ns_pipeline = pipeline(
        pipe=template_pipeline(), parameters={"params:raw_data"}, namespace="train"
    )
    return ns_pipeline
