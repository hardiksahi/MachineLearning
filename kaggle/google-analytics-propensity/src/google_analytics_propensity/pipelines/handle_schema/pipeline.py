"""
This is a boilerplate pipeline 'handle_schema'
generated using Kedro 0.18.12
"""
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import impose_schema, clean_data


def template_pipeline() -> Pipeline:
    node1 = node(
        func=impose_schema,
        inputs=["intermediate_data", "params:schema_impose"],
        outputs="schema_imposed_data",
        name="impose_schema",
    )
    node2 = node(
        func=clean_data,
        inputs=["schema_imposed_data", "params:handle_columns"],
        outputs="primary_data",
        name="clean_data",
    )
    return pipeline([node1, node2])


def create_pipeline(**kwargs) -> Pipeline:
    ns_pipeline = pipeline(
        pipe=template_pipeline(),
        parameters={"params:schema_impose", "params:handle_columns"},
        namespace="train",
    )
    return ns_pipeline
