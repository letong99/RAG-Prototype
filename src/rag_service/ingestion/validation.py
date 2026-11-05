"""Source-specific missing-value checks before chunking or embedding."""

BIOASQ_DATASET_ID = "rag-datasets/rag-mini-bioasq"


def missing_passage(value: object) -> bool:
    """BioASQ exports contain string sentinels as well as actual null values."""
    return not isinstance(value, str) or value.strip().casefold() in {"", "nan", "none", "null"}
