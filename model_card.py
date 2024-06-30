from dataclasses import dataclass

@dataclass
class ModelCard:
    """Class for specifying models"""
    model_name: str
    dataset: str
    dataset_lang: str