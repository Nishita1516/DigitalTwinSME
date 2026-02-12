import shap
import torch
import numpy as np

def get_shap_explainer(model, background_data):
    background = torch.tensor(
        background_data[:50],
        dtype=torch.float32
    )
    explainer = shap.DeepExplainer(model, background)
    return explainer


def explain_prediction(explainer, sample):
    sample = torch.tensor(
        np.expand_dims(sample, axis=0),
        dtype=torch.float32
    )
    shap_values = explainer.shap_values(sample, check_additivity=False)
    return shap_values
