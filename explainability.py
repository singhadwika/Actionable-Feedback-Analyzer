import shap
import torch

def explain_model(model, texts):
    # Use SHAP to explain predictions
    tokenizer = model.tokenizer
    inputs = tokenizer(texts.tolist(), return_tensors='pt', truncation=True, padding=True)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Initialize SHAP explainer and get SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(inputs)
    
    # Return explanations
    return shap_values
