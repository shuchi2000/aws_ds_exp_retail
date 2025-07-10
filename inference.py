import os
import json
import joblib
import pandas as pd
import io

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "polynomial_regression_model.joblib")) # Matches local_model_filename
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        return pd.read_csv(io.StringIO(request_body))
    elif request_content_type == "application/json":
        data = json.loads(request_body)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Content type {request_content_type} not supported.")

def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    return prediction

def output_fn(prediction, accept_content_type):
    if accept_content_type == "application/json":
        return json.dumps(prediction.tolist()), accept_content_type
    else:
        raise ValueError(f"Accept content type {accept_content_type} not supported.")