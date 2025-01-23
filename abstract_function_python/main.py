import json
import re
import pickle
import os
import utils
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
import pandas as pd
from flask import Flask, request

no_of_inferences = 0

hf_model_name = "lgk03/NDD-claroline_test-content" # this gets overwritten by the env variable in load_model_and_tokenizer

# counter for number of inferences
def increase_no_of_inferences():
    global no_of_inferences
    no_of_inferences += 1
    if no_of_inferences % 100 == 0:
        print(f"Number of inferences: {no_of_inferences}")

def load_model_and_tokenizer(feature=None, hf_model_name=None):
    if not feature:
        feature = os.getenv('FEATURE', 'content')
    else:
        feature = feature

    if not hf_model_name:
        hf_model_name = os.getenv('HF_MODEL_NAME', f'lgk03/NDD-addressbook_test-{feature}') # distilBERT-NDD.html.{feature}
    else:
        hf_model_name = hf_model_name

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_name)
    model.eval()  # set model into evaluation mode
    return model, tokenizer, feature

model, tokenizer, feature = load_model_and_tokenizer()

app = Flask(__name__)


# call to route /equals executes equalRoute function
# use URL, DOM String, Dom content and DOM syntax tree as params
@app.route('/equals', methods=('GET', 'POST'))
def equal_route():
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json' or content_type == 'application/json; utf-8':
        fixed_json = utils.fix_json(request.data.decode('utf-8'))
        if fixed_json == "Error decoding JSON":
            print("Exiting due to JSON error")
            exit(1)
        data = json.loads(fixed_json)
    else:
        return 'Content-Type not supported!'

    # get params sent by java
    parametersJava = data

    obj1 = parametersJava['dom1']
    obj2 = parametersJava['dom2']

    # compute equality of DOM objects
    result = utils.bert_equals(obj1, obj2, model, tokenizer, feature)

    result = "true" if result == 1 else "false"

    increase_no_of_inferences()
    # return true if the two objects are clones/near-duplicates => comment was here before
    return result


if __name__ == "__main__":
    print(f"******* We are using the model: {hf_model_name} *******")
    app.run(debug=False)
