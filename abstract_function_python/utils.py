from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from bs4 import BeautifulSoup, Comment
from bs4.element import NavigableString
import gensim
import json
import re
# import main

# File to define utils for abstract function python

def process_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    corpus = ([], [], [])
    retrieve_abstraction_from_html(soup, corpus)
    return corpus


def retrieve_abstraction_from_html(bs, corpus):
    try:
        if type(bs) == NavigableString:
            tokens = gensim.utils.simple_preprocess(bs.string)
            if len(tokens) > 0:
                corpus[0].extend(tokens)
                corpus[2].extend(tokens)
            return

        bs_has_name = bs.name != None
        bs_is_single_tag = str(bs)[-2:] == '/>'

        if bs_has_name and not bs_is_single_tag:
            corpus[1].append(f'<{bs.name}>')
            corpus[2].append(f'<{bs.name}>')
        elif bs_has_name and bs_is_single_tag:
            corpus[1].append(f'<{bs.name}/>')
            corpus[2].append(f'<{bs.name}/>')
        try:
            for c in bs.children:
                if type(c) == Comment:
                    continue
                retrieve_abstraction_from_html(c, corpus)
        except Exception:
            pass
        if bs_has_name and not bs_is_single_tag:
            corpus[1].append(f'</{bs.name}>')
            corpus[2].append(f'</{bs.name}>')
    except Exception as e:
        print('html structure content error', e)
        pass

# Preprocesses the input states for inference by tokenizing them using the provided tokenizer.
def preprocess_for_inference(state1, state2, tokenizer):
    trimmed_state1, trimmed_state2 = trim_common_html(state1, state2)
    tokenized_inputs = tokenizer(trimmed_state1, trimmed_state2,
                                 padding='max_length',
                                 truncation='longest_first',
                                 max_length=512,
                                 return_tensors='pt')
    return tokenized_inputs

# Runs inference on the provided inputs using the provided model.
def get_prediction(model, inputs):
    with torch.no_grad():  # disable gradient computation
        outputs = model(**inputs)
    # extract logits and apply softmax
    probabilities = torch.softmax(outputs.logits, dim=-1)
    # predict the class with the highest probability
    predicted_class_id = probabilities.argmax(dim=-1).item()
    return predicted_class_id

def bert_equals(dom1, dom2, model, tokenizer, feature='content_tags'):
    """
    1. Extract tag, content, or tags + content
    2. Tokenize the extracted data
    3. Predict the similarity between the two states using a fine-tuned BERT model
    :return: The predicted class (0 for distinct, 1 for clone/near-duplicate).
    """
    corpus1 = process_html(dom1)
    corpus2 = process_html(dom2)

    print(f"Logging 2 states: to compare: \ndom1{dom1}\ndom2{dom2}") #TODO: remove

    if feature.endswith('content_tags'):
        data1 = corpus1[2]
        data2 = corpus2[2]
    elif feature.endswith('tags'):
        data1 = corpus1[1]
        data2 = corpus2[1]
    elif feature.endswith('content'):
        data1 = corpus1[0]
        data2 = corpus2[0]
    else:
        raise ValueError(f'Invalid feature type: {feature}') #TODO: handling of distance-all

    processed_inputs = preprocess_for_inference(data1, data2, tokenizer) #TODO: check if data1 and data2 are strings => for trimming
    predicted = get_prediction(model, processed_inputs)

    # print(main.hf_model_name, "=====================================")
    # print(data1)
    # print("compared to:")
    # print(data2)
    print(f'{main.hf_model_name} | Predicted class: {"near-dup/clone" if predicted == 1 else "distinct"}')
    # print(main.hf_model_name, "=====================================")

    # returns 0 for distinct, 1 for clone/near-duplicate
    return predicted

# Fix JSON strings incoming from the crawler
def fix_json(json_string):
    fixed_json = []
    in_string = False
    ix = 0
    char = ''

    fixed_json_string = json_string

    for ix, char in enumerate(json_string):
        if char == '"' and in_string:
            if json_string[ix+1] == ',' or json_string[ix+1] == '}' or json_string[ix+1:].replace(" ", "").startswith('}') or json_string[ix+1:].replace(" ", "").startswith('\n'):
                in_string = False
            else:
                fixed_json.append("'")
                continue
        elif not in_string and char == '"':
            if len(fixed_json) > 2 and (fixed_json[-1] == ':' or fixed_json[-2] == ':'):
                in_string = True
        if char == '\\':
            fixed_json.append('\\')
        fixed_json.append(char)
        if not in_string and char == '}':
            break

    fixed_json_string = ''.join(fixed_json)
    fixed_json_string = fixed_json_string.replace('\n', '')
    fixed_json_string = sanitize_json_string(fixed_json_string)

    try:
        json.loads(fixed_json_string)
        return fixed_json_string
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e},\n fixed JSON: {fixed_json_string}")
        return "Error decoding JSON"


# Sanitize JSON strings by removing non-printable ASCII characters, TODO: check if this destroys the further processing somehow
def sanitize_json_string(json_str):
    # Define a regex pattern for valid printable ASCII characters (excluding control characters)
    printable_ascii_pattern = re.compile(r'[\x20-\x7E]')

    # Remove all characters that do not match the printable ASCII pattern
    sanitized_str = ''.join(char if printable_ascii_pattern.match(char) else '' for char in json_str)

    return sanitized_str


# Trimming the common leading and trailing parts from two HTML page representations (content_tags, tags, content)
def ensure_format(s):
    if s.startswith(","): s = s[1:]
    if not s.startswith('[\"'):
        if not s.startswith('\"'): s = '[\"' + s
        else: s = "[" + s

    if not s.endswith('\"]'):
        if not s.endswith('\"'): s = s + '\"]'
        else: s = s + ']'

    return s

def trim_common_html(state1:str, state2:str) -> tuple[str, str]:
    """
    Trims the common leading and trailing parts from two HTML page representations (content_tags, tags, content).

    :param state1: HTML content of the first page as a string.
    :param state2: HTML content of the second page as a string.
    :return: A tuple of the trimmed HTML contents.
    """
    leading_common_length = 0
    for x, y in zip(state1, state2):
        if x == y:
            leading_common_length += 1
        else:
            break
    trailing_common_length = 0
    for x, y in zip(reversed(state1[leading_common_length:]), reversed(state2[leading_common_length:])):
        if x == y:
            trailing_common_length += 1
        else:
            break
    trimmed_state1 = str(state1[leading_common_length: len(state1) - trailing_common_length])
    trimmed_state2 = str(state2[leading_common_length: len(state2) - trailing_common_length])
    if trimmed_state1.startswith('\",'): trimmed_state1 = trimmed_state1[3:]
    if trimmed_state2.startswith('\",'): trimmed_state2 = trimmed_state2[3:]
    if trimmed_state1.endswith('\"'): trimmed_state1 = trimmed_state1[:-3]
    if trimmed_state2.endswith('\"'): trimmed_state2 = trimmed_state2[:-3]

    # if one page is subset of the other page
    if trimmed_state2 == "" or trimmed_state1 == "":
      # return state1, state2
      return 'exact duplicate', 'exact duplicate'

    # trimmed_states = [trimmed_state1, trimmed_state2]
    # corrected_states = [ensure_format(t) for t in trimmed_states]

    # trimmed_state1, trimmed_state2 = corrected_states

    return trimmed_state1, trimmed_state2


def trim_content_to_body(file_content:list, representation='content_tags') -> list:
    """
    Trims the content of HTML represented as a list of tags to only include content within the <body> tag.

    :param file_content: **List** of strings representing the entire content.
    :return: **List** of strings representing the content within the <body> tag.
    """
    if representation not in ['content_tags']: #, 'content', 'tags'
        print(f'Not yet implemented: {representation}')
        return []

    start_index, end_index = 0, 0

    if representation == 'content_tags':
        # Find the indices of the <body> and </body> tags
        start_index = file_content.index("<body>") + 1
        end_index = file_content.index("</body>")
    # elif representation == 'content': TODO
    # elif representation == 'tags': TODO

    # Extract the content within the <body> tags
    trimmed_file_content = file_content[start_index:end_index]

    return trimmed_file_content
