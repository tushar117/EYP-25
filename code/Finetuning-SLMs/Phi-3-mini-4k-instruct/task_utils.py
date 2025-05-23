import re
import json
import sys

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub('\s+', ' ', text)
    return text

def truncate_string(input_string, max_words):
    max_avg_chars = 5*max_words # space + 4 char word = 200 chars
    input_words = input_string.split(' ')[-max_words:]
    reconstructed_input = " ".join(input_words)
    return reconstructed_input[-max_avg_chars:]

def prepare_string(prefix, in_context, in_suggestion_chip, in_completion, max_word=3):
    # prepare context
    context = [normalize_text(tz) for tz in json.loads(in_context)]
    context = " | ".join(context)
    context = truncate_string(context, 40)
    # prepare suggestion chip
    suggestion_chip = [normalize_text(tz) for tz in json.loads(in_suggestion_chip)]
    suggestion_chip = " | ".join(suggestion_chip)
    suggestion_chip = truncate_string(suggestion_chip, 40)
    # prepare prefix
    prefix = truncate_string(prefix.lower(), 40)
    # complete max 3 words in output
    y_words = in_completion.split(' ')
    y_next_words = " ".join(y_words[:max_word])
    prefix_str = f"### Related Searches: {suggestion_chip}, ### Previous Query: {context}, ### Prefix: {prefix} ### Completion: {y_next_words}"
    return prefix_str

def combine_prefix_suffix(prefix, suffix):
    if prefix[-1]==" ":
        return prefix+suffix
    idx = prefix.rfind(" ")
    if idx==-1:
        return suffix
    return prefix[:idx+1]+suffix

def prepare_stage1_string(query):
    return query.lower().strip()

def is_empty(input_string):
    input_string = input_string.strip()
    return input_string=="" or len(input_string)==0
