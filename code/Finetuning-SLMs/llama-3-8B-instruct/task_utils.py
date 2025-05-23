import re

def normalize_text(text):
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    return text

def truncate_string(input_string, max_words):
    max_avg_chars = 5*max_words # space + 4 char word = 5 chars
    input_words = input_string.split(' ')[-max_words:]
    reconstructed_input = " ".join(input_words)
    return reconstructed_input[-max_avg_chars:]

def get_level_example(level):
    messages = []
    messages.append({"role": "user", "content": "##query##: swiss vacation plan"})
    if level==1:
        messages.append({"role": "system", "content": "Can you create a vacation plan for Switzerland?"})
    elif level==2:
        messages.append({"role": "system", "content": "Can you create a vacation plan for Switzerland for <Number of People> at a budget of <Budget>"})
    elif level==3:
        messages.append({"role": "system", "content": "Can you create a vacation plan for Switzerland for <2 people> at a budget of <5000 Euros>"})
    else:
        messages.append({"role": "system", "content": "Can you create a vacation plan for Switzerland for <2 people/5 people/10 people> at a budget of <5000 Euros/200000 INR/5000 USD>"})
    return messages

def create_template(query, enhanced_query, level):
    messages = []
    # adding instructions
    messages.append({"role": "system", "content": "You are an helpful AI assistant that enhances user queries submitted to conversational LLM-based chat bot system like ChatGPT or Microsoft Copilot. Please think about more relevant details to improve this query and make it more structured. Please do not answer the query, just give back the enhanced query."})
    # adding examples:
    messages.extend(get_level_example(level))
    # adding current user query
    messages.append({"role": "user", "content": f"##query##: {query}"})
    messages.append({"role": "system", "content": f"{enhanced_query}"})
    return messages

def process_text(query, enhanced_query, level, max_length, tokenizer):
    # prepare context
    query = truncate_string(normalize_text(query), 40)
    enhanced_query = truncate_string(normalize_text(enhanced_query), 80)
    text_str = tokenizer.apply_chat_template(create_template(query, enhanced_query, level), tokenize=False, add_generation_prompt=False)
    trimmed_text_str = text_str
    input_tokens = tokenizer.encode(trimmed_text_str)
    if len(input_tokens)>max_length:
        trimmed_text_str = tokenizer.decode(input_tokens[-max_length:])
    return trimmed_text_str