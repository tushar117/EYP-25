Please Ensure the following before running the scripts:
1) Update the peft_checkpoint_path variable in the generate.sh script before running the inference (generate.sh)
2) Edit the num_process to no. of available GPU's before running the finetune.sh
4) Make sure the create_template is same as that used for training
3) Make sure to change the prompt for instruct tuning according to the level of enhancement. You can do so by changing the create_template function inside the finetuning_phi.py file as follows

Prompt for Level - 1:
def create_template(query, enhanced_query):
    messages = []
    # adding instructions
    messages.append({"role": "user", "content": "Can you enhance some queries that I will give you. Please think about more relevant details to improve this query and make it more structured. Please do not answer the query, just give back the enhanced query."})
    # adding bot response
    messages.append({"role": "assistant", "content": "sure, I will not answer the queries. I will just rephrase the query you provide and return the enhanced version of your query."})
    # adding examples:
    messages.append({"role": "user", "content": "##query##: exercise plan"})
    messages.append({"role": "assistant", "content": "Generate an exercise plan for beginners to lose weight and gain muscle, with examples of exercises, duration, frequency, intensity, and tips on nutrition and recovery."})
    
    messages.append({"role": "user", "content": "##query##: how to improve sleep"})
    messages.append({"role": "assistant", "content": "I have trouble sleeping and want to improve my sleep quality. What habits or routines can help me sleep better?"})
    # user prompts
    messages.append({"role": "user", "content": f"##query##: {query}"})
    messages.append({"role": "assistant", "content": f"{enhanced_query}"})
    return messages

Prompt for Level - 2:
def create_template(query, enhanced_query):
    messages = []
    # adding instructions
    messages.append({"role": "user", "content": "Can you enhance some queries that I will give you. Please think about more relevant details to improve this query and make it more structured. Please do not answer the query, just give back the enhanced query."})
    # adding bot response
    messages.append({"role": "assistant", "content": "sure, I will not answer the queries. I will just rephrase the query you provide and return the enhanced version of your query."})
    # adding examples:
    messages.append({"role": "user", "content": "##query##: exercise plan"})
    messages.append({"role": "assistant", "content": "Generate an exercise plan for <current level> to lose weight and gain muscle, with examples of exercises, duration, frequency, intensity, and tips on nutrition and recovery."})
    
    messages.append({"role": "user", "content": "##query##: how to improve sleep"})
    messages.append({"role": "assistant", "content": "I have trouble sleeping and want to improve my sleep quality. What habits or routines can help me sleep better?"})
    # user prompts
    messages.append({"role": "user", "content": f"##query##: {query}"})
    messages.append({"role": "assistant", "content": f"{enhanced_query}"})
    return messages

Prompt for Level - 3:
def create_template(query, enhanced_query):
    messages = []
    # adding instructions
    messages.append({"role": "user", "content": "Can you enhance some queries that I will give you. Please think about more relevant details to improve this query and make it more structured. Please do not answer the query, just give back the enhanced query."})
    # adding bot response
    messages.append({"role": "assistant", "content": "sure, I will not answer the queries. I will just rephrase the query you provide and return the enhanced version of your query."})
    # adding examples:
    messages.append({"role": "user", "content": "##query##: exercise plan"})
    messages.append({"role": "assistant", "content": "Generate an exercise plan for <beginners> to <lose weight>, with examples of exercises, duration, frequency, intensity, and tips on nutrition and recovery."})
    
    messages.append({"role": "user", "content": "##query##: how to improve sleep"})
    messages.append({"role": "assistant", "content": "I have trouble sleeping and want to improve my sleep quality. What habits or routines can help me sleep better?"})
    # user prompts
    messages.append({"role": "user", "content": f"##query##: {query}"})
    messages.append({"role": "assistant", "content": f"{enhanced_query}"})
    return messages

Prompt for Level - 4:
def create_template(query, enhanced_query):
    messages = []
    # adding instructions
    messages.append({"role": "user", "content": "Can you enhance some queries that I will give you. Please think about more relevant details to improve this query and make it more structured. Please do not answer the query, just give back the enhanced query."})
    # adding bot response
    messages.append({"role": "assistant", "content": "sure, I will not answer the queries. I will just rephrase the query you provide and return the enhanced version of your query."})
    # adding examples:
    messages.append({"role": "user", "content": "##query##: exercise plan"})
    messages.append({"role": "assistant", "content": "Generate an exercise plan for <beginners/intermediate/experts> to <lose weight/gain muscle/increase stamina>, with examples of exercises, duration, frequency, intensity, and tips on nutrition and recovery."})
    
    messages.append({"role": "user", "content": "##query##: how to improve sleep"})
    messages.append({"role": "assistant", "content": "I have trouble sleeping and want to improve my sleep quality. What habits or routines can help me sleep better?"})
    # user prompts
    messages.append({"role": "user", "content": f"##query##: {query}"})
    messages.append({"role": "assistant", "content": f"{enhanced_query}"})
    return messages
