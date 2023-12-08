import pandas as pd
import paths


def format_instruction(sample):
    output = \
        f"""### Instruction:
        The following question is in reference to NASA technical standard 5002A. {sample["OutputQuestion"]}

        ### Input:
        {sample['Input']}
    
        ### Response:
        """
    return output


def format_instruction_new(sample, keep_context=True):
    output = \
        f"""### Instruction:
        The following question is in reference to NASA technical standard 5002A. {sample["OutputQuestion"]} {sample["OptionA"]} {sample["OptionB"]} {sample["OptionC"]} 
        
        ### Input:
        """
    if keep_context:
        output += f"""
        {sample['Input']}
        """
    output += """
        ### Response:
        """
    return output

def get_sys_format_question(prompt):
    return """
    <s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
        <</SYS>>
    """ + prompt + "[/INST]"

def get_question(i, keep_context=True, use_old_data=False):
    if use_old_data:
        filepath = paths.HOME_PATH + "/Datasets/QASeed.csv"
        df = pd.read_csv(filepath, encoding="ISO-8859-1")
        return format_instruction(df.iloc[i])
    else:
        filepath = paths.HOME_PATH + "/Datasets/eval_dataset 2.csv"
        df = pd.read_csv(filepath, encoding="ISO-8859-1")
        return format_instruction_new(df.iloc[i], keep_context=keep_context)


if __name__=="__main__":
    print(get_question(6, keep_context=False, use_old_data=False))