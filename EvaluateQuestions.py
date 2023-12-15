from datasets import load_dataset
import paths



def generate_eval_questions_from_dataset(dataset):
    # df = pd.read_csv(file_name, encoding = "ISO-8859-1")

    prompt_list = []
    answer_list = []
    for sample in dataset:
        prompt = f"""### Instruction:
        Use the Input below to answer this multiple choice question pertaining to NASA technical standard 5002A: {sample['Question']} {sample['OptionA']} {sample['OptionB']} {sample['OptionC']}

        ### Input:
        {sample['Input']}

        ### Response:
        """
        try:
            if "A" in sample["Response"]:
                answer = sample['OptionA']
            elif "B" in sample["Response"]:
                answer = sample['OptionB']
            else:
                answer = sample['OptionC']

            prompt_list.append(prompt)
            answer_list.append(answer)

        except:
            pass

    return prompt_list, answer_list

def generate_new_test_prompts():
    data_name = "anniedoris/nasa-tech-std-multiple-choice"
    dataset = load_dataset(data_name, split="train", token=paths.annie_read_token)
    prompt_list, answer_list = generate_eval_questions_from_dataset(dataset)

    prompt_list = prompt_list[0:10]
    answer_list = answer_list[0:10]
    return prompt_list, answer_list

if __name__=="__main__":
    pass
    #
    # for i in range(len(prompt_list)):
    #     # print(prompt_list[i])
    #     print(answer_list[i])
    #     print("="*100)