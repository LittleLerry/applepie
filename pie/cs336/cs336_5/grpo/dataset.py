from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json

def extract_answer_from_gsm8k(answer_string):
    sp = "####"
    last_index = answer_string.rfind(sp)
    if last_index == -1:
        raise ValueError("Bad answer:" + answer_string)
    return answer_string[last_index + len(sp):].strip()

def replace_target_in_file(question_string, replacement):
    return question_string.replace('{question}', replacement)

def get_dataset(path_to_dataset: str, path_to_template: str):
    with open(path_to_dataset, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    with open(path_to_template, 'r', encoding='utf-8') as f:
        prompt = f.read()

    df = pd.DataFrame(data)
    return df['question'].apply(lambda s: replace_target_in_file(prompt,s)).tolist(), df['answer'].apply(lambda s: extract_answer_from_gsm8k(s)).tolist()

class ListDataset(Dataset):
    def __init__(self, input_dataset, input_prompt):
        self.q, self.a = get_dataset(input_dataset, input_prompt)
        assert len(self.q) == len(self.a)
    
    def __len__(self):
        return len(self.q)
    
    def __getitem__(self, idx):
        return (self.q[idx], self.a[idx])
    
    @property
    def data(self):
        return (self.q, self.a)

def collate_strings(batch):
    questions = [item[0] for item in batch]
    answers = [item[1] for item in batch]
    return questions, answers