import os
import json
import pandas as pd
from datasets import Dataset,DatasetDict, load_dataset
import random
#
import torch
def obtain_sepc_datasets(path,max_num = None):
    random.seed(42)
    data = torch.load(path)
    samples = []
    for i in data:
        s = i['text'].strip() + i['label']
        # print('s:',s)
        samples.append({'text': s})
    random.shuffle(samples)
    if max_num is not None:

        samples = samples[:max_num]
    df = pd.DataFrame(data=samples)
    df = df.astype('str')
    data_final = Dataset.from_pandas(df)
    return data_final

