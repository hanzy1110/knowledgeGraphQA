#%%
from json import dump, load
import pandas as pd

df = pd.read_csv('final_dataset_clean_v2 .tsv', delimiter='\t')
keys = list(df.keys())
df.drop(keys[:2], axis=1, inplace=True)
_dict = df.to_dict()
data = {"data":_dict}

with open('dataset.json', 'w+') as outfile:
    dump(data, outfile)


# with open('dataset.json', "r", encoding="utf-8") as reader:
#     input_data = load(reader)['data']
#     print(input_data)
# # %%
