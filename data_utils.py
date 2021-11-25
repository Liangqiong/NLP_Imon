# model.load_state_dict(torch.load(model_path, map_location=device))
import pandas as pd
import random


test_client_names = ['stanford', 'imon']
df = pd.DataFrame(columns=['model_type', 'stanford_data', 'imon_data', ])
df.set_index('model_type', inplace=True)

# just for test

client_names = ['imon', 'stanford']
tmp = {}
len = 0
for model_name in client_names:
    ff = [model_name + '_model']
    for name in ['stanford', 'imon']:
        f1 = random.random()
        print(f1)
        ff.append(f1)
    df.loc[len] = ff
    len += 1



## extract the df


critical = pd.read_csv('/media/veracrypt1/data/NLP/stanford_data.csv')

critical = critical.fillna('N/A')
stanford_df = complex_split(critical)

# get the Imon site
critical = pd.read_csv('/media/veracrypt1/data/NLP/ASU_data.csv')
critical = critical.fillna('N/A')
imon_df = complex_split(critical)

stanford_df = data_split(stanford_df)
imon_df = data_split(imon_df)

stanford_df.to_pickle('/home/liangqiong/Research/Deep_Learning/Pytorch/Data_Distribution/NLP/BERTCriticalFinding-main/src/data/stanford_clinical_df.pkl')
imon_df.to_pickle('/home/liangqiong/Research/Deep_Learning/Pytorch/Data_Distribution/NLP/BERTCriticalFinding-main/src/data/imon_clinical_df.pkl')




