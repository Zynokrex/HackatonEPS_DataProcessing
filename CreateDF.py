#%%
import pandas as pd
import csv
import re
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
#%%

descr_list=[]

#open first description file and save each string dictionary on the list
with open('save_descriptions.csv', mode="r") as csvfile:
    reader = csv.reader(csvfile, delimiter=";")
    for entry in reader:
        descr_list.append(entry)
csvfile.close()

#%%
#open the second description file and do the same as before
descr_list2=[]
with open('save_descriptions_2.csv', mode="r") as csvfile2:
    reader = csv.reader(csvfile2, delimiter=";")
    for entry in reader:
        descr_list2.append(entry)
csvfile2.close()


# %%
#iterate over each description list to obtain the labels and the score using regex expresions
#as it reads the dictionaries as strings
descr_dict = {}
for i in range(len(descr_list[0])):
    labels = re.findall(r"label':\s*'([^']+)'", descr_list[0][i])
    score = re.findall(r"'score':\s*([0-9]+(?:\.[0-9]+)?)", descr_list[0][i])
    if len(labels)==0 or len(score)==0:
        continue
    descr_dict[i] = {'label' : labels, 'score':float(score[0])}

descr_dict2 = {}
for i in range(len(descr_list2[0])):
    labels = re.findall(r"label':\s*'([^']+)'", descr_list2[0][i])
    score = re.findall(r"'score':\s*([0-9]+(?:\.[0-9]+)?)", descr_list2[0][i])
    if len(labels)==0 or len(score)==0:
        continue
    descr_dict2[i] = {'label' : labels, 'score':float(score[0])}
    
# %%
#concatenate both dataframes with all the labels and scores
labeled_df = pd.DataFrame.from_dict(descr_dict).T
labeled_df2 = pd.DataFrame.from_dict(descr_dict2).T
labeled_df = pd.concat([labeled_df, labeled_df2])

# %%
#obtain the number of labels that each row has
labeled_df['num_labels'] = labeled_df.label.apply(len)

# %%
#One-hot encoding of the labels
mlb = MultiLabelBinarizer(sparse_output=True)
labeled_df = labeled_df.join(pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(labeled_df.pop('label')),
                                                                index=labeled_df.index,
                                                                columns=mlb.classes_))
# %%
#Calculate the percentage that each label represents on the dataframe
inform_df = pd.DataFrame((labeled_df.sum()/(labeled_df.sum().num_labels)) * 100).T.drop(columns=['score','num_labels'])

# %%
#saving the procesed data for its posterior use
inform_df.to_csv('pcntg_labels.csv')
labeled_df.to_csv('Processed_dataset.csv')
