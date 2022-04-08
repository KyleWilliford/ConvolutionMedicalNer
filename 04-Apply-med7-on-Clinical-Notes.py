import os
import pandas as pd
import spacy
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
spacy.require_gpu()
med7 = spacy.load("en_core_med7_lg")

preprocessed_df = pd.read_pickle("data/preprocessed_notes.p")

preprocessed_df['ner'] = None

def run_one_row(input):
    ind, text = input
    all_pred = []
    for each_sent in text:
        try:
            doc = med7(each_sent)
            result = ([(ent.text, ent.label_) for ent in doc.ents])
            if len(result) == 0: continue
            all_pred.append(result)
        except:
            print("error..")
            continue
    preprocessed_df.at[ind, 'ner'] = all_pred


count = 0
preprocessed_index = {}
inputs = []
for i in preprocessed_df.itertuples():
    count += 1
    ind = i.Index
    text = i.preprocessed_text
    inputs.append((ind, text))

with tqdm(range(len(inputs)), desc="process") as pbar:
    with ThreadPoolExecutor(max_workers=12) as exe:
        for _ in  exe.map(run_one_row, inputs):
            pbar.update()

pd.to_pickle(preprocessed_df, "data/ner_df.p")