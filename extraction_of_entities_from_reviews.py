# This code uses the multi-processing technique 
# Loading required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pycontractions  import Contractions
from keybert import KeyBERT
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from multiprocessing import Pool, cpu_count
import os
import json

def ner(df_chunk):
    print(os.getpid())
    # Initializing the pre-trained Named Entity Recognition (NER) tokenzier & model
    tokenizer = AutoTokenizer.from_pretrained("Jorgeutd/bert-large-uncased-finetuned-ner")
    model = AutoModelForTokenClassification.from_pretrained("Jorgeutd/bert-large-uncased-finetuned-ner")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    results = []
    for i, text in df_chunk['expanded_clean_comments'].to_dict().items():
        results.append({
            'index': i,
            'entities': ner_pipeline(text)
        })
    print('end' + str(os.getpid()))
    return pd.DataFrame(results)

if __name__ == '__main__':
    df_reviews = pd.read_parquet('comments_with_cleaned.parquet')
    print('read dataframe')
    cpu = cpu_count() // 4
    # Divide the dataframe data across the processors
    chunks = np.array_split(df_reviews, cpu)
    # The Pool object offers a convenient means of parallelizing the execution of a function, distributing the input data across processes.
    with Pool(cpu) as pool:
        dfs = pool.map(ner, chunks)
        pd.concat(dfs).to_parquet('entities_with_index.parquet')
        