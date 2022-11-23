# This code uses the multi-processing technique 
# Loading required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT
from multiprocessing import Pool, cpu_count
import os


def sentences_associated_entities(sentences, entity):
    for i, sentence in enumerate(sentences):
        if entity['start'] >= sentence['start'] and entity['start'] <= sentence['end']:
            return [x['sentence'] for x in sentences[i:i+2]]
    return []

def get_sentences(row):
    # Split sentences
    sentences = row['sentences']

    # NER
    entities = row['entities']

    # Extracting Person Entities and associated key phrases
    ner_person = list(filter(lambda x: x['entity_group'] == 'PER', entities))
    sentences_person = []
    for ner in ner_person:
        # Fetching sentences associated with this entity
        sentences_person += sentences_associated_entities(sentences, ner)
    sentences_person = ' '.join(list(set(sentences_person)))


    # Extracting Location Entities and associated key phrases
    ner_location = list(filter(lambda x: x['entity_group'] == 'LOC', entities))
    sentences_location = []
    for ner in ner_location:
        # Fetching sentences associated with this entity
        sentences_location += sentences_associated_entities(sentences, ner)
    sentences_location = ' '.join(list(set(sentences_location)))

    return {
        'sentences_person': sentences_person,
        'sentences_location': sentences_location
    }


def extract_keyphrase(chunk):
    print('start ' + str(os.getpid()))

    kw_model = KeyBERT()
    vectorizer = KeyphraseCountVectorizer(pos_pattern='<J.*>+(<CC><J.*>)?(<N.*>*|<V.*>*)?')

    sentences_person = chunk.map(lambda x: x['sentences_person']).tolist()
    sentences_location = chunk.map(lambda x: x['sentences_location']).tolist()

    results = []
    for i, index in enumerate(chunk.index):
        print(index)

        keyphrases_person = []
        if sentences_person[i]:
            try:
                keyphrases_person = kw_model.extract_keywords(docs=[sentences_person[i]], vectorizer=vectorizer)
                keyphrases_person = [a[0] for a in keyphrases_person]
            except Exception as ex:
                # print(ex)
                keyphrases_person = []

        keyphrases_location = []
        if sentences_location[i]:
            try:
                keyphrases_location = kw_model.extract_keywords(docs=[sentences_location[i]], vectorizer=vectorizer)
                keyphrases_location = [a[0] for a in keyphrases_location]
            except Exception as ex:
                # print(ex)
                keyphrases_location = []

        # Remove keyphrases from location which are in person too
        keyphrases_location = list(set(keyphrases_location) - set(keyphrases_person))

        results.append({
            'index': index,
            'sentences_person': sentences_person[i],
            'sentences_location': sentences_location[i],
            'keyphrases_person': keyphrases_person,
            'keyphrases_location': keyphrases_location
        })
        


    print('end ' + str(os.getpid()))
    return pd.DataFrame(results)

if __name__ == '__main__':
    df_reviews_entities = pd.read_parquet('./data_after_processing/reviews_related_data/comments_with_entities.parquet')
    sentences = df_reviews_entities.apply(get_sentences, axis=1)
    
    cpu = cpu_count() // 2
    
    # Divide the dataframe data across the processors
    chunks = np.array_split(sentences, cpu)

    # The Pool object offers a convenient means of parallelizing the execution of a function, distributing the input data across processes.
    with Pool(cpu) as pool:
        dfs = pool.map(extract_keyphrase, chunks)
        df_final = pd.concat(dfs)
        df_final.to_parquet('./data_after_processing/reviews_related_data/keyphrases_with_index.parquet')
        print(df_final.head(n=100))
        