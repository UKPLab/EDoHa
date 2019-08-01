import argparse
import glob

import numpy as np
import pandas as pd

import tensorflow as tf

from keras.preprocessing import sequence
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import recurrent, Embedding, Dropout, Dense, Bidirectional
from keras.optimizers import SGD, Adam

from sklearn.metrics import classification_report, accuracy_score


from evidencedetection.vectorizer import EmbeddingVectorizer

def parse_arguments():
    parser = argparse.ArgumentParser("Trains a simple BiLSTM to detect sentential arguments across multiple topics.")

    parser.add_argument("--embeddings", type=str, help="The path to the embedding folder.")
    parser.add_argument("--data", type=str, help="The path to the folder containing the TSV files with the training data.")

    return parser.parse_args()


def read_data(data_path):
    # topic_files = sorted(glob.glob("{0}/*.tsv".format(data_path)))
    # data = pd.concat([pd.read_csv(f, sep="\t") for f in topic_files]) 
    data = pd.read_csv(data_path, sep="\t")
    return data


if __name__=="__main__":

    args = parse_arguments()

    data = read_data(args.data)
    splits = data.groupby("set")
    train = splits.get_group("train")
    dev = splits.get_group("val")

    vectorizer = EmbeddingVectorizer(args.embeddings, label="NoArgument")

    train_sentences = train["sentence"].values
    lengths = map(lambda s: len(s.split(" ")), train_sentences)
    max_length = max(lengths)
    train_data, train_labels = vectorizer.prepare_data(train_sentences, train["annotation"].values)
    padded_train_data = vectorizer.sentences_to_padded_indices(train_sentences, max_length) 

    model = load_model("../models/evidencedetection.h5")

    test_sentences = dev["sentence"]
    _, test_labels = vectorizer.prepare_data(test_sentences, dev["annotation"].values)
    padded_test_data = vectorizer.sentences_to_padded_indices(test_sentences, max_length) 
    raw_preds = model.predict(padded_test_data)
    preds = np.argmax(raw_preds, axis=1)

    from pudb import set_trace; set_trace()
    print(preds.shape)
    print(classification_report(test_labels, preds, target_names=["no Argument", "Argument"]))  # we defined no argument to be label 0 in the embedding vectorizer

    print(accuracy_score(test_labels, preds))
