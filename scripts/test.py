import tensorflow as tf
import pandas as pd
import numpy as np
import scipy as sp

import os

COLUMN_NAMES = [ 'Cancer_Type', 'Tumor_Sample_ID', 'Gene_Name',
                 'Chromosome', 'Start_Position', 'End_Position',
                 'Variant_Type', 'Reference_Allele','Tumor_Allele' ]

def load_data(train_file, test_file, gene_file, cancer_file, sample_file):
    train_dataset = pd.read_csv(train_file)
    test_dataset = pd.read_csv(test_file)
    gene_dataset = pd.read_csv(gene_file)
    cancer_dataset = pd.read_csv(cancer_file)
    sample_dataset = pd.read_csv(sample_file)

    train_sample = train_dataset.groupby('Tumor_Sample_ID').size().reset_index(name='Count') 
    test_sample = test_dataset.groupby('Tumor_Sample_ID').size().reset_index(name='Count')

    gene_index_pair = dict(zip(gene_dataset['Gene_Name'], gene_dataset['Index']))
    cancer_index_pair = dict(zip(cancer_dataset['Cancer_Type'], cancer_dataset['Index']))

    train_dataset['Gene_Index'] = train_dataset['Gene_Name'].map(gene_index_pair)
    test_dataset['Gene_Index'] = test_dataset['Gene_Name'].map(gene_index_pair)
    gene_type_count = len(gene_index_pair)
    cancer_type_count = len(cancer_index_pair)

    train_stat = train_dataset.groupby(['Cancer_Type', 'Tumor_Sample_ID']).size().reset_index(name='Count')
    train_stat = train_stat.groupby(['Cancer_Type']).size().reset_index(name='Count')
    train_stat = dict(zip(train_stat['Cancer_Type'], train_stat['Count']))
    print "train : {}".format(train_stat)

    test_stat = test_dataset.groupby(['Cancer_Type', 'Tumor_Sample_ID']).size().reset_index(name='Count')
    test_stat = test_stat.groupby(['Cancer_Type']).size().reset_index(name='Count')
    test_stat = dict(zip(test_stat['Cancer_Type'], test_stat['Count']))
    print "test : {}".format(test_stat)

    weights = [0] * cancer_type_count
    k = 0
    for cancer in train_stat:
        weights[cancer_index_pair[cancer]] = 1.0 / train_stat[cancer]
        k += 1.0 / train_stat[cancer]
    weights = [w/k for w in weights]

    return train_dataset, test_dataset, train_sample, test_sample, gene_type_count, cancer_type_count, cancer_index_pair, weights, sample_dataset


def main(argv):
    train_file = argv[1] + "/train.csv"
    test_file = argv[1] + "/test.csv"
    gene_file = argv[1] + "/gene.csv"
    cancer_file = argv[1] + "/cancer.csv"
    sample_file = argv[1] + "/sample.csv"

    print "Load dataset"
    train_dataset, test_dataset, train_sample, test_sample, gene_type_count, cancer_type_count, cancer_index_pair, weights, sample_dataset = load_data(train_file, test_file, gene_file, cancer_file, sample_file)

    sample_count = len(train_sample)
    sample_index_dict = dict(zip(sample_dataset['Tumor_Sample_ID'], sample_dataset['Tumor_Sample_Index']))
    matrix = sp.sparse.csr_matrix((sample_count, gene_type_count)).toarray()
    for index, row in train_dataset.iterrows():
        sample = row['Tumor_Sample_ID']
        gene = row['Gene_Index']
        sample_index = sample_index_dict[sample]
        matrix[sample_index, gene] = 1

    print matrix


if __name__ == '__main__':
    tf.app.run(main)
