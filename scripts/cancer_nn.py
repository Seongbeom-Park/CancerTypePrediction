import tensorflow as tf
import pandas as pd
import numpy as np

COLUMN_NAMES = [ 'Cancer_Type', 'Tumor_Sample_ID', 'Gene_Name',
                 'Chromosome', 'Start_Position', 'End_Position',
                 'Variant_Type', 'Reference_Allele','Tumor_Allele' ]

LABELS = [ 'BRCA', 'COADREAD', 'GBM', 'LUAD', 'OV', 'UCEC' ]

def load_data(train_file, test_file, gene_file):
    train_dataset = pd.read_csv(train_file)
    test_dataset = pd.read_csv(test_file)
    gene_dataset = pd.read_csv(gene_file)

    train_sample = train_dataset.groupby('Tumor_Sample_ID').size().reset_index(name='Count') 
    test_sample = test_dataset.groupby('Tumor_Sample_ID').size().reset_index(name='Count')

    gene_index_pair = dict(zip(gene_dataset['Gene_Name'], gene_dataset['Index']))

    train_dataset['Gene_Index'] = train_dataset['Gene_Name'].map(gene_index_pair)
    test_dataset['Gene_Index'] = test_dataset['Gene_Name'].map(gene_index_pair)
    max_index = len(gene_index_pair)

    return train_dataset, test_dataset, train_sample, test_sample, max_index


def build_model(max_index):
    input_layer = tf.placeholder(dtype=tf.int32)

    encoding_layer = tf.convert_to_tensor(input_layer)
    #encoding_layer = tf.Print(encoding_layer, [encoding_layer], summarize=10000)
    encoding_layer = tf.one_hot(encoding_layer, max_index)
    #encoding_layer = tf.Print(encoding_layer, [encoding_layer], summarize=10000)
    encoding_layer = tf.reduce_sum(encoding_layer, 0)
    #encoding_layer = tf.Print(encoding_layer, [encoding_layer], summarize=20000)

    model = encoding_layer
    return input_layer, model

def main(argv):
    train_file = argv[1]
    test_file = argv[2]
    gene_file = argv[3]

    train_dataset, test_dataset, train_sample, test_sample, max_index = load_data(train_file, test_file, gene_file)

    input_layer, model = build_model(max_index)
    

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_global)
        sess.run(init_local)
        for sample_id in train_sample['Tumor_Sample_ID']:
            sample = train_dataset[train_dataset['Tumor_Sample_ID'] == sample_id]
            #print sample.groupby('Gene_Name').size()
            sess.run(
                    model,
                    feed_dict = {
                        input_layer : sample['Gene_Index']
                        }
                    )
            exit()

    return

if __name__ == '__main__':
    tf.app.run(main)
