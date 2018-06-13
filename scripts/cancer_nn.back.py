import tensorflow as tf
import pandas as pd
import numpy as np

import os

COLUMN_NAMES = [ 'Cancer_Type', 'Tumor_Sample_ID', 'Gene_Name',
                 'Chromosome', 'Start_Position', 'End_Position',
                 'Variant_Type', 'Reference_Allele','Tumor_Allele' ]

def load_data(train_file, test_file, gene_file, cancer_file):
    train_dataset = pd.read_csv(train_file)
    test_dataset = pd.read_csv(test_file)
    gene_dataset = pd.read_csv(gene_file)
    cancer_dataset = pd.read_csv(cancer_file)

    train_sample = train_dataset.groupby('Tumor_Sample_ID').size().reset_index(name='Count') 
    test_sample = test_dataset.groupby('Tumor_Sample_ID').size().reset_index(name='Count')

    gene_index_pair = dict(zip(gene_dataset['Gene_Name'], gene_dataset['Index']))
    cancer_index_pair = dict(zip(cancer_dataset['Cancer_Type'], cancer_dataset['Index']))

    train_dataset['Gene_Index'] = train_dataset['Gene_Name'].map(gene_index_pair)
    test_dataset['Gene_Index'] = test_dataset['Gene_Name'].map(gene_index_pair)
    gene_type_count = len(gene_index_pair)
    cancer_type_count = len(cancer_index_pair)

    return train_dataset, test_dataset, train_sample, test_sample, gene_type_count, cancer_type_count, cancer_index_pair


def build_model(gene_type_count, cancer_type_count):
    input_layer = tf.placeholder(dtype=tf.int32)

    encoding_layer = tf.convert_to_tensor(input_layer)
    #encoding_layer = tf.Print(encoding_layer, [encoding_layer], summarize=10000)
    encoding_layer = tf.one_hot(encoding_layer, gene_type_count)
    #encoding_layer = tf.Print(encoding_layer, [encoding_layer], summarize=10000)
    encoding_layer = tf.reduce_sum(encoding_layer, 0)
    #encoding_layer = tf.Print(encoding_layer, [encoding_layer], summarize=1000)
    encoding_layer = tf.clip_by_value(encoding_layer, 0, 1)
    #encoding_layer = tf.Print(encoding_layer, [encoding_layer], summarize=1000)
    encoding_layer = tf.reshape(encoding_layer, [-1, gene_type_count])

    activation = tf.nn.relu
    dense_layer = tf.layers.dense(encoding_layer, 16384, activation=activation)
    dense_layer = tf.layers.dense(dense_layer, 8192, activation=activation)
    dense_layer = tf.layers.dense(dense_layer, 4096, activation=activation)
    dense_layer = tf.layers.dense(dense_layer, 2048, activation=activation)
    dense_layer = tf.layers.dense(dense_layer, 1024, activation=activation)
    dense_layer = tf.layers.dense(dense_layer, 512, activation=activation)
    dense_layer = tf.layers.dense(dense_layer, 256, activation=activation)
    dense_layer = tf.layers.dense(dense_layer, 128, activation=activation)
    dense_layer = tf.layers.dense(dense_layer, 64, activation=activation)
    dense_layer = tf.layers.dense(dense_layer, 32, activation=activation)
    dense_layer = tf.layers.dense(dense_layer, 16, activation=activation)
    #dense_layer = tf.layers.dense(dense_layer, 8, activation=activation)
    dense_layer = tf.layers.dense(dense_layer, cancer_type_count, activation=activation)
    
    softmax_layer = tf.contrib.layers.softmax(dense_layer)
    #softmax_layer = tf.Print(softmax_layer, [softmax_layer], summarize=cancer_type_count)

    model = softmax_layer

    output_layer = tf.placeholder(tf.int32)
    output_encoding_layer = tf.contrib.layers.softmax(dense_layer)

    loss = tf.losses.softmax_cross_entropy(output_encoding_layer, softmax_layer)
    #loss = tf.Print(loss, [loss])
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1).minimize(loss)
    return input_layer, output_layer, model, loss, optimizer

def main(argv):
    train_file = argv[1]
    test_file = argv[2]
    gene_file = argv[3]
    cancer_file = argv[4]

    train_dataset, test_dataset, train_sample, test_sample, gene_type_count, cancer_type_count, cancer_index_pair = load_data(train_file, test_file, gene_file, cancer_file)

    input_layer, output_layer, model, loss, optimizer = build_model(gene_type_count, cancer_type_count)
    
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_global)
        sess.run(init_local)

        saver = tf.train.Saver()
        checkpoint_file = "./test.ckpt"
        if os.path.isfile("./checkpoint"):
            saver.restore(sess, "./test.ckpt")

        sample_count = len(train_sample)
        i=0

        # train
        for sample_id in train_sample['Tumor_Sample_ID']:
            i+=1
            print "progress: {} / {}".format(i, sample_count)
            sample = train_dataset[train_dataset['Tumor_Sample_ID'] == sample_id]
            cancer = train_dataset['Cancer_Type'].iloc[0]
            #print sample.groupby('Gene_Name').size()
            _, val = sess.run(
                    [optimizer, loss],
                    feed_dict = {
                        input_layer : sample['Gene_Index'],
                        output_layer : cancer_index_pair[cancer]
                        }
                    )
            print "{} : {}[{}]".format(sample_id, cancer, cancer_index_pair[cancer])
            print "loss: {}".format(val)
            print ""
        saver.save(sess, "./test.ckpt")

        # test
        for sample_id in test_sample['Tumor_Sample_ID']:
            i+=1
            print "progress: {} / {}".format(i, sample_count)
            sample = test_dataset[test_dataset['Tumor_Sample_ID'] == sample_id]
            cancer = test_dataset['Cancer_Type'].iloc[0]
            #print sample.groupby('Gene_Name').size()
            val = sess.run(
                    [model],
                    feed_dict = {
                        input_layer : sample['Gene_Index'],
                        output_layer : cancer_index_pair[cancer]
                        }
                    )
            print "{} : {}[{}]".format(sample_id, cancer, cancer_index_pair[cancer])
            print "val: {}".format(val)
            print ""

if __name__ == '__main__':
    tf.app.run(main)
