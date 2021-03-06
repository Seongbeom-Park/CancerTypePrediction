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

    return train_dataset, test_dataset, train_sample, test_sample, gene_type_count, cancer_type_count, cancer_index_pair, weights


def build_model(gene_type_count, cancer_type_count, weights):
    input_layer = tf.placeholder(dtype=tf.int32)

    encoding_layer = input_layer
    encoding_layer = tf.one_hot(encoding_layer, gene_type_count)
    encoding_layer = tf.reduce_sum(encoding_layer, 0)
    encoding_layer = tf.clip_by_value(encoding_layer, 0, 1)
    encoding_layer = tf.reshape(encoding_layer, [1, gene_type_count])
    #encoding_layer = tf.Print(encoding_layer, [encoding_layer], summarize=100)

    activation = tf.nn.relu
    dense_layer = encoding_layer
    dense_layer = tf.layers.dense(dense_layer, 256)#, activation=activation)
    dense_layer = tf.layers.dropout(dense_layer)
    dense_layer = tf.layers.dense(dense_layer, 256)#, activation=activation)
    dense_layer = tf.layers.dropout(dense_layer)
    dense_layer = tf.layers.dense(dense_layer, 256)#, activation=activation)
    dense_layer = tf.layers.dropout(dense_layer)
    dense_layer = tf.layers.dense(dense_layer, 256)#, activation=activation)
    dense_layer = tf.layers.dropout(dense_layer)
    dense_layer = tf.layers.dense(dense_layer, 256)#, activation=activation)
    dense_layer = tf.layers.dropout(dense_layer)
    dense_layer = tf.layers.dense(dense_layer, 256)#, activation=activation)
    dense_layer = tf.layers.dropout(dense_layer)
    dense_layer = tf.layers.dense(dense_layer, 256)#, activation=activation)
    dense_layer = tf.layers.dropout(dense_layer)
    dense_layer = tf.layers.dense(dense_layer, 256)#, activation=activation)
    dense_layer = tf.layers.dropout(dense_layer)
    dense_layer = tf.layers.dense(dense_layer, 256)#, activation=activation)
    dense_layer = tf.layers.dropout(dense_layer)
    dense_layer = tf.layers.dense(dense_layer, 128)#, activation=activation)
    dense_layer = tf.layers.dropout(dense_layer)
    dense_layer = tf.layers.dense(dense_layer, 64)#, activation=activation)
    dense_layer = tf.layers.dropout(dense_layer)
    dense_layer = tf.layers.dense(dense_layer, 32)#, activation=activation)
    dense_layer = tf.layers.dropout(dense_layer)
    dense_layer = tf.layers.dense(dense_layer, cancer_type_count)#, activation=activation)
    #dense_layer = tf.layers.dense(dense_layer, cancer_type_count, activation=tf.nn.sigmoid)
    #dense_layer = tf.Print(dense_layer, ["dense_layer", dense_layer], summarize=cancer_type_count)
    
    softmax_layer = dense_layer
    softmax_layer = tf.contrib.layers.softmax(softmax_layer)
    #softmax_layer = tf.Print(softmax_layer, [softmax_layer], summarize=cancer_type_count)

    model = softmax_layer

    output_layer = tf.placeholder(tf.int32)

    output_encoding_layer = output_layer
    output_encoding_layer = tf.one_hot(output_encoding_layer, cancer_type_count)
    output_encoding_layer = tf.cast(output_encoding_layer, tf.float32)
    output_encoding_layer = tf.reshape(output_encoding_layer, [1, cancer_type_count])
    #output_encoding_layer = tf.Print(output_encoding_layer, ['output_encoding_layer', output_encoding_layer], summarize=cancer_type_count)

    #weights_tensor = tf.constant(weights)
    #loss = tf.losses.softmax_cross_entropy(output_encoding_layer, softmax_layer)
    #loss = tf.keras.losses.categorical_crossentropy(output_encoding_layer, softmax_layer)
    #loss = tf.losses.sigmoid_cross_entropy(output_encoding_layer, softmax_layer)
    #loss = tf.losses.softmax_cross_entropy(output_encoding_layer, dense_layer)
    loss = tf.losses.sigmoid_cross_entropy(output_encoding_layer, dense_layer)
    #loss = tf.losses.softmax_cross_entropy(output_encoding_layer, dense_layer, weights=weights_tensor)
    #loss = tf.keras.losses.categorical_crossentropy(output_encoding_layer, dense_layer)
    #loss = tf.losses.softmax_cross_entropy(output_encoding_layer, dense_layer)
    #loss = tf.Print(loss, [loss])

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    return input_layer, output_layer, model, loss, optimizer

def main(argv):
    train_file = argv[1] + "/train.csv"
    test_file = argv[1] + "/test.csv"
    gene_file = argv[1] + "/gene.csv"
    cancer_file = argv[1] + "/cancer.csv"

    print "Load dataset"
    train_dataset, test_dataset, train_sample, test_sample, gene_type_count, cancer_type_count, cancer_index_pair, weights = load_data(train_file, test_file, gene_file, cancer_file)

    print "Build model"
    input_layer, output_layer, model, loss, optimizer = build_model(gene_type_count, cancer_type_count, weights)
    
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    with tf.Session() as sess:
        print "Initialize variables"
        sess.run(init_global)
        sess.run(init_local)

        #print "Load model"
        saver = tf.train.Saver()
        #saver.restore(sess, "./cancer_nn/1.ckpt")

        train_sample_count = len(train_sample)
        test_sample_count = len(test_sample)

        for step in range(0,100):
            # train
            print "Train start"
            # shuffle sample order
            shuffled_train_sample = pd.core.frame.DataFrame(train_sample).reset_index().sample(frac=1.0)
            i=0
            for sample_id in shuffled_train_sample['Tumor_Sample_ID']:
                sample = train_dataset[train_dataset['Tumor_Sample_ID'] == sample_id]
                cancer = sample['Cancer_Type'].iloc[0]
                #print sample.groupby('Gene_Name').size()
                mod, _, val = sess.run(
                        [model, optimizer, loss],
                        feed_dict = {
                            input_layer : sample['Gene_Index'],
                            output_layer : cancer_index_pair[cancer]
                            }
                        )
                i+=1
                output = "{},{}/{},{},{}[{}],{}".format(step, i, train_sample_count, sample_id, cancer, cancer_index_pair[cancer], val)
                print output
                with open("./history_train.csv", "a") as my_file:
                    my_file.write(output + "\n")
                # write train progress

            print "Save model"
            saver.save(sess, "./cancer_nn/" + str(step) + ".ckpt")

            # test
            print "Test start"
            i=0
            correct_count = 0
            for sample_id in test_sample['Tumor_Sample_ID']:
                sample = test_dataset[test_dataset['Tumor_Sample_ID'] == sample_id]
                cancer = sample['Cancer_Type'].iloc[0]
                #print sample.groupby('Gene_Name').size()
                val = sess.run(
                        [model],
                        feed_dict = {
                            input_layer : sample['Gene_Index'],
                            output_layer : cancer_index_pair[cancer]
                            }
                        )
                i+=1
                cancer_index = np.argmax(val[0])
                cancer_type = [t for t in cancer_index_pair if cancer_index_pair[t] == cancer_index][0]
                if cancer_index == cancer_index_pair[cancer]:
                    correct_count+=1
                output = "{},{}/{},{},{}[{}],{}[{}]".format(step, i, test_sample_count, sample_id, cancer, cancer_index_pair[cancer], cancer_type, cancer_index)
                print output
                with open("./history_test.csv", "a") as my_file:
                    my_file.write(output + "\n")
            # write test result
            output = "{}, Correct / Total = {} / {}".format(step, correct_count, test_sample_count)
            with open("./history_test.csv", "a") as my_file:
                my_file.write(output + "\n")
            print output


if __name__ == '__main__':
    tf.app.run(main)
