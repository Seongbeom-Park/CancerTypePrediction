import tensorflow as tf

import pandas as pd
import numpy as np
import time

COLUMN_NAMES = [ 'Cancer_Type', 'Tumor_Sample_ID', 'Gene_Name',
                 'Chromosome', 'Start_Position', 'End_Position',
                 'Variant_Type', 'Reference_Allele','Tumor_Allele' ]

LABELS = [ 'BRCA', 'COADREAD', 'GBM', 'LUAD', 'OV', 'UCEC' ]

def dynamic_dnn(embed_dim, # TODO: change to classifier
        enc_units=[20, 20, 20, 20, 20],
        dec_units=[20, 20, 20, 20, 20],
        activation=None):
    x = tf.placeholder(dtype=tf.float32)
    y = tf.placeholder(dtype=tf.float32)

    input_layer = tf.convert_to_tensor(x)
    input_layer = tf.reshape(input_layer, [-1,1,embed_dim])
    #input_layer = tf.Print(input_layer, [input_layer], summarize=10000)

    encoding_layer = input_layer
    for units in enc_units:
        encoding_layer = tf.layers.dense(encoding_layer, units, activation=activation)
    #encoding_layer = tf.Print(encoding_layer, [encoding_layer], summarize=10000)

    reduction_layer = reduce_sum(encoding_layer, axis=0)
    reduction_layer = tf.Print(reduction_layer, [reduction_layer], summarize=10000)

    decoding_layer = reduction_layer
    for units in enc_units:
        decoding_layer = tf.layers.dense(decoding_layer, units, activation=activation)
    #decoding_layer = tf.Print(decoding_layer, [decoding_layer], summarize=10000)

    output_layer = tf.layers.dense(decoding_layer, 1, activation=activation)
    #output_layer = tf.Print(output_layer, [output_layer], summarize=10000)

    label = tf.convert_to_tensor(y)
    label = tf.reshape(label, [1,1])
    #label = tf.Print(label, [label], summarize=10000)

    loss = tf.losses.mean_squared_error(label, output_layer)
    #loss = tf.Print(loss, [loss], summarize=10000)

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=1).minimize(loss)
    #optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss)

    tf.summary.scalar("loss", loss)
    summary_op = tf.summary.merge_all()

    return output_layer, optimizer, loss, summary_op

def main(argv):
    train_file = argv[1]
    test_file = argv[2]
    steps = int(argv[3])
    event_path = "./events"#argv[4]

    # load data
    train_dataset = pd.read_csv(train_file).sample(frac=1)
    test_dataset = pd.read_csv(test_file).sample(frac=1)
    
    test_samples = test_dataset.groupby('Tumor_Sample_ID').size().reset_index(name='Count')
    train_samples = train_dataset.groupby('Tumor_Sample_ID').size().reset_index(name='Count')

    # build model
    model, optimizer, loss, summary_op = dynamic_dnn(20)

    with tf.Session() as sess:
        # initialize session
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        #event_file = event_path + "/" + str(time.time())
        event_file = "/tmp/tensorboard-layers-api/" + str(time.time())
        summary_writer = tf.summary.FileWriter(event_file, graph=tf.get_default_graph())

        # train
        for step in range(0, steps):
            loss_list = []
            for sample in train_samples:
                sample_id = sample['Tumor_Sample_ID']
                mutations = train_dataset[train_dataset['Tumor_Sample_ID'] == sample_id] # multiple mutations in a sample
                cancer_type = train_batch.iloc[0]['Cancer_Type']
                # TODO: embed gene
                # TODO: convert cancer_type to int
                _, val, summary = sess.run([optimizer, loss, summary_op],
                                           feed_dict={x : mutations, y : cancer_type})
                # TODO: ensemble
                loss_list.append(val)
            if step%5 == 0:
                # TODO: checkpoint
                print "step: {}, average mean squared error: {}".format(step, sum(loss_list)/len(loss_list))
                summary_writer.add_summary(summary, step)

        # test
        loss_list = []
        for sample in test_samples:
            sample_id = sample['Tumor_Sample_ID']
            mutations = test_dataset[test_dataset['Tumor_Sample_ID'] == sample_id] # multiple mutations in a sample
            cancer_type = test_batch.iloc[0]['Cancer_Type']
                # TODO: embed gene
                # TODO: convert cancer_type to int
            val = sess.run(model,
                           feed_dict={x : mutations, y : cancer_type})
            # TODO: ensemble
            loss_list.append(val)
        print "average mean squared error: {}".format(sum(loss_list)/len(loss_list))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
