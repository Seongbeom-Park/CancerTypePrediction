import tensorflow as tf
import uuid
import time

def make_embedding_layer():
    return tf.layers.Dense(6)

def dynamic_nn(x, y):
    return

def main(argv):
    # load data
    #train_csv = argv[1]
    #test_csv = argv[2]
    #train_data = [[[0]], [[1]], [[4]], [[0]], [[0]], [[1]], [[1]], [[4]], [[4]]]
    #train_data = [[[0,0]], [[1,1]], [[2,2]], [[1,0]], [[0,1]], [[2,0.5]], [[0.5,2]], [[4,1]], [[1,4]]]
    train_data = [[[0,0]],
                  [[1,0],[0,1]],
                  [[0,0],[0,0],[1,1],[1,1]],
                  [[0,0],[1,0]],
                  [[0,0],[0,1]],
                  [[0,0],[0,0],[0,0],[0,0],[2,0.5]],
                  [[0,0],[0.5,2]],
                  [[2,1],[0,0],[0,0],[2,0]],
                  [[1,0],[0,4]]]
    train_label = [0, 1, 4, 0, 0, 1, 1, 4, 4]
    test_data = [[[1.414,1.414]], [[1.732,1.732]], [[1.333,3]], [[3,1.333]]]
    test_label = [2, 3, 4, 4]

    # build graph
    #activation=tf.nn.relu
    #activation=tf.nn.sigmoid
    activation=None
    encoding_depth=5#10
    encoding_width=10#20
    nn_depth=5#10
    nn_width=10#20
    x = tf.placeholder(dtype=tf.float32)
    y = tf.placeholder(dtype=tf.float32)

    input_layer = tf.convert_to_tensor(x)
    #input_layer = tf.reshape(input_layer, [-1,1,1])
    input_layer = tf.reshape(input_layer, [-1,1,2])
    #input_layer = tf.Print(input_layer, [input_layer], summarize=10000)

    dense_layer = input_layer
    for i in range(0,encoding_depth):
        dense_layer = tf.layers.dense(dense_layer, encoding_width, activation=activation)#, trainable=True)
    #dense_layer = tf.Print(dense_layer, [dense_layer], summarize=10000)

    reduction_layer = tf.reduce_sum(dense_layer, axis=0)
    #reduction_layer = tf.Print(reduction_layer, [reduction_layer], summarize=10000)
    
    dense_layer2 = reduction_layer
    for i in range(0, nn_depth):
        dense_layer2 = tf.layers.dense(dense_layer2, nn_width, activation=activation)#, trainable=True)
    #dense_layer2 = tf.Print(dense_layer2, [dense_layer2], summarize=10000)

    output_layer = tf.layers.dense(dense_layer2, 1, activation=activation)
    #output_layer = tf.Print(output_layer, [output_layer], summarize=10000)

    label = tf.convert_to_tensor(y)
    label = tf.reshape(label, [1,1])
    #label = tf.Print(label, [label], summarize=10000)

    model = output_layer
    loss = tf.losses.mean_squared_error(label, output_layer)
    #loss = tf.Print(loss, [loss], summarize=10000)

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=1).minimize(loss)
    #optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss)

    tf.summary.scalar("loss", loss)
    merged_summary_op = tf.summary.merge_all()

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_global)
        sess.run(init_local)
        #uniq_id = "/tmp/tensorboard-layers-api/" + uuid.uuid1().__str__()[:6]
        uniq_id = "/tmp/tensorboard-layers-api/" + str(time.time())
        summary_writer = tf.summary.FileWriter(uniq_id, graph=tf.get_default_graph())
        # train
        print "step,val"
        #for step in range(0, 1):
        #for step in range(0, 500):
        for step in range(0, 5000):
            loss_list = []
            for i in range(0, len(train_data)):
                sample = train_data[i]
                y_val = train_label[i]
                #print "sample: " + str(sample)
                #sess.run(model, feed_dict={x : sample})
                _, val, summary = sess.run([optimizer, loss, merged_summary_op], feed_dict={x : sample, y : y_val})
                loss_list.append(val)
            if step %5 == 0:
                print "{},{}".format(step, sum(loss_list)/len(loss_list))
                summary_writer.add_summary(summary, step)
        
        # test
        loss_list = []
        for i in range(0, len(test_data)):
            sample = test_data[i]
            y_val = test_label[i]
            _, val = sess.run([model, loss], feed_dict = {x : sample, y : y_val})
            loss_list.append(val)
            print val
        print "{}".format(sum(loss_list)/len(loss_list))

if __name__ == '__main__':
    tf.app.run(main)
