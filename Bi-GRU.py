import tensorflow as tf
import numpy as np

def load_data(dataPath, labelPath):
    feature = np.loadtxt(dataPath)
    feature = feature.astype(float)

    label = np.loadtxt(labelPath)
    label = label.astype(float)
    return feature,label

def extract_batch_size(data, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.

    shape = list(data.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step - 1) * batch_size + i) % len(data)
        batch_s[i] = data[index]

    return batch_s

def one_hot(y_,n_classes):
    y_ = y_.reshape(len(y_))
    n_values = n_classes
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

if __name__ == '__main__':
    print("CNN & Bi-GRU")
    # train param
    n_inputs = 30
    n_steps = 64
    n_classes = 3
    learning_rate = 0.0001
    min_loss = 0.0005
    max_epochs = 5
    batch_size = 256
    n_rnn_cells = 510
    n_fc1_cells = 512
    # data
    X_train, Y_train = load_data("./data/GesRecData/SkeletonData.txt", "./data/GesRecData/GesID.txt")
    X_test, Y_test = load_data("./data/TestData/SkeletonData.txt", "./data/TestData/GesID.txt")
    X_train = X_train.reshape(-1, n_steps, n_inputs)
    X_test = X_test.reshape(-1, n_steps, n_inputs)
    print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

    # net structure
    X = tf.placeholder("float", [None, n_steps, n_inputs], name='x_input')
    Y = tf.placeholder("float", [None, n_classes], name='y_input')
    # full connection layer 1
    cov1_in = tf.transpose(X, [1, 0, 2])
    cov1_in = tf.reshape(cov1_in, [-1, n_inputs, 1, 1])
    cov1_Weights = tf.Variable(tf.truncated_normal([1, 1, 1, 17], stddev=0.1), name='conv1_W')
    cov1_Biases = tf.Variable(tf.constant(0.1, shape=[17]), name='conv1_b')
    conv1_out = tf.nn.relu(tf.nn.conv2d(cov1_in, cov1_Weights, strides=[1, 1, 1, 1], padding='SAME') + cov1_Biases, name='Conv1')
    conv1_out = tf.reshape(conv1_out, [-1, 30*1*17])
    # Rnn layer
    rnn_in = tf.split(conv1_out, n_steps, 0)
    rnn_fw_cell = tf.nn.rnn_cell.GRUCell(n_rnn_cells)
    rnn_bw_cell = tf.nn.rnn_cell.GRUCell(n_rnn_cells)
    rnn_status, _, _ = tf.contrib.rnn.static_bidirectional_rnn(rnn_fw_cell, rnn_bw_cell, rnn_in, dtype=tf.float32)
    rnn_out = rnn_status[-1]
    # full connection layer 1(soft-max)
    fc1_Weights = tf.Variable(tf.random_normal([n_rnn_cells * 2, n_fc1_cells]), name='fc1_W')
    fc1_Biases = tf.Variable(tf.random_normal([n_fc1_cells]), name='fc1_b')
    fc1_out = tf.nn.relu(tf.matmul(rnn_out, fc1_Weights) + fc1_Biases, name='FC1')
    keep_prob = tf.placeholder("float", name= 'keep_prob')
    h_fc1_drop = tf.nn.dropout(fc1_out, keep_prob ,name='FC1_DropOut')
    #(soft-max)
    fc2_Weights = tf.Variable(tf.random_normal([n_fc1_cells, n_classes]), name='fc2_W')
    fc2_Biases = tf.Variable(tf.random_normal([n_classes]), name='fc2_b')
    fc2_out = tf.matmul(h_fc1_drop, fc2_Weights) + fc2_Biases
    #training op
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=fc2_out), name='Cost')
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    Y_pred = tf.equal(tf.argmax(fc2_out, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(Y_pred, tf.float32), name='Acc')

    # training param
    test_loss = []
    test_accuracy = []
    train_loss = []
    train_accuracy = []
    training_step = 1
    Y_test = one_hot(Y_test, n_classes)
    epcho_acc = 9999
    max_test_acc = 0.0

    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement = True))
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)

    while training_step <= max_epochs and epcho_acc > min_loss:
        x_batch = extract_batch_size(X_train, training_step, batch_size)
        y_batch = one_hot(extract_batch_size(Y_train, training_step, batch_size), n_classes)

        # trainning on batch data
        _, loss, acc = sess.run([train_op, cost, accuracy], feed_dict={X: x_batch, Y: y_batch, keep_prob:0.5})
        #_, loss, acc = sess.run([train_op, cost, accuracy], feed_dict={X: x_batch, Y: y_batch})
        train_loss.append(loss)
        train_accuracy.append(acc)
        epcho_acc = loss

        loss_test, acc_test = sess.run([cost, accuracy], feed_dict={X: X_test, Y: Y_test, keep_prob: 1.0})
        #loss_test, acc_test = sess.run([cost, accuracy], feed_dict={X: X_test, Y: Y_test})
        test_loss.append(loss_test)
        test_accuracy.append(acc_test)

        if float(acc_test) > max_test_acc:
            max_test_acc = float(acc_test)

        print("Training epochs " + str(training_step) + \
              ":    batch loss = " + "{:.6f}".format(loss) + \
              ",    accuracy = {}".format(acc))
        print("Performance on test data" + \
              ":    loss = {}".format(loss_test) + \
              ",    accuracy = {}".format(acc_test))

        training_step += 1

    saver.save(sess,'./model/GestureRecognition')
    print("Trainning Finished!")
    print("Max accuracy on test data is :", max_test_acc)
