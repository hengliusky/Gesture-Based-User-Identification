import tensorflow as tf
import numpy as np


def load_data(dataPath, labelPath):
    feature = np.loadtxt(dataPath)
    feature = feature.astype(float)
    label = np.loadtxt(labelPath)
    label = label.astype(float)

    return feature, label


def extract_batch_size(data, step, batch_size):
    shape = list(data.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step - 1) * batch_size + i) % len(data)
        batch_s[i] = data[index]

    return batch_s


def one_hot(y_, n_classes):
    y_ = y_.reshape(len(y_))
    n_values = n_classes

    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


if __name__ == '__main__':
    print("CNN & Bi-GRU")
    # train param
    n_inputs = 72  # each fame has 24 joints(without spine_base joint) with 3d coordinate, 24*4=72
    n_steps = 65  # all gesture sequence are scaled to 65 frames
    n_classes = 60  # 60 people with 60 user identity
    learning_rate = 0.0005
    min_loss = 0.0005
    max_epochs = 100000
    batch_size = 256
    n_fc1_cells = 512  # the node number of 1st full connection layer
    n_gru_cells = 512  # the node number of rnn layer

    # dataload: get the training and validation set
    X_train, Y_train = load_data("./LabeledData/Train/Gesture_V/skeleton.txt", "./LabeledData/Train/Gesture_V/userID.txt")
    X_test, Y_test = load_data("./LabeledData/Test/Gesture_V/skeleton.txt", "./LabeledData/Test/Gesture_V/userID.txt")
    # reshape the each gesture sequence into size (n_steps, n_inputs)
    X_train = X_train.reshape(-1, n_steps, n_inputs)
    X_test = X_test.reshape(-1, n_steps, n_inputs)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    '''Model Structure'''
    # Input and output formate
    X = tf.placeholder("float", [None, n_steps, n_inputs], name='x_input')
    Y = tf.placeholder("float", [None, n_classes], name='y_input')
    # 1st full connection layer
    fc1_Weights = tf.Variable(tf.random_normal([n_inputs, n_fc1_cells]), name='fc1_W')
    fc1_Biases = tf.Variable(tf.random_normal([n_fc1_cells]), name='fc1_b')
    fc1_in = tf.transpose(X, [1, 0, 2])
    fc1_in = tf.reshape(fc1_in, [-1, n_inputs])
    fc1_out = tf.nn.relu(tf.matmul(fc1_in, fc1_Weights) + fc1_Biases)
    # Bi-GRU layer
    gru_in = tf.split(fc1_out, n_steps, 0)
    gru_fw_cell = tf.nn.rnn_cell.GRUCell(n_gru_cells)
    gru_bw_cell = tf.nn.rnn_cell.GRUCell(n_gru_cells)
    gru_status, _, _ = tf.contrib.rnn.static_bidirectional_rnn(gru_fw_cell, gru_bw_cell, gru_in, dtype=tf.float32)
    gru_out = gru_status[-1]
    # (soft-max)
    softmax_Weights = tf.Variable(tf.random_normal([n_gru_cells*2, n_classes]), name='fc2_W')
    softmax_Biases = tf.Variable(tf.random_normal([n_classes]), name='fc2_b')
    softmax_out = tf.matmul(gru_out, softmax_Weights) + softmax_Biases

    # training op
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=softmax_out), name='Loss')
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    Y_pred = tf.equal(tf.argmax(softmax_out, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(Y_pred, tf.float32), name='Acc')

    # training param
    training_step = 1
    Y_test = one_hot(Y_test, n_classes)
    epcho_acc = 9999
    max_test_acc = 0.0

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    while training_step <= max_epochs and epcho_acc > min_loss:
        x_batch = extract_batch_size(X_train, training_step, batch_size)
        y_batch = one_hot(extract_batch_size(Y_train, training_step, batch_size), n_classes)

        # trainning on batch data
        _, loss_train, acc = sess.run([train_op, loss, accuracy], feed_dict={X: x_batch, Y: y_batch})
        epcho_acc = loss_train
        # test on test data
        loss_test, acc_test = sess.run([loss, accuracy], feed_dict={X: X_test, Y: Y_test})

        if float(acc_test) > max_test_acc:
            max_test_acc = float(acc_test)

        print("Training epochs " + str(training_step) + \
              ":    batch loss = " + "{:.6f}".format(loss_train) + \
              ",    accuracy = {}".format(acc))
        print("Performance on test data" + \
              ":    loss = {}".format(loss_test) + \
              ",    accuracy = {}".format(acc_test))

        training_step += 1

    print("Trainning Finished!")
    print("Max accuracy on test data is :", max_test_acc)
