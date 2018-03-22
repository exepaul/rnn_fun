import numpy as np
import random
from tensorflow.contrib import rnn
import collections
import tensorflow as tf
vocabulary=[]
with open('belling_the_cat.txt','r') as f:
    for line in f:
        vocabulary.extend(line.strip().split())
print(vocabulary)

def most_common(file):
    chr_2_idx = {}
    idx_2_chr = {}
    most_common_words =[words[0] for words in collections.Counter(file).most_common()]
    for i in most_common_words:
        chr_2_idx[i]=len(chr_2_idx)
    for i,j in chr_2_idx.items():
        idx_2_chr[j]=i

    return chr_2_idx,idx_2_chr

chr_to_index,index_to_chr = most_common(vocabulary)

print(chr_to_index)
print(index_to_chr)

input_x=tf.placeholder(tf.float32,[None,4,1])
input_y=tf.placeholder(tf.int32,[None,1])



cell = rnn.BasicLSTMCell(num_units=len(chr_to_index), state_is_tuple=True)
initial_cell = cell.zero_state(1, tf.float32)
rnn_model, states = tf.nn.dynamic_rnn(cell, input_x, initial_state=initial_cell, dtype=tf.float32)
x_for_fc = tf.reshape(rnn_model, [-1, len(chr_to_index)])
real_output = tf.contrib.layers.fully_connected(inputs=x_for_fc, num_outputs=len(chr_to_index), activation_fn=None)


real_output = tf.reshape(real_output, [4, len(chr_to_index), 1])


weights = tf.ones([4, len(chr_to_index)])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=real_output, targets=input_y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

prediction = tf.argmax(real_output, axis=2)


prediction = tf.argmax(real_output, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        random_no = random.randint(0, len(chr_to_index) - 4)
        one_hotx = [0] * len(chr_to_index)
        real_inputx = []
        for i in range(random_no, random_no + 4):
            one_hotx[chr_to_index[vocabulary[i]]] = 1.0
            real_inputx.append(one_hotx)
        print("ree",real_inputx)
        real_inputxx = np.reshape(real_inputx, [-1, 4, 1])
        real_output = chr_to_index[vocabulary[random_no + 4]]
        real_output = np.reshape(real_output, [1, -1])
        l, _ = sess.run([loss, train], feed_dict={input_x: real_inputxx, input_y: real_output})
        result = sess.run(prediction, feed_dict={input_x: real_inputxx})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", real_output)

        # print char using dic
        result_str = [index_to_chr[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))










