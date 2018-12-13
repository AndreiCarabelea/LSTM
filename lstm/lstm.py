import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import pandas as pd
from tensorflow.contrib.labeled_tensor import batch





#sequence of three tokens 
#each word token is input at a new time step

#how many sequences used for training in one batch , one sequence  has more tokens. 
# one bacth one dquence 
batch_size = 5 

# hidden state size, the same size as outputs  
lstm_units = 128

#max number of tokens in a sequence 
max_sequence_length = 30

# the lstm outputs if sentiment iis positive or negative 
num_classes = 4

# how long is  token embedding 
embedding_dimension = 4 


# used in training 
num_iterations = 50000


# 
maxnumdays_prediction = 30



graph = tf.Graph()
# one batch  is one sequence 
with graph.as_default(): 
    # each elemt ina sequence is an embedding ( array ) 
    tf_data = tf.placeholder(dtype = tf.float32, shape = (batch_size, max_sequence_length, embedding_dimension))
    tf_labels = tf.placeholder(dtype = tf.float32, shape = (batch_size, num_classes))
    
    tf_single_data = tf.placeholder(dtype = tf.float32, shape = (1, max_sequence_length, embedding_dimension))
    
    weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]), dtype = tf.float32)
    bias = tf.Variable(tf.constant(0.0, shape=[num_classes]), dtype = tf.float32) 
   
    lstm_cell = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units = lstm_units)
    wrapped_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell,output_keep_prob=0.8)
    output, state = tf.nn.dynamic_rnn(wrapped_lstm_cell, tf_data, dtype=tf.float32)
    
    
#     output = tf.transpose(output, [1, 0, 2])
#     last = tf.gather(output, int(output.get_shape()[0]) - 1)
    #extract the output from the last cell, result is bach_size * num_classes 
    last = tf.gather(output, indices = max_sequence_length - 1, axis = 1)
    
    # batch_size * num_classes
    prediction  = tf.matmul(last, weight) + bias 
    loss = tf.reduce_max(tf.losses.absolute_difference(predictions = prediction, labels = tf_labels))/tf.reduce_mean(tf_labels)
    optimizer = tf.train.AdamOptimizer(0.05).minimize(loss)
    
    opredict, _ =  tf.nn.dynamic_rnn(wrapped_lstm_cell, tf_single_data, dtype=tf.float32)
    
    lastpredict = tf.gather(opredict, indices = max_sequence_length - 1, axis = 1)
    
#     opredict = tf.transpose(opredict, [1, 0, 2])
#     lastpredict = tf.gather(opredict, int(output.get_shape()[0]) - 1)
    
    lastpredict  = tf.matmul(lastpredict, weight) + bias
    
with tf.Session(graph = graph, config=tf.ConfigProto(intra_op_parallelism_threads=24)) as sess:
    tf.global_variables_initializer().run();
    
    my_data = pd.read_excel('bitcoin.xls').values
    my_data = my_data[::-1,(0,1,2,3)]
     
    index = 0 
    maxStartOffset = len(my_data) - max_sequence_length  - 1
    
    data_batch  = np.zeros(shape = (maxStartOffset + 1, max_sequence_length, num_classes), dtype = np.float32)
    label_batch = np.zeros(shape = (maxStartOffset + 1, num_classes), dtype = np.float32)
    
    while index <= maxStartOffset:    
        data_batch[index] = my_data[index: index + max_sequence_length, :] 
        label_batch[index] = my_data[index + max_sequence_length, :]
        index += 1
        
    data_batch.shape =  (index, max_sequence_length, num_classes)
    label_batch.shape = (index, num_classes)
    
    index = 0
    learnIters = 0
    while True:
        if index > data_batch.shape[0] - 1 - batch_size:
            index = 0
            
        data_batch_learning = data_batch[index : index+batch_size, :,: ]
        labels_learning = label_batch[index : index+batch_size, :] 
        #fd={ tf_data=data_batch_learning,  tf_labels = labels_learning }
        fd = {tf_data: data_batch_learning, tf_labels: labels_learning}
        _,evaluatedLoss = sess.run([optimizer, loss], feed_dict=fd)
        
        if learnIters % 100 == 0:
            print(evaluatedLoss)
        
        index+= 1
        learnIters+= 1
        
        if learnIters > num_iterations:
            break
    
    # one sequence    
     
    data_batch_predict = data_batch[-1, :,: ]
    data_batch_predict.shape = (1,max_sequence_length,num_classes)
    fd = {tf_single_data: data_batch_predict}
    lp = sess.run(lastpredict, feed_dict = fd)
    print(lp)
    
    for i in range(maxnumdays_prediction):
        data_batch_predict = np.vstack((data_batch_predict[0,1:,:], lp))
        data_batch_predict.shape = (1,max_sequence_length,num_classes)
        fd = {tf_single_data: data_batch_predict}
        lp = sess.run(lastpredict, feed_dict = fd)
        print(lp)
        
    
    
        
    
            
    