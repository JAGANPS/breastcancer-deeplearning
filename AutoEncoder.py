import tensorflow as tf

weights={
    'layer1': tf.Variable(tf.random_normal([11163,400]),trainable=False,name='weights_layer1'),
    'layer2': tf.Variable(tf.random_normal([400,225]),trainable=False,name='weights_layer2'),
    'layer3': tf.Variable(tf.random_normal([225,2]),name='weights_layer3')
}

biases=tf.Variable(tf.random_normal([2]),name='biases_layer3')

x=tf.placeholder(tf.float32, [None,61,61,3])
y=tf.placeholder(tf.float32, [None,2])
learning_rate=tf.placeholder(tf.float32)

def network(x):
    input=tf.reshape(x,[-1,11163],name='reshape_input')
    layer1=tf.matmul(input,weights['layer1'])
    layer2=tf.matmul(layer1,weights['layer2'])
    layer3=tf.add(tf.matmul(layer2,weights['layer3']),biases)
    return layer3

inference=tf.nn.softmax(network(x))

predict_y=network(x)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=predict_y))
optimize=tf.train.AdamOptimizer(learning_rate).minimize(loss)

def save(sess):
    path='./layer3/var'
    saver=tf.train.Saver({'weights_layer3': weights['layer3'],'biases_layer3': biases})
    saver.save(sess,path)

def restore(sess):
    saver1=tf.train.Saver({'weights_layer1': weights['layer1']})
    saver1.restore(sess,tf.train.latest_checkpoint('./layer1/'))
    saver2=tf.train.Saver({'weights_layer2': weights['layer2']})
    saver2.restore(sess,tf.train.latest_checkpoint('./layer2/'))
    saver3=tf.train.Saver({'weights_layer3': weights['layer3'], 'biases_layer3': biases})
    saver3.restore(sess,tf.train.latest_checkpoint('./layer3/'))
