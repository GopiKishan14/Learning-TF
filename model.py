import tensorflow as tf

#features
x = tf.constant(1.0 , name="input")
#weight
w = tf.Variable(0.8 , name="weight")
#hypothesis , predicted output
y = tf.multiply(w,x , name="output")

#true output
y_ = tf.constant(0.0 , name="correct_value")

#L2 loss funcion
loss = tf.pow(y - y_ , 2 , name="loss")

#Optimizer
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)


for value in [x,w,y,y_,loss]:
    tf.summary.scalar(value.op.name , value)

summaries = tf.summary.merge_all()

sess = tf.Session()

summary_writer = tf.summary.FileWriter('log_simple_stats' , sess.graph)

init  = tf.global_variables_initializer()

sess.run(init)

for i in range(100):
    summary_writer.add_summary(sess.run(summaries), i)

    sess.run(train_step)