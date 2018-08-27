import tensorflow as tf

# pip install --ignore-installed --upgrade "Download URL"

print('tensorflow v' + tf.__version__)

a = tf.constant(5.0)
b = tf.constant(3.0)
c = a * b

session = tf.Session()
print(session.run(c))