import tensorflow as tf
import pickle
from hw1 import utils
import numpy as np
from hw1 import tf_util


def read_data(filename):
    with open(file=filename, mode="rb") as f:
        data = pickle.load(file=f)
    observations = data['observations']
    actions = np.squeeze(data['actions'])
    return observations, actions


def create_model(mean, std, input_size=1, output_size=1, hidden_units=20):
    # create input PlaceHolder
    # in the behavioral clonning case, you would use, for example, 7 instead of 1 for the
    # last dimension, since 7 represents 7 joins of our robot
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, input_size])

    # normalize the input
    input_ph_norm = (input_ph - mean) / (std + 1e-6)

    # create the output PlaceHolder: this is what we expect as the output and what we use
    # to compute the error and back-propagate it.
    # for the output in the behavioral cloning algorithm you could use 4 for the 4 possible
    # actions to be takes for each of the 2 joins, apply the force to the right or left, we
    # would use 4 instead of 1 in the second dimension in the output_ph
    # the ground truth labels are fed into the output_ph placeholdergT
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

    # create variables
    W0 = tf.get_variable(
        name='W0', shape=[input_size, hidden_units],
        initializer=tf.contrib.layers.xavier_initializer())
    W1 = tf.get_variable(
        name='W1', shape=[hidden_units, hidden_units],
        initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable(
        name='W2', shape=[hidden_units, output_size],
        initializer=tf.contrib.layers.xavier_initializer())

    b0 = tf.get_variable(name='b0', shape=[hidden_units],
                         initializer=tf.constant_initializer(0.))
    b1 = tf.get_variable(name='b1', shape=[hidden_units],
                         initializer=tf.constant_initializer(0.))
    b2 = tf.get_variable(name='b2', shape=[1],
                         initializer=tf.constant_initializer(0.))

    weights = [W0, W1, W2]
    biases = [b0, b1, b2]
    """
    We want to predict the values of the sin function, which has the output values in the
    range from -1 to +1, so we do not use relu non-linearit in the last layer because it would
    limit our final output to only the positive values. However, we could imagine using the 
    tanh non-linearity that has the scope from -1 to + 1. 
    """
    # activations = [tf.nn.relu, tf.nn.relu, None]
    activations = [tf.nn.relu, tf.nn.relu, tf.nn.tanh]
    # activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu]

    # create computation graph
    layer = input_ph_norm
    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W) + b
        if activation is not None:
            layer = activation(layer)
    output_pred = layer

    return input_ph, output_ph, output_pred


def train_model(inputs, outputs, output_pred, input_ph, output_ph, env_name,
                sess, train_steps=10000):
    # create loss (mean squared error loss)
    # we compute the average loss over the mini-batch
    # the output_pred and output_ph have many data items - the size of the mini-batch
    # then we reduce over the mini-batch to get a single metric on how well our model
    # predicted the output
    mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))

    # create optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)

    # initialize variables
    sess.run(tf.global_variables_initializer())
    # create saver to save model variables
    saver = tf.train.Saver()

    # run training
    batch_size = 32
    for training_step in range(train_steps):
        # get a random subset of the training data
        indices = np.random.randint(low=0, high=len(inputs), size=batch_size)
        input_batch = inputs[indices]
        output_batch = outputs[indices]

        # run the optimizer and get the mse
        # passing opt to the sess.run do the one time backprop
        _, mse_run = sess.run([opt, mse],
                              feed_dict={input_ph: input_batch,
                                         output_ph: output_batch})

        # print the mse every so often
        if training_step % 1000 == 0:
            print('{0:04d} mse: {1:.9f}'.format(training_step, mse_run))
            saver.save(sess, utils.model_prefix + env_name + '.ckpt')


def get_mean_std(obs):
    mean = np.mean(obs, axis=0)
    std = np.std(obs, axis=0)
    return mean, std


def train_policy(env_name, hidden_units=20, train_steps=10000):
    obs, actions = read_data(utils.expert_data_dir + "/" + env_name + ".pkl")
    mean, std = get_mean_std(obs)
    print("mean, std: ", mean, " ", std)
    input_size = obs.shape[1]
    output_size = actions.shape[1]
    with tf.Session() as sess:
        input_ph, output_ph, output_pred = create_model(
            mean=mean, std=std, input_size=input_size, output_size=output_size,
            hidden_units=hidden_units)
        train_model(inputs=obs, outputs=actions, input_ph=input_ph,
                    output_ph=output_ph, output_pred=output_pred,
                    env_name=env_name, sess=sess, train_steps=train_steps)
    return input_size, output_size


def load_policy(env_name, hidden_units):
    try:
        input_size, output_size, mean, std = utils.input_output_size[env_name]
    except KeyError:
        obs, actions = read_data(
            utils.expert_data_dir + "/" + env_name + ".pkl")
        input_size = obs.shape[1]
        output_size = actions.shape[1]
        mean, std = get_mean_std(obs)

    input_ph, output_ph, output_pred = create_model(
        mean=mean, std=std, input_size=input_size, output_size=output_size,
        hidden_units=hidden_units)

    # restore the saved model
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, utils.model_prefix + env_name + '.ckpt')

    policy_fn = tf_util.function([input_ph], output_pred)
    return policy_fn


if __name__ == "__main__":
    env_name = 'Reacher-v2'
    hidden_units = 100
    train_steps = 50000
    input_size, output_size = train_policy(env_name=env_name,
                                           hidden_units=hidden_units,
                                           train_steps=train_steps)
    print("input size: ", input_size)
    print("output size: ", output_size)
