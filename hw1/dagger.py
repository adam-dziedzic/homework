import tensorflow as tf
from hw1 import params
import numpy as np
from hw1 import tf_util
import time
from hw1.utils import get_mean_std, read_data, create_model


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
    min_mse = float('inf')
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
            if mse_run < min_mse:
                min_mse = mse_run
                saver.save(sess, params.get_model_file(env_name=env_name))


def train_policy(env_name, hidden_units=100, train_steps=10000, rollouts=10000):
    obs, actions = read_data(
        params.expert_data_dir + "/" + env_name + '-' + str(rollouts) + ".pkl")
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
        input_size, output_size, mean, std = params.input_output_size[env_name]
    except KeyError:
        obs, actions = read_data(
            params.expert_data_dir + "/" + env_name + ".pkl")
        input_size = obs.shape[1]
        output_size = actions.shape[1]
        mean, std = get_mean_std(obs)

    input_ph, output_ph, output_pred = create_model(
        mean=mean, std=std, input_size=input_size, output_size=output_size,
        hidden_units=hidden_units)

    # restore the saved model
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, params.get_model_file(env_name=env_name))

    policy_fn = tf_util.function([input_ph], output_pred)
    return policy_fn


if __name__ == "__main__":
    env_name = 'Reacher-v2'
    start = time.time()
    input_size, output_size = train_policy(env_name=env_name,
                                           hidden_units=params.hidden_units,
                                           train_steps=params.train_steps,
                                           rollouts=params.rollouts)
    stop = time.time()
    print('elapsed time (sec): ', stop - start)
    print("input size: ", input_size)
    print("output size: ", output_size)