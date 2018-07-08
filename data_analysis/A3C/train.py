import tensorflow as tf
import os
import multiprocessing
import threading
from data_analysis.A3C.ActorCriticNetwork import ActorCriticNetwork
from data_analysis.A3C.Worker import Worker
from MazeEnv import MazeEnv

gamma = 0.95
load_model = False
train = True
model_path = './save'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
    global_network = ActorCriticNetwork('global', None, action_size=3)  # Generate global network
    num_workers = multiprocessing.cpu_count()  # Set workers ot number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(
            MazeEnv(randomized=False, seed=None),
            i, trainer, model_path, global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        thread = threading.Thread(target=lambda: worker.work(gamma, sess, coord, saver, train))
        thread.start()
        worker_threads.append(thread)
    coord.join(worker_threads)
