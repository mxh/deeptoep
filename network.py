from game import *
import os
import ipdb
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def card_to_one_hot(card):
    if (card == []):
        return np.zeros[12]

    val = values.index(card[0])
    suit = suits.index(card[1])

    oh = np.zeros([12])
    oh[val] = 1
    oh[suit + 8] = 1

    return oh

def cards_to_one_hot(cards, n_cards=-1):
    if n_cards < 0:
        n_cards = len(cards)

    oh = np.zeros([12 * n_cards])
    for card_idx, card in enumerate(cards):
        oh[card_idx * 12:(card_idx + 1)*12] = card_to_one_hot(card)

    return oh

class ToepState:
    # state vector:
    # player hand, card 1, val, one-hot (x8)
    # player hand, card 1, suit, one-hot (x4)
    # player hand, card 2, val, one-hot (x8)
    # player hand, card 2, suit, one-hot (x4)
    # player hand, card 3, val, one-hot (x8)
    # player hand, card 3, suit, one-hot (x4)
    # player hand, card 4, val, one-hot (x8)
    # player hand, card 4, suit, one-hot (x4)
    # player table, card 1, val, one-hot (x8)
    # player table, card 1, suit, one-hot (x4)
    # player table, card 2, val, one-hot (x8)
    # player table, card 2, suit, one-hot (x4)
    # player table, card 3, val, one-hot (x8)
    # player table, card 3, suit, one-hot (x4)
    # player table, card 4, val, one-hot (x8)
    # player table, card 4, suit, one-hot (x4)
    # opponent table, card 1, val, one-hot (x8)
    # opponent table, card 1, suit, one-hot (x4)
    # opponent table, card 2, val, one-hot (x8)
    # opponent table, card 2, suit, one-hot (x4)
    # opponent table, card 3, val, one-hot (x8)
    # opponent table, card 3, suit, one-hot (x4)
    # opponent table, card 4, val, one-hot (x8)
    # opponent table, card 4, suit, one-hot (x4)
    # total 144
    def __init__(self, game):
        current_player_hand = game.players[game.current_player].hand
        table = [game.players[player_idx % len(game.players)].table for player_idx in range(game.current_player, game.current_player + len(game.players))]

        current_player_hand_vec = cards_to_one_hot(current_player_hand, 4)
        table_vecs = [cards_to_one_hot(player_table, 4) for player_table in table]
        
        self.state_vec = np.concatenate([current_player_hand_vec] + table_vecs)

class ToepQNetwork:
    # the ToepQNetwork takes a state s, and outputs the expected value of
    # Q(s,a) for each action. note that not all actions are always valid -
    # this is not explicitly modelled right now.
    def __init__(self):
        self.state_size = 144
        with tf.variable_scope('Input'):
            self.state_input = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)
            self.res_input = tf.reshape(self.state_input, shape=[-1, 1, 144])
        with tf.variable_scope('FeatureExtraction'):
            self.hidden_1 = slim.fully_connected(self.res_input, 256, activation_fn=tf.nn.relu, scope='FeatureExtraction/Hidden1')
            self.hidden_2 = slim.fully_connected(self.hidden_1,  512, activation_fn=tf.nn.relu, scope='FeatureExtraction/Hidden2')

        # split output into two streams; one for advantage and one for value
        with tf.variable_scope('AVSeparation'):
            self.advantage_hidden_nested, self.value_hidden_nested = tf.split(self.hidden_2, 2, 2)

        with tf.variable_scope('Advantage'):
            self.advantage_hidden = slim.flatten(self.advantage_hidden_nested)
            self.advantage = slim.fully_connected(self.advantage_hidden, 4, activation_fn=None)
        with tf.variable_scope('Value'):
            self.value_hidden = slim.flatten(self.value_hidden_nested)
            self.value     = slim.fully_connected(self.value_hidden, 1, activation_fn=None)

        with tf.variable_scope('Prediction'):
            self.Q_predict = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
            self.a_predict = tf.argmax(self.Q_predict, 1)

        with tf.variable_scope('TargetQ'):
            self.target_Q = tf.placeholder(shape=[None], dtype=tf.float32)
        with tf.variable_scope('Actions'):
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_one_hot = tf.one_hot(self.actions, 4, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Q_predict, self.actions_one_hot), axis=1)
        self.td_error = tf.square(self.target_Q - self.Q)
        self.loss = tf.reduce_mean(self.td_error)

        tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('Trainer'):
            self.trainer = tf.train.AdamOptimizer(learning_rate=0.00025)
            self.update_model = self.trainer.minimize(self.loss)

class ToepExperienceBuffer:
    def __init__(self, buffer_size=1000000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

def update_target_network_op(variables, tau):
    n_variables = len(variables)
    ops = []
    with tf.variable_scope('TargetNetworkUpdate'):
        for idx, var in enumerate(variables[0:n_variables // 2]):
            ops.append(variables[idx + n_variables // 2].assign(\
                (var.value() * tau) + ((1 - tau) * variables[idx + n_variables // 2].value())\
            ))
    return ops

class ToepQNetworkTrainer:
    def __init__(self):
        self.tau = 0.001
        self.start_e = 1
        self.end_e = 0.1
        self.e_steps = 1000000
        self.pretrain_steps = 50000
        self.n_episodes = 1000000
        self.batch_size = 32
        self.gamma = 0.9
        self.save_path = '/home/moos/jobhunt/practice/toepen/nets'
        self.log_path = '/home/moos/jobhunt/practice/toepen/logs'

        self.reset()

    def reset(self):
        tf.reset_default_graph()
        with tf.variable_scope('MainNet'):
            self.main_net = ToepQNetwork()
        with tf.variable_scope('TargetNet'):
            self.target_net = ToepQNetwork()

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        # setup the operations that update the target network
        trainable_vars = tf.trainable_variables()
        self.target_update_ops = update_target_network_op(trainable_vars, self.tau)

        # setup experience buffer
        self.experience_buffer = ToepExperienceBuffer()

        # we will decrease the amount of random actions taken gradually
        self.e = self.start_e
        self.step_drop = (self.start_e - self.end_e) / self.e_steps

        self.n_steps = 0
        self.r_list = [[], []]

        self.summary = tf.summary.merge_all()

        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        self.summary_writer = tf.summary.FileWriter(self.log_path, self.session.graph)

        self.session.run(self.init)

    def play_round(self, game, action):
        orig_player = game.current_player
        next_game = game.move(action)
        if next_game.winner != None:
            return next_game
        next_state = ToepState(next_game)
        while next_game.current_player != orig_player and next_game.winner == None:
            action = self.session.run(self.main_net.a_predict, feed_dict={self.main_net.state_input: [next_state.state_vec]})[0]
            valid_actions = next_game.get_valid_actions()
            if action not in valid_actions:
                action = valid_actions[np.random.randint(0, len(valid_actions))]
            next_game = next_game.move(action)
            next_state = ToepState(next_game)

        return next_game

    def play_round_random(self, game, action):
        orig_player = game.current_player
        next_game = game.move(action)
        if next_game.winner != None:
            return next_game
        next_state = ToepState(next_game)
        while next_game.current_player != orig_player and next_game.winner == None:
            valid_actions = next_game.get_valid_actions()
            action = valid_actions[np.random.randint(0, len(valid_actions))]
            next_game = next_game.move(action)
            next_state = ToepState(next_game)

        return next_game

    def test_against_random(self):
        n_games = 100
        n_actions = 0
        n_invalid_actions = 0
        n_wins = 0
        # games where player starts
        for episode_idx in range(0, n_games / 2):
            game = ToepGame()
            state = ToepState(game)

            while game.winner == None:
                # select action according to greedy policy
                valid_actions = game.get_valid_actions()
                action = self.session.run(self.main_net.a_predict, feed_dict={self.main_net.state_input: [state.state_vec]})[0]
                n_actions += 1
                if action not in valid_actions:
                    action = valid_actions[np.random.randint(0, len(valid_actions))]
                    n_invalid_actions += 1

                game_next = self.play_round_random(game, action)
                state_next = ToepState(game_next)
                if game_next.winner == game.current_player:
                    n_wins += 1
                    break
                elif game_next.winner != None:
                    break

                game = game_next
                state = ToepState(game)
        # games where player is second
        for episode_idx in range(0, n_games / 2):
            game = ToepGame()

            valid_actions = game.get_valid_actions()
            action = valid_actions[np.random.randint(0, len(valid_actions))]
            game = game.move(action)
            state = ToepState(game)

            while game.winner == None:
                # select action according to greedy policy
                valid_actions = game.get_valid_actions()
                action = self.session.run(self.main_net.a_predict, feed_dict={self.main_net.state_input: [state.state_vec]})[0]
                n_actions += 1
                if action not in valid_actions:
                    action = valid_actions[np.random.randint(0, len(valid_actions))]
                    n_invalid_actions += 1

                game_next = self.play_round_random(game, action)
                state_next = ToepState(game_next)
                if game_next.winner == game.current_player:
                    n_wins += 1
                    break
                elif game_next.winner != None:
                    break

                game = game_next
                state = ToepState(game)

        return [float(n_wins) / n_games, float(n_invalid_actions) / n_actions]

    def train_episode(self):
        game = ToepGame()
        state = ToepState(game)

        ep_buffer = ToepExperienceBuffer()

        round_idx = 0
        while game.winner == None:
            # select action according to eps-greedy policy
            valid_actions = game.get_valid_actions()
            if np.random.rand(1) < self.e or self.n_steps < self.pretrain_steps:
                action = valid_actions[np.random.randint(0, len(valid_actions))]
            else:
                action = self.session.run(self.main_net.a_predict, feed_dict={self.main_net.state_input: [state.state_vec]})[0]
                if action not in valid_actions:
                    action = valid_actions[np.random.randint(0, len(valid_actions))]

            game_next = self.play_round(game, action)
            state_next = ToepState(game_next)
            reward = 0
            if game_next.winner == game.current_player:
                reward = 1
            elif game_next.winner != None:
                reward = -1

            has_winner = game_next.winner != None

            self.n_steps += 1
            ep_buffer.add(np.reshape(np.array([state.state_vec, action, reward, state_next.state_vec, has_winner]), [1, 5]))

            if self.n_steps > self.pretrain_steps:
                if self.e > self.end_e:
                    self.e -= self.step_drop

                train_batch = self.experience_buffer.sample(self.batch_size)

                action_predict = self.session.run(self.main_net.a_predict, feed_dict={self.main_net.state_input: np.vstack(train_batch[:, 3])})
                value_predict = self.session.run(self.target_net.Q_predict, feed_dict={self.target_net.state_input: np.vstack(train_batch[:, 3])})

                end_multiplier = -(train_batch[:, 4] - 1)

                # was: double_Q = value_predict[range(self.batch_size), action_predict]
                double_Q = value_predict[range(self.batch_size), action_predict]
                target_Q = train_batch[:, 2] + (self.gamma * double_Q * end_multiplier)

                _ = self.session.run(self.main_net.update_model, \
                        feed_dict={self.main_net.state_input: np.vstack(train_batch[:, 0]),\
                                   self.main_net.target_Q: target_Q,\
                                   self.main_net.actions: train_batch[:, 1]})

                for op in self.target_update_ops:
                    self.session.run(op)

            game = game_next
            state = state_next
            round_idx += 1

        self.experience_buffer.add(ep_buffer.buffer)
        self.r_list[game.winner].append(1)
        self.r_list[game.winner ^ 1].append(-1)

        return game

    def train(self, n_episodes):
        with tf.device('/gpu:0'):
            for episode_idx in range(0, self.n_episodes):
                game = trainer.train_episode()

                if episode_idx % 100 == 0:
                    test_result = self.test_against_random()
                    print("Episode {0}, mean reward against random: {1}, ratio of invalid actions: {2}".format(episode_idx, test_result[0], test_result[1]))
                    #print(str(game))
                if episode_idx > 0 and episode_idx % 1000 == 0:
                    self.saver.save(self.session, os.path.join(self.save_path, "model_{0:02d}.ckpt".format(episode_idx)))
                    print("Saved model")

if __name__=="__main__":
    trainer = ToepQNetworkTrainer()

    trainer.train(20000)
