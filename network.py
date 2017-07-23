from game import *
import os
import ipdb
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from timeit import default_timer as timer

action_idx_to_name = {0: 0, 1: 1, 2: 2, 3: 3, 4: 't', 5: 'c', 6: 'f'}
action_name_to_idx = {0: 0, 1: 1, 2: 2, 3: 3, 't': 4, 'c': 5, 'f': 6}

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
    # stake (x1)
    # betting_phase (x1)
    # is action 1 valid, (x1)
    # is action 2 valid, (x1)
    # is action 3 valid, (x1)
    # is action 4 valid, (x1)
    # is action 5 valid, (x1)
    # is action 6 valid, (x1)
    # is action 7 valid, (x1)
    # total 153
    def __init__(self, game):
        current_player_hand = game.players[game.phase.current_player].hand
        table = [game.players[player_idx % len(game.players)].table for player_idx in range(game.phase.current_player, game.phase.current_player + len(game.players))]

        current_player_hand_vec = cards_to_one_hot(current_player_hand, 4)
        table_vecs = [cards_to_one_hot(player_table, 4) for player_table in table]

        valid_actions = game.get_valid_actions()
        all_actions = [0, 1, 2, 3, 't', 'c', 'f']
        action_vec = np.array([1 if action in valid_actions else 0 for action in all_actions])
        
        self.state_vec = np.concatenate([current_player_hand_vec] + table_vecs + [np.array([game.stake, 1 if game.phase == game.betting_phase else 0])] + [action_vec])

def state_vec_to_game(state_vec, n_players=2):
    game = ToepGame(n_players)

    for player in game.players:
        player.table = []
        player.hand = []

    for hand_card_idx in range(0, 4):
        start_idx = hand_card_idx * 12
        val_idx = np.where(state_vec[start_idx + 0:start_idx + 8] == 1)[0]
        if len(val_idx) != 0:
            val = values[val_idx[0]]
            suit = suits[np.where(state_vec[start_idx + 8:start_idx + 12] == 1)[0][0]]
        else:
            break

        game.players[0].hand.append((val, suit))

    game.players[0].hand = sort_cards(game.players[0].hand)

    for player_idx in range(0, n_players):
        for table_card_idx in range(0, 4):
            start_idx = (player_idx + 1) * 48 + table_card_idx * 12
            val_idx = np.where(state_vec[start_idx + 0:start_idx + 8] == 1)[0]
            if len(val_idx) != 0:
                val = values[val_idx[0]]
                suit = suits[np.where(state_vec[start_idx + 8:start_idx + 12] == 1)[0][0]]
            else:
                break
            game.players[player_idx].table.append((val, suit))

    game.stake = int(state_vec[48 * (n_players + 1)])
    if state_vec[48 * (n_players + 1) + 1]:
        game.phase = game.betting_phase
        game.betting_phase.current_player = 0

    return game

class ToepQNetwork:
    # the ToepQNetwork takes a state s, and outputs the expected value of
    # Q(s,a) for each action. note that not all actions are always valid -
    # this is not explicitly modelled right now.
    def __init__(self):
        self.state_size = 153
        self.training = True
        with tf.variable_scope('Input'):
            self.state_input = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)
            self.res_input = tf.reshape(self.state_input, shape=[-1, 1, self.state_size])
        with tf.variable_scope('FeatureExtraction'):
            self.hidden_1   = slim.fully_connected(self.res_input, 256, activation_fn=None, scope='FeatureExtraction/Hidden1')
            #self.bn_1       = slim.batch_norm(self.hidden_1, center=True, scale=True, is_training=self.training, scope='FeatureExtraction/BN1')
            self.relu_1     = tf.nn.relu(self.hidden_1, 'relu')
            self.hidden_2   = slim.fully_connected(self.relu_1,    256, activation_fn=None, scope='FeatureExtraction/Hidden2')
            #self.bn_2       = slim.batch_norm(self.hidden_2, center=True, scale=True, is_training=self.training, scope='FeatureExtraction/BN2')
            self.relu_2     = tf.nn.relu(self.hidden_2, 'relu')

        # split output into two streams; one for advantage and one for value
        with tf.variable_scope('AVSeparation'):
            self.advantage_hidden_nested, self.value_hidden_nested = tf.split(self.relu_2, 2, 2)

        with tf.variable_scope('Advantage'):
            self.advantage_hidden = slim.flatten(self.advantage_hidden_nested)
            self.advantage = slim.fully_connected(self.advantage_hidden, 7, activation_fn=None)
        with tf.variable_scope('Value'):
            self.value_hidden = slim.flatten(self.value_hidden_nested)
            self.value     = slim.fully_connected(self.value_hidden, 1, activation_fn=None)

        with tf.variable_scope('Prediction'):
            self.Q_predict = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
            self.a_predict = tf.argmax(self.Q_predict, 1) # we will phase this out, as this does not take into account action validity

        with tf.variable_scope('TargetQ'):
            self.target_Q = tf.placeholder(shape=[None], dtype=tf.float32)
        with tf.variable_scope('Actions'):
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_one_hot = tf.one_hot(self.actions, 7, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Q_predict, self.actions_one_hot), axis=1)
        self.td_error = tf.square(self.target_Q - self.Q)
        self.loss = tf.reduce_mean(self.td_error)

        tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('Trainer'):
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.polynomial_decay(1e-5, self.global_step, 100000, 1e-6, power=0.5)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.update_model = self.trainer.minimize(self.loss, global_step=self.global_step)

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

def get_number_of_variables():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def get_highest_valid_action(Q, valid_actions):
    value_sorted_actions = sorted(range(0, len(Q)), key=lambda x: -Q[x])
    valid_value_sorted_actions = [action for action in value_sorted_actions if action_idx_to_name[action] in valid_actions]
    action = action_idx_to_name[valid_value_sorted_actions[0]]

    return action

def softmax(x, t):
    e_x = np.exp(x / t)
    return e_x / np.sum(e_x)

class ToepQNetworkTrainer:
    def __init__(self):
        self.tau = 0.01
        self.start_e = 1
        self.end_e = 0.1
        self.e_steps = 100000
        self.pretrain_steps = 10000
        self.n_episodes = 10000000
        self.batch_size = 128
        self.gamma = 0.9
        self.save_path = '/home/moos/jobhunt/practice/toepen/nets'
        self.log_path = '/home/moos/jobhunt/practice/toepen/logs'
        self.load_model = False

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

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        self.summary_writer = tf.summary.FileWriter(self.log_path, self.session.graph)

        self.session.run(self.init)

        print("Number of parameters:  {0}".format(get_number_of_variables()))

        if self.load_model:
            print("Loading model...")
            ckpt = tf.train.get_checkpoint_state(self.save_path)
            self.saver.restore(self.session, ckpt.model_checkpoint_path)

            self.e = self.end_e#self.e - (self.step_drop * self.pretrain_steps)
            self.step_drop = 0

    def play_round(self, game, action):
        orig_player = game.phase.current_player
        [next_game, reward, player_finished, game_finished] = game.move(action)
        next_player_game = next_game.copy()
        if player_finished:
            return [next_game, next_player_game, reward, player_finished, game_finished]

        next_state = ToepState(next_game)
        while next_game.phase.current_player != orig_player and not game_finished:
            [action, _] = self.get_action(next_game, next_state, True)

            [next_game, _, _, game_finished] = next_game.move(action)
            next_state = ToepState(next_game)

        if game_finished:
            if next_game.get_winner() == orig_player:
                reward = next_game.stake
                if np.all([next_game.players[player_idx].did_fold for player_idx in range(0, len(next_game.players)) if player_idx != orig_player]):
                    reward -= 1
            else:
                reward = -next_game.stake

        return [next_game, next_player_game, reward, game_finished, game_finished]

    def play_round_random(self, game, action):
        orig_player = game.phase.current_player
        [next_game, reward, player_finished, game_finished] = game.move(action)
        if player_finished:
            return [next_game, reward, player_finished, game_finished]

        next_state = ToepState(next_game)
        while next_game.phase.current_player != orig_player and not game_finished:
            valid_actions = next_game.get_valid_actions()
            action = valid_actions[np.random.randint(0, len(valid_actions))]
            [next_game, _, _, game_finished] = next_game.move(action)
            next_state = ToepState(next_game)

        if game_finished:
            if next_game.get_winner() == orig_player:
                reward = next_game.stake
            else:
                reward = -next_game.stake

        return [next_game, reward, game_finished, game_finished]

    def test_against_random(self):
        n_games = 100
        games = [ToepGame() for _ in range(0, n_games)]
        n_actions = 0
        n_invalid_actions = 0
        overall_reward_first = 0
        overall_reward_second = 0
        # games where player starts
        for episode_idx in range(0, n_games):
            game = games[episode_idx].copy()
            state = ToepState(game)

            game_finished = False
            while not game_finished:
                # select action according to greedy policy
                [action, _] = self.get_action(game, state, True)

                [game_next, reward, player_finished, game_finished] = self.play_round_random(game, action)
                state_next = ToepState(game_next)
                if player_finished:
                    overall_reward_first += reward
                    break

                game = game_next
                state = ToepState(game)

            if episode_idx == 0:
                print(str(game_next))

        # games where player is second
        for episode_idx in range(0, n_games):
            game = games[episode_idx].copy()

            valid_actions = game.get_valid_actions()
            action = valid_actions[np.random.randint(0, len(valid_actions))]
            [game, _, _, _] = game.move(action)
            state = ToepState(game)

            game_finished = False
            while not game_finished:
                # select action according to greedy policy
                [action, _] = self.get_action(game, state, True)

                [game_next, reward, player_finished, game_finished] = self.play_round_random(game, action)
                state_next = ToepState(game_next)
                if player_finished:
                    overall_reward_second += reward
                    break

                game = game_next
                state = ToepState(game)

        return [float(overall_reward_first) / n_games, float(overall_reward_second) / n_games]

    def test_against_self(self):
        n_games = 100
        games = [ToepGame() for _ in range(0, n_games)]
        n_actions = 0
        n_invalid_actions = 0
        overall_reward_first = 0
        overall_reward_second = 0
        # games where player starts
        for episode_idx in range(0, n_games):
            game = games[episode_idx].copy()
            state = ToepState(game)

            game_finished = False
            while not game_finished:
                # select action according to greedy policy
                [action, _] = self.get_action(game, state, True)

                [round_next, game_next, reward, player_finished, game_finished] = self.play_round(game, action)
                state_next = ToepState(round_next)
                if player_finished:
                    overall_reward_first += reward
                    break

                game = round_next
                state = ToepState(game)

            if episode_idx == 0:
                print(str(game_next))

        # games where player is second
        for episode_idx in range(0, n_games):
            game = games[episode_idx].copy()

            valid_actions = game.get_valid_actions()
            action = valid_actions[np.random.randint(0, len(valid_actions))]
            [game, _, _, _] = game.move(action)
            state = ToepState(game)

            game_finished = False
            while not game_finished:
                # select action according to greedy policy
                [action, _] = self.get_action(game, state, True)

                [round_next, game_next, reward, player_finished, game_finished] = self.play_round(game, action)
                state_next = ToepState(round_next)
                if player_finished:
                    overall_reward_second += reward
                    break

                game = round_next
                state = ToepState(game)

        return [float(overall_reward_first) / n_games, float(overall_reward_second) / n_games]

    def get_action(self, game, state, valid_only=False):
        valid_actions = game.get_valid_actions()

        Q = self.session.run(self.main_net.Q_predict, feed_dict={self.main_net.state_input: [state.state_vec]})[0]

        if valid_only:
            valid_action_indices = [action_name_to_idx[action] for action in valid_actions]
            Q_valid = np.array([Q[idx] for idx in valid_action_indices])
            Q_valid_softmax = softmax(Q_valid, 5)
            action = valid_action_indices[np.random.choice(np.arange(0, len(Q_valid_softmax)), p=Q_valid_softmax)]
            action = action_idx_to_name[action]
        else:
            Q_softmax = softmax(Q, 5)
            action = np.random.choice(np.arange(0, len(Q_softmax)), p=Q_softmax)
            action = action_idx_to_name[action]

        return [action, Q]

    def train_episode(self, verbose=False):
        if verbose:
            print("----------------------------------------------")
            print("----------------------------------------------")
        game = ToepGame()
        state = ToepState(game)

        ep_buffer = ToepExperienceBuffer()

        round_idx = 0
        game_finished = False
        while game.get_winner() == None:
            # select action according to eps-greedy policy
            [action, Q] = self.get_action(game, state, False)

            if verbose:
                print("----------------------------------------------")
                print("game: {0}".format(str(game)))
                print("Q: {0}".format(Q))
                print("action: {0}".format(action))

            [round_next, game_next, reward, player_finished, game_finished] = self.play_round(game, action)
            if verbose:
                print("chosen action: {0}".format(action))
                print("reward: {0}".format(reward))
                print("next game: {0}".format(str(game_next)))

            state_next = ToepState(round_next)

            self.n_steps += 1
            ep_buffer.add(np.reshape(np.array([state.state_vec, action_name_to_idx[action], reward, state_next.state_vec, player_finished]), [1, 5]))

            if self.n_steps > self.pretrain_steps:
                
                if self.e > self.end_e:
                    self.e -= self.step_drop

                train_batch = self.experience_buffer.sample(self.batch_size)

                action_predict = self.session.run(self.main_net.a_predict, feed_dict={self.main_net.state_input:   np.vstack(train_batch[:, 3])})
                #main_val_predict  = self.session.run(self.main_net.Q_predict, feed_dict={self.main_net.state_input:   np.vstack(train_batch[:, 3])})
                #value_sorted_actions = np.argsort(-main_val_predict)
                #valid_value_sorted_actions = np.full([self.batch_size, 7], False)
                #for row_idx in range(0, self.batch_size):
                #    valid_value_sorted_actions[row_idx, :] = np.isin(value_sorted_actions[row_idx, :], train_batch[row_idx, 4])

                #action_predict = value_sorted_actions[range(self.batch_size), np.argmax(valid_value_sorted_actions, axis=1)]

                target_val_predict = self.session.run(self.target_net.Q_predict, feed_dict={self.target_net.state_input: np.vstack(train_batch[:, 3])})

                end_multiplier = -(train_batch[:, 4] - 1)

                # was: double_Q = value_predict[range(self.batch_size), action_predict]
                double_Q = target_val_predict[range(self.batch_size), action_predict]
                target_Q = train_batch[:, 2] + (self.gamma * double_Q * end_multiplier)

                _ = self.session.run(self.main_net.update_model, \
                        feed_dict={self.main_net.state_input: np.vstack(train_batch[:, 0]),\
                                   self.main_net.target_Q: target_Q,\
                                   self.main_net.actions: train_batch[:, 1]})

                for op in self.target_update_ops:
                    self.session.run(op)

            game = game_next
            state = ToepState(game_next)

        self.experience_buffer.add(ep_buffer.buffer)
        self.r_list[game.get_winner()].append(1)
        self.r_list[game.get_winner() ^ 1].append(-1)

        return game

    def train(self, n_episodes):
        with tf.device('/gpu:0'):
            for episode_idx in range(0, self.n_episodes):
                verbose = episode_idx % 100 == 0
                #verbose = False
                game = trainer.train_episode(verbose)

                if episode_idx % 100 == 0:
                    #test_result = self.test_against_random()
                    test_result = self.test_against_self()
                    print("Episode {0}  MR1 {1}  MR2 {2}  E {3}".format(episode_idx, test_result[0], test_result[1], self.e))
                if episode_idx > 0 and episode_idx % 5000 == 0:
                    self.saver.save(self.session, os.path.join(self.save_path, "model_{0:02d}.ckpt".format(episode_idx)))
                    print("Saved model")

if __name__=="__main__":
    trainer = ToepQNetworkTrainer()

    trainer.train(20000)
