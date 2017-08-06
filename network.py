from game import *
import os
import ipdb
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from timeit import default_timer as timer

action_idx_to_name = {0: 0, 1: 1, 2: 2, 3: 3, 4: 't', 5: 'c', 6: 'f'}
action_name_to_idx = {0: 0, 1: 1, 2: 2, 3: 3, 't': 4, 'c': 5, 'f': 6}
n_actions = len(action_idx_to_name)

def card_to_one_hot(card):
    """Converts a single card, in the form of (val, suit), to a one-hot vector.
       The first 8 elements correspond to the value, the last 4 to the suit."""
    if (card == []):
        return np.zeros[12]

    val = values.index(card[0])
    suit = suits.index(card[1])

    oh = np.zeros([12])
    oh[val] = 1
    oh[suit + 8] = 1

    return oh

def cards_to_one_hot(cards, n_cards=-1):
    """Converts multiple cards to a concatenation of one-hot vectors."""
    if n_cards < 0:
        n_cards = len(cards)

    oh = np.zeros([12 * n_cards])
    for card_idx, card in enumerate(cards):
        oh[card_idx * 12:(card_idx + 1)*12] = card_to_one_hot(card)

    return oh

def actions_to_one_hot(actions):
    oh = np.zeros([7])
    for action in actions:
        oh[action_name_to_idx[action]] = 1

    return oh > 0

class ToepState:
    """Toep game state representation in form of feature vector.

       Format:
       player hand, card 1, val, one-hot (x8)
       player hand, card 1, suit, one-hot (x4)
       player hand, card 2, val, one-hot (x8)
       player hand, card 2, suit, one-hot (x4)
       player hand, card 3, val, one-hot (x8)
       player hand, card 3, suit, one-hot (x4)
       player hand, card 4, val, one-hot (x8)
       player hand, card 4, suit, one-hot (x4)
       player table, card 1, val, one-hot (x8)
       player table, card 1, suit, one-hot (x4)
       player table, card 2, val, one-hot (x8)
       player table, card 2, suit, one-hot (x4)
       player table, card 3, val, one-hot (x8)
       player table, card 3, suit, one-hot (x4)
       player table, card 4, val, one-hot (x8)
       player table, card 4, suit, one-hot (x4)
       opponent table, card 1, val, one-hot (x8)
       opponent table, card 1, suit, one-hot (x4)
       opponent table, card 2, val, one-hot (x8)
       opponent table, card 2, suit, one-hot (x4)
       opponent table, card 3, val, one-hot (x8)
       opponent table, card 3, suit, one-hot (x4)
       opponent table, card 4, val, one-hot (x8)
       opponent table, card 4, suit, one-hot (x4)
       stake (x1)
       betting_phase (x1)
       current player score (x1)
       opponent player score (x1)
       is action 1 valid, (x1)
       is action 2 valid, (x1)
       is action 3 valid, (x1)
       is action 4 valid, (x1)
       is action 5 valid, (x1)
       is action 6 valid, (x1)
       is action 7 valid, (x1)
       total 155"""
    STATE_SIZE = 155
    def __init__(self, game):
        current_player_hand = game.players[game.phase.current_player].hand
        table = [game.players[player_idx % len(game.players)].table for player_idx in range(game.phase.current_player, game.phase.current_player + len(game.players))]

        current_player_hand_vec = cards_to_one_hot(current_player_hand, 4)
        table_vecs = [cards_to_one_hot(player_table, 4) for player_table in table]

        single_values_vec = [game.stake, 1 if game.phase == game.betting_phase else 0,
                             game.players[game.phase.current_player].score, game.players[game.phase.current_player ^ 1].score]
        
        valid_actions = game.get_valid_actions()
        all_actions = [0, 1, 2, 3, 't', 'c', 'f']
        action_vec = np.array([1 if action in valid_actions else 0 for action in all_actions])

        self.state_vec = np.concatenate([current_player_hand_vec] + table_vecs + [np.array(single_values_vec)] + [action_vec])

def state_vec_to_game(state_vec, n_players=2):
    """Converts a Toep game state vector (from ToepState) back to a ToepGame instance.
       The state vector does not have access to the opponent's hand, and this is thus left empty."""
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
    game.players[0].score = int(state_vec[48 * (n_players + 1) + 2])
    game.players[1].score = int(state_vec[48 * (n_players + 1) + 3])

    return game

class ToepQNetwork:
    """A Deep Q-Network for learning to play Toepen.

      The ToepQNetwork takes a state vector s (a ToepState), and outputs the expected value of
      Q(s,a) for each action. It employs both Double Q-Learning (van Hasselt et al. 2016)
      and Dueling DQN (Wang et al. 2016).
    """
    def __init__(self):
        self.state_size = ToepState.STATE_SIZE
        with tf.variable_scope('Input'):
            self.state_input = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)
            self.res_input = tf.reshape(self.state_input, shape=[-1, 1, self.state_size])
            self.valid_actions = tf.placeholder(shape=[None, n_actions], dtype=tf.bool)
            self.res_valid_actions = tf.reshape(self.valid_actions, shape=[-1, 1, n_actions])
        with tf.variable_scope('FeatureExtraction'):
            self.hidden_1 = slim.fully_connected(self.res_input, 128, activation_fn=None, scope='FeatureExtraction/Hidden_1')
            self.elu_1    = tf.nn.elu(self.hidden_1, 'elu_1')

            self.hidden_2 = slim.fully_connected(self.elu_1,      96,  activation_fn=None, scope='FeatureExtraction/Hidden_2')
            self.elu_2    = tf.nn.elu(self.hidden_2, 'elu_2')

            self.hidden_3 = slim.fully_connected(self.elu_2,      64,  activation_fn=None, scope='FeatureExtraction/Hidden_3')
            self.elu_3    = tf.nn.elu(self.hidden_3, 'elu_3')

        # split output into two streams; one for advantage and one for value (Dueling DQN)
        with tf.variable_scope('AVSeparation'):
            self.advantage_hidden_nested, self.value_hidden_nested = tf.split(self.elu_3, 2, 2)

        with tf.variable_scope('Advantage'):
            self.advantage_hidden = slim.flatten(self.advantage_hidden_nested)
            self.advantage = slim.fully_connected(self.advantage_hidden, n_actions, activation_fn=None)
        with tf.variable_scope('Value'):
            self.value_hidden = slim.flatten(self.value_hidden_nested)
            self.value     = slim.fully_connected(self.value_hidden, 1, activation_fn=None)

        with tf.variable_scope('Prediction'):
            """
            To combine value and advantage we use the averaging operator as shown in Wang et al. 2016,
            which stabilizes the optimization
            """
            self.Q_predict = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True))

            dummy_min = tf.constant(-100.0, shape=[128, n_actions])
            self.Q_valid = tf.where(self.valid_actions, self.Q_predict, dummy_min)
            self.a_predict = tf.arg_max(self.Q_valid, 1)

        with tf.variable_scope('TargetQ'):
            """
            Target Q generation is decoupled from the action generation network, and is passed
            in from the target network
            (Double DQN, van Hasselt et al. 2016)
            """
            self.target_Q = tf.placeholder(shape=[None], dtype=tf.float32)
        with tf.variable_scope('Actions'):
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_one_hot = tf.one_hot(self.actions, n_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Q_predict, self.actions_one_hot), axis=1)
        self.td_error = tf.square(self.target_Q - self.Q)
        self.loss = tf.reduce_mean(self.td_error)

        tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('Trainer'):
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.polynomial_decay(1e-4, self.global_step, 500000, 1e-6, power=0.5)

            self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.update_model = self.trainer.minimize(self.loss, global_step=self.global_step)

class ToepExperienceBuffer:
    """
    Stores experiences of playing the game, together with rewards gained,
    which are used to train the network (experience replay).
    """
    def __init__(self, buffer_size=1000000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 6])

def update_target_network_op(variables, tau):
    """
    This function updates the trainable variables of the target network
    slowly in the direction of the main network (Double DQN).
    """
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
    """
    This returns the action that has the highest value according to Q, but is also valid.
    """
    value_sorted_actions = sorted(range(0, len(Q)), key=lambda x: -Q[x])
    valid_value_sorted_actions = [action for action in value_sorted_actions if action_idx_to_name[action] in valid_actions]
    action = action_idx_to_name[valid_value_sorted_actions[0]]

    return action

def softmax(x, t):
    x_t = x / t
    x_shift = x_t - np.max(x_t)
    e_x = np.exp(x_shift)
    return e_x / np.sum(e_x)

class ToepQNetworkTrainer:
    def __init__(self):
        self.tau = 0.001 # rate at which the target network is moved in direction of the main network

        self.pretrain_steps = 100000
        self.n_episodes = 100000
        self.batch_size = 128
        self.gamma = 1
        self.start_boltzmann_temp = 1
        self.end_boltzmann_temp = 0.01
        self.boltzmann_steps = 500000
        self.save_path = 'nets'
        self.log_path = 'logs'
        self.load_model = True

        self.reset()

    def reset(self):
        tf.reset_default_graph()
        # Two networks - action selection and target value generation are decoupled
        # (Double DQN, van Hasselt et al. 2016)
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

        # we employ Boltzmann exploration
        self.boltzmann_temp = self.start_boltzmann_temp
        self.boltzmann_step = (self.start_boltzmann_temp - self.end_boltzmann_temp) / self.boltzmann_steps

        self.n_steps = 0

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
            self.boltzmann_temp = self.end_boltzmann_temp
            self.n_steps = self.boltzmann_steps

    def play_round(self, game, action):
        """
        Continues the game from its current position for one round,
        until it is once more the current player's turn.
        """
        orig_player = game.phase.current_player

        next_game = game.move(action)
        next_player_game = next_game.copy()

        next_state = ToepState(next_game)
        while not next_game.players[orig_player].did_fold and next_game.phase.current_player != orig_player and next_game.get_winner() == None:
            [action, _] = self.get_action(next_game, next_state)
            
            next_game = next_game.move(action)
            next_state = ToepState(next_game)

        winner = next_game.get_winner()
        reward = 0
        if winner != None:
            if next_game.get_winner() == orig_player:
                reward = 1
            else:
                reward = -1

        return [next_game, next_player_game, reward]

    def test_rounds(self, n_games=100):
        games = [ToepGame() for _ in range(0, n_games)]
        n_actions = 0
        n_invalid_actions = 0
        overall_reward_first = 0
        overall_reward_second = 0
        # games where player starts
        for episode_idx in range(0, n_games):
            game = games[episode_idx].copy()
            state = ToepState(game)

            while game.get_winner() == None:
                # select action according to greedy policy
                [action, _] = self.get_action(game, state)

                [round_next, _, _] = self.play_round(game, action)

                game = round_next
                state = ToepState(game)

            if game.get_winner() == 0:
                overall_reward_first += 1
            else:
                overall_reward_second += 1

        return [float(overall_reward_first) / n_games, float(overall_reward_second) / n_games]

    def get_action(self, game, state):#, valid_only=False):
        valid_actions = game.get_valid_actions()
        valid_actions_oh = actions_to_one_hot(valid_actions)

        Q = self.session.run(self.main_net.Q_predict, feed_dict={self.main_net.state_input: [state.state_vec],
                                                                 self.main_net.valid_actions: [valid_actions_oh]})[0]

        valid_action_indices = [action_name_to_idx[action] for action in valid_actions]
        Q_valid = np.array([Q[idx] for idx in valid_action_indices])
        Q_valid_softmax = softmax(Q_valid, self.boltzmann_temp)
        action = valid_action_indices[np.random.choice(np.arange(0, len(Q_valid_softmax)), p=Q_valid_softmax)]
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

        ep_loss = 0
        ep_steps = 0
        while game.get_winner() == None:
            # if pre-training, we randomly select our actions
            if self.n_steps < self.pretrain_steps:
                valid_actions = game.get_valid_actions()
                action = random.choice(valid_actions)
                Q = "NA"
            else:
                [action, Q] = self.get_action(game, state)

            if verbose:
                print("----------------------------------------------")
                print("game: \n{0}".format(str(game)))
                print("Q: {0}".format(Q))
                print("action: {0}".format(action))

            [round_next, game_next, reward] = self.play_round(game, action)
            if verbose:
                print("reward: {0}".format(reward))
                print("next game: \n{0}".format(str(game_next)))

            state_next = ToepState(round_next)
            valid_actions_next = actions_to_one_hot(round_next.get_valid_actions())

            self.n_steps += 1
            ep_buffer.add(np.reshape(np.array([state.state_vec, action_name_to_idx[action], reward, state_next.state_vec, valid_actions_next, reward == 1 or reward == -1]), [1, 6]))

            if self.n_steps > self.pretrain_steps:
                if self.boltzmann_temp > self.end_boltzmann_temp:
                    self.boltzmann_temp -= self.boltzmann_step

                train_batch = self.experience_buffer.sample(self.batch_size)

                action_predict     = self.session.run(self.main_net.a_predict,   feed_dict={self.main_net.state_input:     np.vstack(train_batch[:, 3]),\
                                                                                            self.main_net.valid_actions:   np.vstack(train_batch[:, 4])})
                target_val_predict = self.session.run(self.target_net.Q_predict, feed_dict={self.target_net.state_input:   np.vstack(train_batch[:, 3]),\
                                                                                            self.target_net.valid_actions: np.vstack(train_batch[:, 4])})

                # end states are treated differently - there is no future reward
                end_multiplier = -(train_batch[:, 5] - 1)

                double_Q = target_val_predict[range(self.batch_size), action_predict] # need to specify range exactly for TF to fully know shape
                target_Q = train_batch[:, 2] + (self.gamma * double_Q * end_multiplier)

                [_, loss] = self.session.run([self.main_net.update_model, self.main_net.loss], \
                                   feed_dict={self.main_net.state_input: np.vstack(train_batch[:, 0]),\
                                              self.main_net.target_Q: target_Q,\
                                              self.main_net.actions: train_batch[:, 1]})

                ep_steps += 1
                ep_loss += loss

                for op in self.target_update_ops:
                    self.session.run(op)

            game = game_next
            state = ToepState(game_next)

        self.experience_buffer.add(ep_buffer.buffer)

        return [game, ep_loss / ep_steps if ep_steps > 0 else 0]

    def train(self):
        with tf.device('/gpu:0'):
            for episode_idx in range(0, self.n_episodes):
                verbose = episode_idx % 100 == 0
                #verbose = False
                [game, ep_loss] = trainer.train_episode(verbose)

                if episode_idx % 100 == 0:
                    test_result = self.test_rounds(10)
                    print("Episode {0} P1 {1} P2 {2} L {3} BT {4}".format(episode_idx, test_result[0], test_result[1], ep_loss, self.boltzmann_temp))
                if episode_idx > 0 and episode_idx % 2000 == 0:
                    self.saver.save(self.session, os.path.join(self.save_path, "model_{0:02d}.ckpt".format(episode_idx)))
                    print("Saved model")

if __name__=="__main__":
    trainer = ToepQNetworkTrainer()

    trainer.train()
