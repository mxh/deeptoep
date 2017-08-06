from game import *
from network import *

class ConsoleToepHandler:
    def __init__(self):
        self.name = "Console"
        pass

    def get_move(self, game):
        valid_actions = game.get_valid_actions()
        action = None
        while action not in [str(a) for a in valid_actions]:
            action = raw_input("Action [{0}]: ".format(", ".join([str(a) for a in valid_actions])))

        if action.isdigit():
            action = int(action)
        return action

class NetworkToepHandler:
    def __init__(self, network, session):
        self.network = network
        self.session = session

        self.name = "Network"

    def get_move(self, game):
        state = ToepState(game)
        valid_actions = game.get_valid_actions()
        valid_actions_oh = actions_to_one_hot(valid_actions)

        #print("".join([str(int(x)) for x in state.state_vec]))
        Q = self.session.run(self.network.Q_predict, feed_dict={self.network.state_input: [state.state_vec], self.network.valid_actions: [valid_actions_oh]})[0]
        #print(Q)
        valid_action_indices = [action_name_to_idx[action] for action in valid_actions]
        Q_valid = np.array([Q[idx] for idx in valid_action_indices])
        Q_valid_softmax = softmax(Q_valid, 0.01)
        action = valid_action_indices[np.random.choice(np.arange(0, len(Q_valid_softmax)), p=Q_valid_softmax)]
        action = action_idx_to_name[action]

        return action

class ToepGameServer:
    def __init__(self, game=None):
        self.init_game(game);

        self.handlers = {}

    def init_game(self, game=None):
        if game is not None:
            self.game = game
        else:
            self.game = ToepGame()

    def set_player_handler(self, player_idx, handler):
        self.handlers[player_idx] = handler

    def start(self):
        if not len(self.handlers) == len(self.game.players):
            print("Error: cannot start game, need {0} handlers but have {1}".format(len(self.game.players), len(self.handlers)))
            return

        while self.game.get_winner() == None:
            #print("Player {0} - {1}:".format(self.game.phase.current_player, self.handlers[self.game.phase.current_player].name))
            #print(self.game.str_hidden())
            next_game = self.game.move(\
                self.handlers[self.game.phase.current_player].get_move(self.game)\
            )

            self.game = next_game

        #print("Game finished (winner: {0}). Final state:".format(self.game.get_winner()))
        #print(str(self.game))

        return self.game.get_winner()

if __name__=="__main__":
    trainer = ToepQNetworkTrainer()

    server = ToepGameServer()

    network_1_handler = NetworkToepHandler(trainer.main_net, trainer.session)
    network_2_handler = RandomHandler()

    handlers = [network_1_handler, network_2_handler]

    winner_freq = [0, 0]
    n_games = 1000
    for game_idx in range(0, n_games / 2):
        print("Game {0}/{1}".format(game_idx + 1, n_games))
        server.init_game()

        server.set_player_handler(0, handlers[0])
        server.set_player_handler(1, handlers[1])

        winner = server.start()
        winner_freq[winner] += 1
    for game_idx in range(0, n_games / 2):
        print("Game {0}/{1}".format(n_games / 2 + game_idx + 1, n_games))
        server.init_game()

        server.set_player_handler(1, handlers[0])
        server.set_player_handler(0, handlers[1])

        winner = server.start()
        winner_freq[winner ^ 1] += 1

    print("Win stats:")
    for handler_idx, handler in enumerate(handlers):
        print("{0} - {1}: {2:.2f}%".format(handler_idx, handler.name, 100.0 * winner_freq[handler_idx] / float(n_games)))
