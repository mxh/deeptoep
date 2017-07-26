from game import *
from network import *

class ConsoleToepHandler:
    def __init__(self):
        pass

    def get_move(self, game):
        print(game.str_hidden())
        print
        valid_actions = game.get_valid_actions()
        action = None
        while action not in [str(a) for a in valid_actions]:
            action = raw_input("Action [{0}]: ".format(", ".join([str(a) for a in valid_actions])))
        print

        if action.isdigit():
            action = int(action)
        return action

class NetworkToepHandler:
    def __init__(self, network, session):
        self.network = network
        self.session = session

    def get_move(self, game):
        state = ToepState(game)

        [action, Q, advantage] = self.session.run([self.network.a_predict, self.network.Q_predict, self.network.advantage], feed_dict={self.network.state_input: [state.state_vec]})
        action = action_idx_to_name[action[0]]

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
            [next_game, reward, player_finished, game_finished] = self.game.move(\
                self.handlers[self.game.phase.current_player].get_move(self.game)\
            )

            self.game = next_game

        print("Game finished (winner: {0}). Final state:".format(self.game.get_winner()))
        print(str(self.game))

if __name__=="__main__":
    trainer = ToepQNetworkTrainer()

    server = ToepGameServer()

    manual_handler = ConsoleToepHandler()
    network_handler = NetworkToepHandler(trainer.main_net, trainer.session)

    handlers = [manual_handler, network_handler]

    while True:
        server.init_game()
        random.shuffle(handlers)

        server.set_player_handler(0, handlers[0])
        server.set_player_handler(1, handlers[1])

        server.start()
