from game import *
from network import *

from progressbar import ProgressBar

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

class RandomHandler:
    def __init__(self):
        self.name = "Random"

    def get_move(self, game):
        valid_actions = game.get_valid_actions()
        return random.choice(valid_actions)

class RandomNoFoldHandler:
    def __init__(self):
        self.name = "RandomNoFold"

    def get_move(self, game):
        valid_actions = [action for action in game.get_valid_actions() if action != "f"]
        return random.choice(valid_actions)

class LowestCardHandler:
    def __init__(self):
        self.name = "LowestCard"

    def get_move(self, game):
        valid_actions = game.get_valid_actions()
        card_plays = [action for action in valid_actions if action in [0, 1, 2, 3]]

        if len(card_plays) > 0:
            action = min(card_plays)
        else:
            action = random.choice(valid_actions)

        return action

class LowestCardNoFoldHandler:
    def __init__(self):
        self.name = "LowestCardNoFold"

    def get_move(self, game):
        valid_actions = game.get_valid_actions()
        card_plays = [action for action in valid_actions if action in [0, 1, 2, 3]]

        if len(card_plays) > 0:
            action = min(card_plays)
        else:
            action = random.choice([action for action in valid_actions if action != "f"])

        return action

class HighestCardHandler:
    def __init__(self):
        self.name = "HighestCard"

    def get_move(self, game):
        if game.phase == game.betting_phase:
            return random.choice(game.get_valid_actions())
        else:
            if game.game_phase.card_to_beat != None:
                valid_actions = [action for action in game.get_valid_actions() if action in [0, 1, 2, 3]]
                if len(valid_actions) > 0:
                    if game.players[game.phase.current_player].hand[valid_actions[0]][1] == game.phase.card_to_beat[0][1]:
                        return valid_actions[-1]
                    else:
                        return valid_actions[0]
                else:
                    return random.choice(game.get_valid_actions())
            else:
                return 0 # lowest card

class HighestCardNoFoldHandler:
    def __init__(self):
        self.name = "HighestCardNoFold"

    def get_move(self, game):
        if game.phase == game.betting_phase:
            return random.choice([action for action in game.get_valid_actions() if action != "f"])
        else:
            if game.game_phase.card_to_beat != None:
                valid_actions = [action for action in game.get_valid_actions() if action in [0, 1, 2, 3]]
                if len(valid_actions) > 0:
                    if game.players[game.phase.current_player].hand[valid_actions[0]][1] == game.phase.card_to_beat[0][1]:
                        return valid_actions[-1]
                    else:
                        return valid_actions[0]
                else:
                    return random.choice(game.get_valid_actions())
            else:
                return 0 # lowest card

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

def print_win_stats(win_stats, handlers):
    max_len = max([len(handler.name) for handler in handlers])
    top_row = "{0}{1}".format(' ' * (max_len + 1), "".join([handler.name + (max_len - len(handler.name) + 1) * ' ' for handler in handlers]))
    print(top_row)

    for handler in handlers:
        row = "{0}{1}".format(handler.name + (max_len - len(handler.name) + 1) * ' ', "".join(["{0:.2f}".format(win_stats[(handler.name, other.name)]).ljust(max_len + 1) for other in handlers]))
        print(row)

if __name__=="__main__":
    # setup handlers

    trainer = ToepQNetworkTrainer()
    network_handler = NetworkToepHandler(trainer.main_net, trainer.session)

    random_handler = RandomHandler()
    random_nf_handler = RandomNoFoldHandler()

    lowest_card_handler = LowestCardHandler()
    lowest_card_nf_handler = LowestCardNoFoldHandler()

    highest_card_handler = HighestCardHandler()
    highest_card_nf_handler = HighestCardNoFoldHandler()

    available_handlers = [network_handler, random_handler, random_nf_handler, lowest_card_handler, lowest_card_nf_handler, highest_card_handler, highest_card_nf_handler]

    server = ToepGameServer()

    n_games = 1000

    win_stats = {}

    for handler_pair in itertools.combinations(available_handlers, 2):
        print("Testing {0} vs. {1}".format(handler_pair[0].name, handler_pair[1].name))
        winner_freq = [0, 0]
        bar = ProgressBar()
        for game_idx in bar(range(0, n_games)):
            server.init_game()

            if game_idx < n_games / 2:
                server.set_player_handler(0, handler_pair[0])
                server.set_player_handler(1, handler_pair[1])

                winner = server.start()
            else:
                server.set_player_handler(1, handler_pair[0])
                server.set_player_handler(0, handler_pair[1])

                winner = server.start() ^ 1
            winner_freq[winner] += 1

        win_stats[(handler_pair[0].name, handler_pair[1].name)] = 100.0 * winner_freq[0] / float(n_games)
        win_stats[(handler_pair[1].name, handler_pair[0].name)] = 100.0 * winner_freq[1] / float(n_games)

    for handler in available_handlers:
        win_stats[(handler.name, handler.name)] = 50.0

    print_win_stats(win_stats, available_handlers)
