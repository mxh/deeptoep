import itertools
import random
import numpy as np
import ipdb

# all values and suits available in a piquet deck
values = ["J", "Q", "K", "A", 7, 8, 9, 10]
suits = ["D", "C", "H", "S"]

def card_to_string(card):
    return "{0: <4}".format("{0}o{1}".format(card[0], card[1]))

def cards_to_string(cards):
    return ", ".join([card_to_string(card) for card in cards])

def sort_cards(cards):
    sorted_cards = sorted(cards, key=lambda x: values.index(x[0]) * 4 + suits.index(x[1]))
    return sorted_cards

class Deck:
    def __init__(self):
        self.cards = list(itertools.product(values, suits))

    def copy(self):
        deck = Deck()
        deck.cards = self.cards[:]

        return deck

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, n):
        assert(n <= len(self.cards))
        dealt_cards = self.cards[0:n]
        self.cards = self.cards[n:]

        return dealt_cards

    def __str__(self):
        return cards_to_string(self.cards)

class PlayerState:
    def __init__(self, hand=[]):
        self.hand = sort_cards(hand)
        self.table = []
        self.can_toep = True
        self.did_fold = False

    def copy(self):
        player = PlayerState()
        player.hand = self.hand[:]
        player.table = self.table[:]
        player.can_toep = self.can_toep
        player.did_fold = self.did_fold

        return player

    def play(self, action):
        if action in range(0, len(self.hand)):
            self.table.append(self.hand.pop(action))
            return self.table[-1]
        elif action == "t":
            self.can_toep = False
        elif action == "f":
            self.did_fold = True

        return None

    def __str__(self):
        return "{0: <28}  {1} {2}".format("H: [{0}]".format(cards_to_string(self.hand)), "T: [{0}]".format(cards_to_string(self.table)), "F" if self.did_fold else "")

class ToepGame:
    def __init__(self, n_players=2):
        self.deck = Deck()
        self.deck.shuffle()
        self.players = [PlayerState(self.deck.deal(4)) for _ in range(0, n_players)]
        self.stake = 1
        self.first_player_raising_stake = None
        self.player_raising_stake = None

        self.round_number = 0
        self.current_player = 0
        self.card_to_beat = None

        self.winner = None

    def copy(self):
        game = ToepGame()
        game.deck = self.deck.copy()
        game.players = [player.copy() for player in self.players]
        game.stake = self.stake
        game.first_player_raising_stake = self.first_player_raising_stake
        game.player_raising_stake = self.player_raising_stake

        game.round_number = self.round_number
        game.current_player = self.current_player
        game.card_to_beat = self.card_to_beat

        game.winner = self.winner

        return game

    def get_valid_actions(self):
        if self.player_raising_stake != None and self.current_player != self.player_raising_stake:
            available_toep_action = ['t'] if self.stake < 5 else []
            return ['c', 'f'] + available_toep_action
        else:
            available_toep_action = ['t'] if self.players[self.current_player].can_toep else []
            if self.card_to_beat == None:
                return range(0, len(self.players[self.current_player].hand)) + available_toep_action
            else:
                cards_of_asked_suit = [idx for idx in range(0, len(self.players[self.current_player].hand)) if \
                                       self.players[self.current_player].hand[idx][1] == self.card_to_beat[0][1]]
                if len(cards_of_asked_suit) > 0:
                    return cards_of_asked_suit + available_toep_action
                else:
                    return range(0, len(self.players[self.current_player].hand)) + available_toep_action

    def next_player(self):
        self.current_player = (self.current_player + 1) % len(self.players)
        while self.players[self.current_player].did_fold:
            self.current_player = (self.current_player + 1) % len(self.players)

        if self.player_raising_stake != None and self.current_player == self.player_raising_stake:
            self.current_player = self.first_player_raising_stake
            self.player_raising_stake = None
            self.first_player_raising_stake = None
            while self.players[self.current_player].did_fold:
                self.current_player = (self.current_player + 1) % len(self.players)

    def move(self, action):
        # if the game is finished, an action doesn't "do" anything, we just return with the next player
        game = self.copy()
        if game.game_finished():
            game.current_player = (game.current_player + 1) % len(game.players)
            return game

        # make sure this action is actually valid
        assert(action in game.get_valid_actions())

        card = game.players[game.current_player].play(action)
        if action == 't':
            game.stake += 1
            # if this is not the first raising of the stakes, then the previous
            # player to raise the stakes is free to do so again now
            if self.player_raising_stake != None:
                game.players[self.player_raising_stake].can_toep = True
            game.player_raising_stake = game.current_player
            if game.first_player_raising_stake == None:
                game.first_player_raising_stake = game.current_player
            game.next_player()
        elif action == 'c':
            game.next_player()
        elif action == 'f':
            game.next_player()
        else:
            if game.player_raising_stake == game.current_player:
                game.player_raising_stake = None
            if game.card_to_beat == None:
                game.card_to_beat = [card, game.current_player]
            else:
                if game.card_to_beat[0][1] == card[1] and values.index(game.card_to_beat[0][0]) < values.index(card[0]):
                    game.card_to_beat = [card, game.current_player]

            if game.round_finished():
                game.round_number += 1
                if game.game_finished():
                    game.winner = game.card_to_beat[1]
                    game.next_player()
                else:
                    game.current_player = game.card_to_beat[1]
                game.card_to_beat = None
            else:
                game.next_player()

        return game

    def round_finished(self):
        table_lengths = np.array([len(player.table) for player in self.players])
        return np.all(table_lengths == table_lengths[0])

    def game_finished(self):
        has_won_toep = np.all([self.players[player_idx].did_fold for player_idx in range(0, len(self.players)) if player_idx != self.current_player])
        return has_won_toep or (self.round_finished() and self.round_number == 4)

    def __str__(self):
        player_str = "\n\n".join(["P{0} | {1}".format(idx, self.players[idx]) for idx in range(0, len(self.players))])
        if self.winner:
            current_player_str = "W: P{0}".format(self.winner)
        else:
            current_player_str = "C: P{0}".format(self.current_player)
        card_to_beat_str = "B: {0}".format(card_to_string(self.card_to_beat) if self.card_to_beat else "X")
        stake_str= "S: {0}".format(self.stake)
        toep_str = "TP: {0}".format(self.player_raising_stake if self.player_raising_stake != None else "X")
        return "\n\n".join([player_str, current_player_str, card_to_beat_str, stake_str, toep_str])

class RandomPolicy:
    def __init__(self):
        pass

    def get_action(self, game):
        valid_actions = game.get_valid_actions()
        return random.choice(valid_actions)

def generate_episode(game, policy):
    states = [game]

    while not game.game_finished():
        new_game = game.move(policy.get_action(game))
        states.append(new_game)
        game = new_game

    return states

if __name__=="__main__":
    #ipdb.set_trace()
    policy = RandomPolicy()
    game = ToepGame(2)
    episode = generate_episode(game, policy)

    for state in episode:
        print(state)
