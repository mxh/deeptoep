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

    def copy(self):
        player = PlayerState()
        player.hand = self.hand[:]
        player.table = self.table[:]

        return player

    def play(self, card_idx):
        self.table.append(self.hand.pop(card_idx))
        return self.table[-1]

    def __str__(self):
        return "{0: <28}  {1}".format("H: [{0}]".format(cards_to_string(self.hand)), "T: [{0}]".format(cards_to_string(self.table)))

class ToepGame:
    def __init__(self, n_players=2):
        self.deck = Deck()
        self.deck.shuffle()
        self.players = [PlayerState(self.deck.deal(4)) for _ in range(0, n_players)]

        self.round_number = 0
        self.current_player = 0
        self.card_to_beat = None

        self.winner = None

    def copy(self):
        game = ToepGame()
        game.deck = self.deck.copy()
        game.players = [player.copy() for player in self.players]
        game.round_number = self.round_number
        game.current_player = self.current_player
        game.card_to_beat = self.card_to_beat
        game.winner = self.winner

        return game

    def get_valid_actions(self):
        if self.card_to_beat == None:
            return range(0, len(self.players[self.current_player].hand))
        else:
            cards_of_asked_suit = [idx for idx in range(0, len(self.players[self.current_player].hand)) if \
                                   self.players[self.current_player].hand[idx][1] == self.card_to_beat[0][1]]
            if len(cards_of_asked_suit) > 0:
                return cards_of_asked_suit
            else:
                return range(0, len(self.players[self.current_player].hand))

    def move(self, action):
        # if the game is finished, an action doesn't "do" anything, we just return with the next player
        game = self.copy()

        if game.game_finished():
            game.current_player = (game.current_player + 1) % len(game.players)
            return game

        assert(action in game.get_valid_actions())

        card = game.players[game.current_player].play(action)
        if game.card_to_beat == None:
            game.card_to_beat = [card, game.current_player]
        else:
            if game.card_to_beat[0][1] == card[1] and values.index(game.card_to_beat[0][0]) < values.index(card[0]):
                game.card_to_beat = [card, game.current_player]

        if game.round_finished():
            game.round_number += 1
            if game.game_finished():
                game.winner = game.card_to_beat[1]
                game.current_player = (game.current_player + 1) % len(game.players)
            else:
                game.current_player = game.card_to_beat[1]
            game.card_to_beat = None
        else:
            game.current_player = (game.current_player + 1) % len(game.players)

        return game

    def round_finished(self):
        table_lengths = np.array([len(player.table) for player in self.players])
        return np.all(table_lengths == table_lengths[0])

    def game_finished(self):
        return self.round_finished() and self.round_number == 4

    def __str__(self):
        player_str = "\n\n".join(["P{0} | {1}".format(idx, self.players[idx]) for idx in range(0, len(self.players))])
        if self.winner:
            current_player_str = "W: P{0}".format(self.winner)
        else:
            current_player_str = "C: P{0}".format(self.current_player)
        card_to_beat_str = "B: {0}".format(card_to_string(self.card_to_beat) if self.card_to_beat else "X")
        deck_str = "D: {0}".format(self.deck)
        return "\n\n".join([player_str, current_player_str, card_to_beat_str, deck_str])

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
