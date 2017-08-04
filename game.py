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
        self.score = 0
        self.reset_round(hand)

    def reset_round(self, hand):
        self.hand = hand
        self.table = []
        self.can_toep = True
        self.did_fold = False

    def copy(self):
        player = PlayerState()
        player.hand = self.hand[:]
        player.table = self.table[:]
        player.can_toep = self.can_toep
        player.did_fold = self.did_fold
        player.score = self.score

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

    def str_hidden(self):
        return "S: {0: <2} {1: <28}  {2}".format(self.score, "T: [{0}]".format(cards_to_string(self.table)), "F" if self.did_fold else "")

    def __str__(self):
        return "S: {0: <2} {1: <28}  {2} {3}".format(self.score, "H: [{0}]".format(cards_to_string(self.hand)), "T: [{0}]".format(cards_to_string(self.table)), "F" if self.did_fold else "")

class ToepGamePhase:
    def __init__(self, game):
        self.game = game
        self.reset()

    def reset(self, starting_player=0):
        self.current_player = starting_player
        self.trick_number = 0
        self.card_to_beat = None

    def copy(self, game):
        phase = ToepGamePhase(game)

        phase.current_player = self.current_player

        phase.trick_number = self.trick_number
        phase.card_to_beat = self.card_to_beat

        return phase

    def get_valid_actions(self):
        valid_actions = []

        if self.game.betting_phase.last_player_to_raise != self.current_player and self.game.stake < 15:
            valid_actions.append('t')

        if self.card_to_beat == None:
            valid_actions.extend(range(0, len(self.game.players[self.current_player].hand)))
        else:
            cards_of_asked_suit = [idx for idx in range(0, len(self.game.players[self.current_player].hand)) if \
                                   self.game.players[self.current_player].hand[idx][1] == self.card_to_beat[0][1]]
            if len(cards_of_asked_suit) > 0:
                valid_actions.extend(cards_of_asked_suit)
            else:
                valid_actions.extend(range(0, len(self.game.players[self.current_player].hand)))

        return valid_actions

    def move(self, action):
        game = self.game.copy()
        game.game_phase.self_move(action)

        return game

    def self_move(self, action):
        if not action in self.get_valid_actions():
            return [-10, False, False]

        assert(action in self.get_valid_actions())
        if action == 't':
            # switch to betting
            self.game.phase = self.game.betting_phase
            self.game.betting_phase.current_player = self.current_player
            return self.game.betting_phase.self_move(action)
        else:
            card = self.game.players[self.current_player].play(action)

            if self.card_to_beat == None:
                # first card of the round
                self.card_to_beat = [card, self.current_player]
            else:
                # if the player followed suit and beat the card to beat, we have a new leader
                if self.card_to_beat[0][1] == card[1] and values.index(self.card_to_beat[0][0]) < values.index(card[0]):
                    self.card_to_beat = [card, self.current_player]

            if self.trick_finished():
                self.trick_number += 1
                if self.game_finished():
                    for player_idx in range(0, len(self.game.players)):
                        if player_idx != self.card_to_beat[1]:
                            self.game.players[player_idx].score += self.game.stake
                    self.game.reset_round((self.card_to_beat[1] + 1) % len(self.game.players))
                else:
                    self.current_player = self.card_to_beat[1]
                    self.card_to_beat = None
            else:
                self.next_player()

    def trick_finished(self):
        table_lengths = np.array([len(player.table) for player in self.game.players])
        return np.all(table_lengths == table_lengths[0])

    def game_finished(self):
        table_lengths = np.array([len(player.table) for player in self.game.players])
        return np.all(table_lengths == 4)

    def next_player(self):
        self.current_player = (self.current_player + 1) % len(self.game.players)
        while self.game.players[self.current_player].did_fold:
            self.current_player = (self.current_player + 1) % len(self.game.players)

    def __str__(self):
        return "CP: {0}  T: {1}  CB: {2}".format(self.current_player, self.trick_number, card_to_string(self.card_to_beat[0]) if self.card_to_beat != None else "X")

class ToepBettingPhase:
    def __init__(self, game):
        self.game = game
        self.reset()

    def reset(self, starting_player=0):
        self.current_player = starting_player
        self.last_player_to_raise = None

    def copy(self, game):
        phase = ToepBettingPhase(game)

        phase.current_player = self.current_player
        phase.last_player_to_raise = self.last_player_to_raise

        return phase

    def get_valid_actions(self):
        valid_actions = ['c', 'f']

        if self.last_player_to_raise != self.current_player and self.game.stake < 15:
            valid_actions.append('t')

        return valid_actions

    def move(self, action):
        game = self.game.copy()
        game.betting_phase.self_move(action)

        return game

    def self_move(self, action):
        if not action in self.get_valid_actions():
            return

        if action == 't':
            self.game.stake += 1
            self.last_player_to_raise = self.current_player
        if action == 'f':
            self.game.players[self.current_player].did_fold = True
            self.game.players[self.current_player].score += self.game.stake - 1

        self.next_player()
        if self.current_player == self.last_player_to_raise:
            if self.game_finished():
                round_winner = [player_idx for player_idx in range(0, len(self.game.players)) if not self.game.players[player_idx].did_fold][0]
                self.game.reset_round((round_winner + 1) % len(self.game.players))
            else:
                self.game.phase = self.game.game_phase

    def game_finished(self):
        return len([player for player in self.game.players if not player.did_fold]) == 1

    def next_player(self):
        self.current_player = (self.current_player + 1) % len(self.game.players)
        while self.game.players[self.current_player].did_fold:
            self.current_player = (self.current_player + 1) % len(self.game.players)

    def __str__(self):
        return "BR! CP: {0}".format(self.current_player)

class ToepGame:
    def __init__(self, n_players=2):
        self.deck = Deck()
        self.deck.shuffle()
        self.players = [PlayerState(sort_cards(self.deck.deal(4))) for _ in range(0, n_players)]
        self.stake = 1

        self.game_phase = ToepGamePhase(self)
        self.betting_phase = ToepBettingPhase(self)
        self.phase = self.game_phase

    def copy(self):
        game = ToepGame()
        game.deck = self.deck.copy()
        game.players = [player.copy() for player in self.players]
        game.stake = self.stake

        game.game_phase = self.game_phase.copy(game)
        game.betting_phase = self.betting_phase.copy(game)
        game.phase = game.betting_phase if self.phase == self.betting_phase else game.game_phase

        return game

    def reset_round(self, starting_player):
        self.deck = Deck()
        self.deck.shuffle()

        for player in self.players:
            player.reset_round(sort_cards(self.deck.deal(4)))

        self.stake = 1

        self.game_phase.reset(starting_player)
        self.betting_phase.reset()

        self.phase = self.game_phase

    def get_valid_actions(self):
        return self.phase.get_valid_actions()

    def get_winner(self):
        non_losing_players = [player_idx for player_idx in range(0, len(self.players)) if self.players[player_idx].score < 15]
        if len(non_losing_players) == 1:
            return non_losing_players[0]

        return None

    def move(self, action):
        return self.phase.move(action)

    def str_hidden(self):
        player_str = "\n\n".join(["P{0} | {1}".format(idx, self.players[idx] if idx == self.phase.current_player else self.players[idx].str_hidden()) for idx in range(0, len(self.players))])
        phase_str = str(self.phase)
        last_to_raise_str = "T: {0}".format(self.betting_phase.last_player_to_raise)
        stake_str = "S: {0}".format(self.stake)
        return "\n\n".join([player_str, phase_str, stake_str])

    def __str__(self):
        player_str = "\n\n".join(["P{0} | {1}".format(idx, self.players[idx]) for idx in range(0, len(self.players))])
        phase_str = str(self.phase)
        last_to_raise_str = "T: {0}".format(self.betting_phase.last_player_to_raise)
        stake_str = "S: {0}".format(self.stake)
        return "\n\n".join([player_str, phase_str, stake_str])

class RandomPolicy:
    def __init__(self):
        pass

    def get_action(self, game):
        valid_actions = game.get_valid_actions()
        return random.choice(valid_actions)

def generate_episode(game, policy):
    states = [game]

    while game.get_winner() == None:
        print([player.score for player in game.players])
        new_game = game.move(policy.get_action(game))
        states.append(new_game)
        game = new_game

    print([player.score for player in game.players])

    return states

def print_episode(episode):
    for state in episode:
        print(state)
        print("-------")

if __name__=="__main__":
    policy = RandomPolicy()
    game = ToepGame(2)
    episode = generate_episode(game, policy)
    
    print_episode(episode)
