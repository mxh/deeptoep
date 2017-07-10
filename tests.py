from state import *

def deck_test():
    deck = Deck()
    print(str(deck))
    deck.shuffle()
    print(str(deck))

def toep_game_test():
    game = ToepGame()
    print(str(game))

if __name__=="__main__":
    toep_game_test()
