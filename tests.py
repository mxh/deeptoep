from game import *
from network import *

def np_cmp_test(expected, returned):
    return np.all(expected == returned)

def deck_test():
    deck = Deck()
    print("Unshuffled deck: {0}".format(str(deck)))
    deck.shuffle()
    print("Shuffled deck: {0}".format(str(deck)))

    return [0, 0]

def toep_game_test():
    game = ToepGame()
    print("Random Toep game:")
    print(str(game))
    
    return [0, 0]

def card_to_one_hot_test():
    test_cards = [("A", "S"), (10, "H"), ("J", "D"), (7, "C")]
    test_expected = [np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]),\
                     np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]),\
                     np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),\
                     np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])]

    n_tests = len(test_cards)
    n_passed = 0
    for idx in range(0, len(test_cards)):
        card = test_cards[idx]
        oh = card_to_one_hot(card)
        expected = test_expected[idx]
        print("Ret: {0}".format(oh))
        print("Exp: {0}".format(expected))
        test = np_cmp_test(expected, oh)
        n_passed += test
        print("Test {0}".format("PASSED" if test else "FAILED"))

    return [n_tests, n_passed]

def toep_state_test():
    game = ToepGame()
    game.players[0].hand = [("A", "S"), (9, "H"), (10, "H")]
    game.players[0].table = [(7, "C")]
    game.players[1].hand = [("Q", "H"), (7, "D"), (8, "H")]
    game.players[1].table = [("J", "C")]
    state = ToepState(game)
    expected = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
                         0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,\
                         0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,\
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
                         0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,\
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
                         1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,\
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32)
    print("Ret: {0}".format(state.state_vec))
    print("Exp: {0}".format(expected))
    test = np_cmp_test(expected, state.state_vec)
    print("Test {0}".format("PASSED" if test else "FAILED"))

    return [1, test]

def state_vec_to_game_test():
    policy = RandomPolicy()
    game = ToepGame(2)
    episode = generate_episode(game, policy)

    game = episode[-1]
    game_str = str(game)
    print("Original game:")
    print(game_str)
    state = ToepState(game)
    reconstructed_game = state_vec_to_game(state.state_vec)
    reconstructed_game_str = str(reconstructed_game)
    print("Reconstructed game:")
    print(reconstructed_game_str)

    return [1, reconstructed_game_str == game_str]

tests = {"Deck test": deck_test,\
         "ToepGame test": toep_game_test,\
         "card_to_one_hot test": card_to_one_hot_test,\
         "ToepState test": toep_state_test,\
         "state_vec to ToepGame test": state_vec_to_game_test}

if __name__=="__main__":
    all_n_tests = 0
    all_n_passed = 0
    for test_name, test_func in tests.iteritems():
        print("-- {0} --".format(test_name))
        [n_tests, n_passed] = test_func()
        all_n_tests += n_tests
        all_n_passed += n_passed
        print

    print("Ran {0} tests, {1} passed, {2} failed".format(all_n_tests, all_n_passed, all_n_tests - all_n_passed))
