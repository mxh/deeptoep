from math import factorial
import numpy as np
import ipdb

def n_partitions(n, value):
    arr = [0]*n  # create an array of size n, filled with zeroes
    accum = []
    n_partitions_recursive(n, value, 0, n, arr, accum);
    return accum

def n_partitions_recursive(n, value, sumSoFar, topLevel, arr, accum):
    if n == 1:
        if sumSoFar <= value:
            #Make sure it's in ascending order (or only level)
            if topLevel == 1 or (value - sumSoFar >= arr[-2]):
                arr[(-1)] = value - sumSoFar #put it in the n_th last index of arr
                accum.append(arr[:])
    elif n > 0:
        #Make sure it's in ascending order
        start = 0
        if (n != topLevel):
            start = arr[(-1*n)-1]   #the value before this element

        for i in range(start, value+1): # i = start...value
            arr[(-1*n)] = i  # put i in the n_th last index of arr
            n_partitions_recursive(n-1, value, sumSoFar + i, topLevel, arr, accum)

def n_partitions_max(n, value, max_value=8):
    parts = n_partitions(n, value)
    parts = [part for part in parts if np.all(np.array(part) <= max_value)]
    print(parts)

    return parts

def n_toep_states(n_players, n_cards_per_player):
    n_cards = n_players * n_cards_per_player
    suit_partitions = n_partitions_max(4, n_cards, 8)
    orderings_per_partition = factorial(n_cards) / (n_players * factorial(n_cards_per_player))
    return len(suit_partitions) * orderings_per_partition

if __name__=="__main__":
    for n_players in range(2, 4):
        for n_cards in range(1, 5):
            print("Number of states for {0} players and {1} cards: {2}".format(n_players, n_cards, n_toep_states(n_players, n_cards)))
