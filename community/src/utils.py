import random

def find_most_occurence(num_list, candidates):
    occurence = {}
    for item in candidates:
        occurence[item] = 0
    for item in num_list:
        occurence[item] += 1
    maximum = 0
    next_candidates = []
    for item in candidates:
        if occurence[item] > maximum:
            maximum = occurence[item]
            next_candidates = [item]
        elif occurence[item] == maximum:
            next_candidates.append(item)
    index = random.randint(0, len(next_candidates) - 1)
    return next_candidates[index]