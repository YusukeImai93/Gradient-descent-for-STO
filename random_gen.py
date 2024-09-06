import random

with open('Random_continuous_input.txt', 'a') as f:
    for i in range(3000):
        print(i, random.uniform(0,1),file=f)