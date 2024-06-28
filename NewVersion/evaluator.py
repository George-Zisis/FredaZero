import numpy as np
import torch
import pickle
import time

from torchsummary import summary

from scipy.signal import convolve2d
from scipy.signal import correlate2d

from core.mcts import MonteCarloTreeSearch
from plots import plot_results

from tqdm import trange

class Evaluate:
    def __init__(self, game, args, model, mcts):
        self.game = game
        self.args = args

        self.model = model

        self.mcts = mcts
    
    def random_game(self):
        results = {'win': 0, 'loss': 0, 'draw': 0}
        timers = []
        for _ in trange(100):
            state = self.game.get_initial_state()
            player = 1
            start_time = time.time()
            while True:
                reward = self.game.get_reward(state, 1)
                if reward is not None:
                    total_time = time.time() - start_time
                    timers.append(total_time)
                    if reward == 1:
                        results['win'] += 1
                    elif reward == -1:
                        results['loss'] += 1
                    else:
                        results['draw'] += 1
                    break

                if player == 1:
                    root = self.mcts.search(time_limit=1, state=state, to_play=player)
                    action_probs = root.get_action_probs(self.game, temperature=1)
                    action_probs[np.isnan(action_probs)] = 0
                else:
                    root = self.mcts.search(time_limit=1, state=state, to_play=player)
                    action_probs = root.get_action_probs(self.game, temperature=1)
                    action_probs[np.isnan(action_probs)] = 0
                action = np.random.choice(self.game.action_size, p=action_probs)
                state = self.game.get_next_state(state, action, player)
                # self.game.render(state)
                player = player * -1
        average_time = sum(timers) / len(timers)
        print(f"Average game time:{average_time}")
        plot_results(results)

    def dataset_stats(self, datasetID):
        with open(f'datasets/{repr(self.game)}{datasetID}.pkl', 'rb') as f:
            print(f"Open {repr(self.game)}{datasetID}.pkl...")
            dataset = pickle.load(f)

        print(f"Type:{type(dataset)} Length: {len(dataset)}")
        count1, count2 = 0, 0
        for element in dataset:
            _, _, reward = element
            if reward == 1:
                count1 += 1
            elif reward == -1:
                count2 += 1
        print(f"Reward = 1: {count1} Reward = -1: {count2}")

    def vs_human(self):
        state = self.game.get_initial_state()
        self.game.render(state)
        player = 1
        while True:
            value = self.game.get_reward(state, player=1)
            if value is not None:
                print(f"Game is over with winner:{value}")
                break
            if player == 1:
                root = self.mcts.search(self.args.time_limit, state, player)
                action_probs = root.get_action_probs(self.game, temperature=1)
                action = np.random.choice(self.game.action_size, p=action_probs)
                # action = root.select_action(temperature=0)
            else:
                action = int(input("Give me your fucking play: "))
            state = self.game.get_next_state(state, action, player)
            self.game.render(state)
            player = player * -1
    
    def vs_bot(self, bot1, bot2):
        self.model.load_state_dict(torch.load(f"./model/model{repr(self.game)}{bot1}.pt"))
        mcts1 = MonteCarloTreeSearch(self.game, self.args, self.model)
        self.model.load_state_dict(torch.load(f"./model/model{repr(self.game)}{bot2}.pt"))
        mcts2 = MonteCarloTreeSearch(self.game, self.args, self.model)

        state = self.game.get_initial_state()
        player = 1
        while True:
            value = self.game.get_reward(state, player=1)
            if value is not None:
                print(f"Game is over with winner:{value}")
                break
            if player == 1:
                root = mcts2.search(self.args.time_limit, state, player)
                action_probs = root.get_action_probs(self.game, temperature=1)
                print(action_probs)
                action = np.random.choice(self.game.action_size, p=action_probs)
                # action = root.select_action(temperature=1)
            else:
                root = mcts1.search(self.args.time_limit, state, player)
                action_probs = root.get_action_probs(self.game, temperature=1)
                print(action_probs)
                action = np.random.choice(self.game.action_size, p=action_probs)
                # action = root.select_action(temperature=1)
            state = self.game.get_next_state(state, action, player)
            self.game.render(state)
            player = player * -1