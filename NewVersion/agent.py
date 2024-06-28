import numpy as np
import pickle
import random
import time
import os

import concurrent.futures

import torch
import torch.nn.functional as F

from tqdm import trange

class AlphaZero:
    def __init__(self, model, optimizer, scheduler, game, args, mcts):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.game = game
        self.args = args

        self.mcts = mcts
    
    def _experience_buffer(self, memory, value):
        experience_buffer = []

        for state, policy in memory:
            experience_buffer.append((self.game.get_encoded_state(state), policy, value))
        return experience_buffer
    
    def selfplay(self, gameID):
        print(f"Game {gameID} starts...")
        memory = []
        move_count = 1

        state = self.game.get_initial_state()
        player = 1
        while True:
            reward = self.game.get_reward(state, player=1)
            if reward is not None:
                print(f"Game {gameID} ends...")
                return self._experience_buffer(memory, reward)

            root = self.mcts.search(self.args.time_limit, state, player)
            if move_count < 30:
                action_probs = root.get_action_probs(self.game, self.args.temperature)
            else:
                action_probs = root.get_action_probs(self.game, temperature=0)
            move_count += 1
            action_probs[np.isnan(action_probs)] = 0

            memory.append((state, action_probs))

            action = np.random.choice(self.game.action_size, p=action_probs)
            state = self.game.get_next_state(state, action, player)
            player = player * -1
    
    def create_dataset(self, datasetID, parallel=False):
        print("===> Creating Dataset...")
        start_time = time.time()
        dataset = []
        self.model.eval()
        if parallel:
            for _ in trange(self.args.num_self_play_iterations):
                with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
                    futures = [executor.submit(self.selfplay, gameID) for gameID in range(1, 16+1)]
                    results = [future.result() for future in concurrent.futures.as_completed(futures)]
                for result in results:
                    dataset += result
        else:
            for sp_iteration in trange(self.args.num_self_play_iterations):
                dataset += self.selfplay(sp_iteration+1)
    
        random.shuffle(dataset)
        
        if not os.path.isdir('datasets'):
            os.mkdir('datasets')
        with open(f'datasets/{repr(self.game)}{datasetID}.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset {datasetID} created in {time.time() - start_time} seconds...")

    def train(self, dataset):
        value_loss, policy_loss, total_loss, correct_predictions, total_predictions = 0, 0, 0, 0, 0

        for batch_idx in range(0, len(dataset), self.args.batch_size):
            sample = dataset[batch_idx: min(len(dataset) - 1, batch_idx + self.args.batch_size)]

            states, policy_targets, value_targets = zip(*sample)

            states = np.array(states, dtype=np.float32) 
            policy_targets = np.array(policy_targets, dtype=np.float32) 
            value_targets = np.array(value_targets, dtype=np.float32).reshape(-1, 1)

            states = torch.tensor(states, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(states)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            value_loss += value_loss.item()
            policy_loss += policy_loss.item()
            total_loss += loss.item()

            model_predictions = torch.softmax(out_policy, axis=1).squeeze(0).cpu().detach().numpy().argmax(axis=1)
            target_predictions = torch.softmax(policy_targets, axis=1).squeeze(0).cpu().detach().numpy().argmax(axis=1)
            correct_predictions = np.equal(model_predictions, target_predictions).sum()
            
            total_predictions += len(model_predictions)

        print(f"value_loss: {value_loss:.3f} -- policy_loss:{policy_loss:.3f} -- total_loss:{total_loss:.3f} -- train_accuracy: {100.*correct_predictions/total_predictions:.3f}")
    
    def train_dataset(self, datasetID):
        with open(f'datasets/{repr(self.game)}{datasetID}.pkl', 'rb') as f:
            print(f"Open {repr(self.game)}{datasetID}.pkl...")
            dataset = pickle.load(f)

        print("==> Start training...")
        self.model.train()
        for epoch in range(self.args.num_epochs):
            print(f"Epoch number {epoch} ---------------------------------------------------------------------------")
            self.train(dataset)
            self.scheduler.step()
        
        print("==> Saving Model...")
        if not os.path.isdir("model"):
            os.mkdir("model")
        torch.save(self.model.state_dict(), f"model/model{repr(self.game)}{datasetID}.pt")
        if not os.path.isdir("optim"):
            os.mkdir("optim")
        torch.save(self.optimizer.state_dict(),  f"optim/optim{repr(self.game)}{datasetID}.pt")