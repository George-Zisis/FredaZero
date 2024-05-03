import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import time
import math
import random
import pickle

class AlphaZeroParallel:
	def __init__(self, model, optimizer, game, args, mcts):
		self.model = model
		self.optimizer = optimizer

		self.game = game
		self.args = args

		self.mcts = mcts

	def _experience_buffer(self, memory, value):
		experienceBuffer = []

		for state, policy in memory:
			experienceBuffer.append((self.game.get_encoded_state(state), policy, value))
		return experienceBuffer

	def selfplay(self, params):
		torch.set_num_threads(math.floor(torch.get_num_interop_threads() / torch.get_num_threads()))
		memory = []
		print(f"Game starts...")
		gameStart = time.time()
		state = self.game.get_initial_state()
		while True:
			value, terminated = self.game.check_game_over(state)
			if terminated:
				gameEnd = time.time()
				print(f"Game ends in {gameEnd-gameStart} seconds...")
				return self._experience_buffer(memory, value)

			neutralState = self.game.change_perspective(state)
			mctsProbs = self.mcts.search(neutralState)
			
			memory.append((neutralState, mctsProbs))

			mctsProbs = mctsProbs ** (1 / self.args['temperature'])
			mctsProbs = mctsProbs / np.sum(mctsProbs)

			action = np.random.choice(self.game.actionSpace, p=mctsProbs)

			state = self.game.get_next_state(state, action)

	def train(self, memory):
		random.shuffle(memory)

		for batchIdx in range(0, len(memory), self.args['batchSize']):
			sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batchSize'])]
			state, policyTargets, valueTargets = zip(*sample)

			state = torch.tensor(np.array(state), dtype=torch.float32, device=self.model.device)
			policyTargets = torch.tensor(np.array(policyTargets), dtype=torch.float32, device=self.model.device)
			valueTargets = torch.tensor(np.array(valueTargets).reshape(-1,1), dtype=torch.float32, device=self.model.device)

			policy, value = self.model(state)

			policyLoss = F.cross_entropy(policy, policyTargets)
			valueLoss = F.mse_loss(value, valueTargets)

			loss = policyLoss + valueLoss

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

	def get_dataset(self, datasetID):
		memory = []
		startTime = time.time()

		self.model.eval()
		with mp.Pool(processes=self.args['numWorkers'], initializer=np.random.seed) as pool:
			results = list(pool.map(self.selfplay, range(self.args['spIterations'])))

		for result in results:
			memory += result

		with open(f'datasets/{repr(self.game)}{datasetID}.pkl', 'wb') as f:
			pickle.dump(memory, f)

		endTime = time.time()
		print(f"Dataset produced in {endTime-startTime} seconds")

	def train_dataset(self, datasetID):
		with open(f'datasets/{repr(self.game)}{datasetID}.pkl', 'rb') as f:
			print(f"Open {repr(self.game)}{datasetID}.pkl")
			memory = pickle.load(f)

		self.model.train()
		for epoch in range(self.args['epochs']):
			self.train(memory)

		torch.save(self.model.state_dict(), f"model/model{repr(self.game)}{datasetID}.pt")
		torch.save(self.optimizer.state_dict(),  f"optim/optim{repr(self.game)}{datasetID}.pt")
