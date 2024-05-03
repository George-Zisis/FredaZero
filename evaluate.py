import numpy as np
import torch
import time

class Evaluation:
	def __init__(self, game, args, model, mcts):
		self.game = game
		self.args = args
		self.model = model
		self.mcts = mcts

	def get_states(self):
		np.random.seed(10)

		states, numTurns = [], [5, 6, 7, 8, 10, 11, 11, 11, 15, 15, 17, 17, 20, 20, 24, 25, 27, 30, 31, 32, 33]

		for turns in numTurns:

			state = self.game.get_initial_state()
			for i in range(turns):
				reward, terminated = self.game.check_game_over(state)
				if terminated:
					break
				player = self.game.get_current_player(state)
				neutralState = self.game.change_perspective(state)

				validMoves = self.game.get_valid_moves(state)
				action = np.random.choice(np.where(validMoves==1)[0])

				state = self.game.get_next_state(state, action)

			states.append(state)

		return states 

	def evaluateMCTS(self):
		gameStart = time.time()
		actionTime = []
		print(f"Game Starts...")
		state = self.game.get_initial_state()
		while True:
			reward, terminated = self.game.check_game_over(state)
			if terminated:
				gameEnd = time.time()
				print(f"Game has ended...")
				print(f"Winner is {self.game.winner} in {gameEnd-gameStart} seconds")
				print(f"Average action selection in {np.average(actionTime)} seconds")
				break

			player = self.game.get_current_player(state)
			neutralState = self.game.change_perspective(state)

			validMoves = self.game.get_valid_moves(state)
			print(f"Valid Moves: {validMoves}")

			searchStart = time.time()
			actionProbs = self.mcts.search(neutralState)
			searchEnd = time.time()
			actionTime.append(searchEnd-searchStart)

			action = np.argmax(actionProbs)
			state = self.game.get_next_state(state, action)

	def evaluate_state(self, state):
		self.model.eval()

		player = self.game.get_current_player(state)
		print(f"Current Player: {player}")

		validMoves = self.game.get_valid_moves(state)

		neutralState = self.game.change_perspective(state)
		tensorState = torch.tensor(self.game.get_encoded_state(neutralState), device=self.model.device).unsqueeze(0)

		policy, value = self.model(tensorState)

		policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
		policy = policy * validMoves
		policy = policy / np.sum(policy)
		value = value.item()

		np.set_printoptions(suppress=True)

		print(state)
		print(f"Best Action: {np.argmax(policy)} with probability {np.max(policy)}")
		print(f"Policy:")
		print(np.array2string(policy, precision=8, floatmode='fixed'))
		print(f"Value: {value} and policySum: {np.sum(policy)}")
		print(f"The five best actions are: {policy.argsort()[-5:][::-1]}")

	def evaluateDataset(self, datasetID):
		with open(f'datasets/{repr(self.game)}{datasetID}.pkl', 'rb') as f:
			dataset = pickle.load(f)
		state, policyTargets, valueTargets = zip(*dataset)
		state, policyTargets = np.array(state), np.array(policyTargets)
		valueTargets = np.array(valueTargets).reshape(-1, 1)
		# for i in range(0, len(valueTargets), 20):
		# 	print(policyTargets[i])
		print(policyTargets)

	def vsHuman(self):
		state = self.game.get_initial_state()
		while True:
			reward, terminated = self.game.check_game_over(state)
			if terminated:
				print(f"Winner:{self.game.winner}")
				print(state)
				break

			player = self.game.get_current_player(state)
			if player == "WHITE":
				legalMoves = self.game.get_valid_moves(state)
				action = int(input("Where do you play?"))
				if action not in np.where(legalMoves==1)[0]:
					print("Illegal Action")
					continue
			else:
				neutralState = self.game.change_perspective(state)
				tensorState = torch.tensor(self.game.get_encoded_state(neutralState), device=self.model.device).unsqueeze(0)

				policy, value = self.model(tensorState)
				policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
				validMoves = self.game.get_valid_moves(state)
				policy = policy * validMoves
				policy = policy / np.sum(policy)
				action = np.argmax(policy)

			state = self.game.get_next_state(state, action)
			print(state)