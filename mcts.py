import numpy as np

import torch 
import math

class VanillaNode:
	def __init__(self, game, args, state, parent=None, actionTaken=None):
		self.game = game
		self.args = args
		self.state = state
		self.parent = parent
		self.actionTaken = actionTaken

		self.children = []
		self.expandableMoves = game.get_valid_moves(state)

		self.visitCount = 0
		self.valueSum = 0

	def is_fully_expanded(self):
		return np.sum(self.expandableMoves) == 0 and len(self.children) > 0

	def select(self):
		bestChild = None
		bestUCB = -np.inf

		for child in self.children:
			ucb = self.get_ucb(child)
			if ucb > bestUCB:
				bestChild = child
				bestUCB = ucb
		return bestChild

	def get_ucb(self, child):
		qValue = 1 - ((child.valueSum / child.visitCount) + 1) / 2
		return qValue + self.args['C'] * math.sqrt(math.log(self.visitCount) / child.visitCount)

	def expand(self):
		action = np.random.choice(np.where(self.expandableMoves == 1)[0])
		self.expandableMoves[action] = 0

		childState = self.state.copy()
		childState = self.game.get_next_state(childState, action)
		childState = self.game.change_perspective(childState)

		child = VanillaNode(self.game, self.args, childState, self, action)
		self.children.append(child)
		return child

	def simulate(self):
		value, terminated = self.game.check_game_over(self.state)
		if terminated:
			return value
		rolloutState = self.state.copy()
		while True:
			validMoves = self.game.get_valid_moves(rolloutState)
			action = np.random.choice(np.where(validMoves==1)[0])
			rolloutState = self.game.get_next_state(rolloutState, action)
			value, terminated = self.game.check_game_over(rolloutState)
			if terminated:
				return value

	def backpropagate(self, value):
		self.valueSum += value
		self.visitCount += 1

		if self.parent is not None:
			self.parent.backpropagate(value)


class VanillaMCTS:
	def __init__(self, game, args):
		self.game = game
		self.args = args

	def search(self, state):
		root = VanillaNode(self.game, self.args, state)

		for search in range(self.args['numSearches']):
			node = root

			while node.is_fully_expanded():
				node = node.select()
			assert not node.is_fully_expanded(), f"Selection Bug"

			value, terminated = self.game.check_game_over(node.state)

			if not terminated:
				node = node.expand()
				value = node.simulate()
			node.backpropagate(value)

		actionProbs = np.zeros(self.game.actionSpace)
		for child in root.children:
			actionProbs[child.actionTaken] = child.visitCount
		actionProbs = actionProbs / np.sum(actionProbs)
		return actionProbs


class Node:
	def __init__(self, game, args, state, parent=None, actionTaken=None, prior=0):
		self.game = game
		self.args = args
		self.state = state
		self.prior = prior
		self.parent = parent
		self.actionTaken = actionTaken

		self.children = []

		self.visitCount = 0
		self.valueSum = 0

	def is_fully_expanded(self):
		return len(self.children) > 0

	def select(self):
		bestChild = None
		bestUCB = -np.inf

		for child in self.children:
			ucb = self.get_ucb(child)
			if ucb > bestUCB:
				bestChild = child
				bestUCB = ucb
		return bestChild

	def get_ucb(self, child):
		if child.visitCount == 0:
			qValue = 0
		else:
			qValue = 1 - ((child.valueSum / child.visitCount) + 1) / 2
		return qValue + self.args['C'] * math.sqrt(self.visitCount) / (child.visitCount + 1) * child.prior

	def expand(self, policy):
		for action, prob in enumerate(policy):
			if prob > 0:
				childState = self.state.copy()
				childState = self.game.get_next_state(childState, action)
				childState = self.game.change_perspective(childState)

				child = Node(self.game, self.args, childState, self, action, prob)
				self.children.append(child)

	def backpropagate(self, value):
		self.valueSum += value
		self.visitCount += 1

		if self.parent is not None:
			self.parent.backpropagate(value)


class MCTS:
	def __init__(self, game, args, model):
		self.game = game
		self.args = args
		self.model = model

	@torch.no_grad()
	def search(self, state):
		root = Node(self.game, self.args, state)

		for search in range(self.args['numSearches']):
			node = root

			while node.is_fully_expanded():
				node = node.select()
			assert not node.is_fully_expanded(), f"Selection Bug"

			value, terminated = self.game.check_game_over(node.state)

			if not terminated:
				policy, value = self.model(torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0))
				policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
				validMoves = self.game.get_valid_moves(node.state)
				policy = policy * validMoves
				policy = policy / np.sum(policy)
				value = value.item()
				node.expand(policy)
			node.backpropagate(value)

		actionProbs = np.zeros(self.game.actionSpace)
		for child in root.children:
			actionProbs[child.actionTaken] = child.visitCount
		actionProbs = actionProbs / np.sum(actionProbs)
		return actionProbs