import numpy as np
import chess
from tqdm import tqdm
from mcts import Node, MCTS

class ChessEnv:
	def __init__(self):
		self.board = None
		self.winner = None
		self.result = None 
		self.resigned = False

	def new_game(self):
		self.board = chess.Board()
		self.winner = None
		self.result = None 
		self.resigned = False

	def get_state_fen(self):
		return self.board.fen()

	def get_valid_moves(self, state):
		assert self.board.fen() == state, "In get_valid_moves: not the same fens!"
		valid_moves = list(self.board.legal_moves)
		return valid_moves

	def resign(self, action):
		if action is None and self.board.result() == "*":
			self.resigned = True
			if self.board.turn == chess.WHITE:
				self.winner = "BLACK"
				self.result = "0-1"
				return -1, True
			else:
				self.winner = "WHITE"
				self.result = "1-0"
				return 1, True

	def get_value_and_terminated(self, state):
		self.result = self.board.result(claim_draw=True)
		if self.result == "1-0":
			self.winner = "WHITE"
			return 1, True
		if self.result == "0-1":
			self.winner = "BLACK"
			return -1, True
		if self.result == "1/2-1/2":
			self.winner = "DRAW"
			return 0, True 
		if self.result == "*":
			return 0, False

"""Test the environment using vanilla MCTS"""
game = ChessEnv()
args = {
		'C':2,
		'num_searches':100
}
mcts = MCTS(game, args)

counter = 1
game.new_game()
state = game.get_state_fen()
for i in range(10):
	print(f"This is the number {counter} turn")
	valid_moves = game.get_valid_moves(state)
	p = mcts.search(state)
	best_action = np.argmax(p)
	action = valid_moves[best_action]
	game.board.push(action)
	value, is_terminal = game.get_value_and_terminated(state)
	if is_terminal:
		break
	state = game.get_state_fen()
	counter += 1 

print(f"Winner:{game.winner}, Result:{game.result}, Resigned:{game.resigned}")
print(game.board)
