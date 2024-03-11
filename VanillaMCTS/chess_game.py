import numpy as np
import chess
import chess.engine

from action_mod import create_uci_labels

class Game:
	def __init__(self):
		self.winner = None
		self.result = None

	def new_game(self):
		self.winner = None
		self.result = None
		return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

	def player(self, state):
		return "BLACK" if state.split()[1] == 'b' else "WHITE"

	def flip(self, state):
		return True if self.player(state) == "BLACK" else False

	def get_board(self, state):
		board = chess.Board()
		board.set_fen(state)
		return board

	def get_result(self, state):
		board = self.get_board(state)
		result = board.result(claim_draw=True)
		return result 

	def game_over(self, state):
		self.result = self.get_result(state)
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

	def get_next_state(self, state, action):
		board = self.get_board(state)
		board.push(chess.Move.from_uci(action))
		return board.fen()

	def get_valid_actions(self, state):
		board = self.get_board(state)
		return [str(obj) for obj in list(board.legal_moves)]

	def get_bin_actions(self, state):
		valid_actions = self.get_valid_actions(state)
		all_possible_actions = create_uci_labels()

		for idx, action in enumerate(all_possible_actions):
			all_possible_actions[idx] = 1 if action in valid_actions else 0
		return all_possible_actions

	def get_str_actions(self, state):
		valid_actions = self.get_valid_actions(state)
		all_possible_actions = create_uci_labels()

		for idx, action in enumerate(all_possible_actions):
			all_possible_actions[idx] = action if action in valid_actions else 0
		return all_possible_actions

	def stockfish_score(self, state):
		engine = chess.engine.SimpleEngine.popen_uci("stockfish")

		board = self.get_board(state)

		info = engine.analyse(board, chess.engine.Limit(time=0.1))

		engine.quit()

		return info['score']

	def stockfish_action(self, state):
		engine = chess.engine.SimpleEngine.popen_uci("stockfish")

		board = self.get_board(state)

		best_move = engine.play(board, chess.engine.Limit(time=0.1))

		engine.quit()

		return str(best_move.move)

	def compare_actions(self, state, action):
		return 1 if self.stockfish_action(state) == action else 0 
