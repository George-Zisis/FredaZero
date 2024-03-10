import numpy as np
import chess
import chess.engine
import action_mod

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

	def get_initial_state(self):
		return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

	def get_fen_state(self):
		return self.board.fen()

	def get_move_table(self):
		all_moves_str = action_mod.create_uci_labels()
		all_moves_int = action_mod.create_uci_labels()

		valid_moves_str = self.get_valid_moves()
		for idx, move in enumerate(all_moves_str):
			if move not in valid_moves_str:
				all_moves_str[idx] = 0
				all_moves_int[idx] = 0
			if move in valid_moves_str:
				all_moves_int[idx] = 1 
		return all_moves_str, all_moves_int

	def correct_policy(self, action_probs):
		all_moves_str, all_moves_int = self.get_move_table()
		count = 0
		for idx, move in enumerate(all_moves_int):
			all_moves_int[idx] = action_probs[count] if move == 1 else 0
			count = count + 1 if move == 1 else count
		return all_moves_int

	def get_valid_moves(self):
		valid_moves_obj = list(self.board.legal_moves)
		valid_moves_str = [str(obj) for obj in valid_moves_obj]
	
		return valid_moves_str

	def get_all_action_size(self):
		return len(action_mod.create_uci_labels())

	def get_action_size(self):
		return len(list(self.board.legal_moves))

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

	def is_flip(self, fen):
		return True if fen.split()[1] == 'b' else False 

	def eval_stockfish(self, fen):
		engine = chess.engine.SimpleEngine.popen_uci("stockfish")

		self.board.set_fen(fen)

		info = engine.analyse(self.board, chess.engine.Limit(time=0.1))
		print("Score:", info['score'])
		engine.quit()
