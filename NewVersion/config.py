import argparse

from core.chess_game import ChessGame
from core.tictactoe import TicTacToe
from core.connect import ConnectFour
from core.gomoku import Gomoku

def get_game_and_args():
	parser = argparse.ArgumentParser(description="Pytorch Board Games Training")
	parser.add_argument("--env", type=str, default="TicTacToe", help="Name of the game to play")

	parser.add_argument("--cpuct", default=1.41, type=float, help="Exploration constant for ucb score")
	parser.add_argument("--temperature", default=1.0, type=float, help="Starting temperature constant")
	parser.add_argument("--dirichlet_epsilon", default=0.25, type=float, help="Dirichlet equation, epsilon consant")
	parser.add_argument("--dirichlet_alpha", default=0.3, type=float, help="Dirichlet equation, alpha or eta constant")
	parser.add_argument("--time_limit", default=5, type=int, help="Time given for a mcts search")
	parser.add_argument("--iteration", default=1, type=int,  help="Number of iterations for main pipeline")
	parser.add_argument("--num_self_play_iterations", type=int, default=16, help="Number of self play games per iteration")

	parser.add_argument("--input_channels", default=3, type=int, help="Input channels for Residual Model")
	parser.add_argument("--filters", default=128, type=int, help="Number of filters for Residual Model")
	parser.add_argument("--res_blocks", default=9, type=int, help="Number of blocks for ResTower")

	parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer")
	parser.add_argument("--weight_decay", default=0.0001, type=float, help="Weight decay for optimizer")
	parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training")
	parser.add_argument("--num_epochs", default=4, type=int, help="Number of epochs for trainining")

	parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
	parser.add_argument("--parallelize", "-p", action="store_true", help="parallelize selfplay games")
	parser.add_argument("--get_dataset", "-d", action="store_true", help="start selfplay process")
	parser.add_argument("--train_model", "-t", action="store_true", help="start training the model")
	parser.add_argument("--play_vs_bot", action="store_true", help="player versus the bot")
	parser.add_argument("--bot_vs_previous_bot", action="store_true", help="bot versus the bot of the previous model")

	args = parser.parse_args()

	if args.env == "Gomoku":
		game = Gomoku()
	elif args.env == "ConnectFour":
		game = ConnectFour()
	elif args.env == "ChessGame":
		game = ChessGame()
	else:
		game = TicTacToe()

	return game, args