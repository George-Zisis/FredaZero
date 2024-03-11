### Random chess games

Run 100 games of chess, from the starting fen to a terminal state, choosing random actions

It took approximately 14 seconds for all games to run and a mean of 335 plies for each game

85% of the games ended in a draw 

When comparing actions to stockfish's best action only 13.12% of the actions chosen randomly were the optimal actions

When comparing actions it took much longer to train ~ 2 minutes

### Vanilla MCTS

Implementation of the vanilla MCTS algorithm for the game of chess

- 5 runs with number of searches set to 100 and C to 2 takes 120 minutes to run with each game taking around 24 minutes to finish, an average of 105 plies and all games ending in a draw

If the game starts from initial state and takes 10 random plies:

```
First Iteration:
Stockfish actions:['e2e4', 'd7d5', 'e2e4', 'd7d5', 'e2e4', 'e7e5', 'e2e4', 'f5e4', 'e4f5', 'd7d6']
MCTS actions: ['g1h3', 'g8h6', 'g1h3', 'g8h6', 'c3d5', 'g8h6', 'g1h3', 'g8h6', 'g1h3', 'g8h6']
```
```
Second Iteration:
Stockfish actions:['e2e4', 'e7e5', 'g1f3', 'g7g6', 'd2d4', 'd7d5', 'e2e4', 'h6f5', 'e2e4', 'h6f5']
MCTS actions: ['g1h3', 'g8h6', 'g1h3', 'h8g8', 'g1h3', 'h8g8', 'h1h2', 'h8g8', 'h1h2', 'h8g8']
```
There are a lot of the same moves picked over and over again from MCTS

The board is cycling through the same states over and over again

- Run the game for ten steps starting from a different state
```
fen_26 = "2R2n2/Nqrp4/P2PP2R/pP2b3/2Bp3p/1pp1P1k1/2PP1N1p/2K3B1 w - - 0 1"

Stockfish actions:['g1h2', 'h2g1q', 'g1h2', 'h2g1q', 'c1d1', 'b2b1r', 'f8g8', 'b2b1q', 'c8g8', 'g3g4']
MCTS actions: ['c8f8', 'f8h7', 'c8f8', 'b7c8', 'c1d1', 'b7c8', 'f8h8', 'b7c8', 'c8h8', 'g3g4']

fen_22 = "2R5/P1r3p1/2bq1PQ1/7K/P1p4b/BPk1pr1P/1p1N2P1/3B4 w - - 0 1"

Stockfish actions:['g6c2', 'd6g6', 'g6d6', 'd6e5', 'a3b2', 'c7h7', 'h7h6', 'f3f5', 'h6d6', 'f3f5']
MCTS actions: ['c8h8', 'c7c8', 'c8h8', 'c7c8', 'c8h8', 'c7c8', 'h5h6', 'c7c8', 'g8e7', 'c8g8']

fen_16 = "8/2P2p2/2P1P3/1k3P2/1p4B1/2q3Q1/Kp2r1P1/R3N3 w - - 0 1"

Stockfish actions:['g4e2', 'b2a1q', 'a2b1', 'e2b2', 'e1d3', 'e2e1', 'f4b4', 'e6c6', 'f4b4', 'e3e4']
MCTS actions: ['g4h5', 'b5c6', 'a2b1', 'b5c6', 'b1c1', 'b5c6', 'g4h5', 'e6e8', 'g4h5', 'b5b6']

fen_12 = "2N5/2KBp1P1/7n/4B3/3N4/1R4r1/6kP/8 w - - 0 1"

Stockfish actions:['b3g3', 'g3g4', 'b3c3', 'c3c6', 'c7c6', 'c6c5', 'b3g3', 'f6g6', 'e5f6', 'f2g2']
MCTS actions: ['c8e7', 'h6g8', 'c7d8', 'g2h3', 'c7d8', 'h6g8', 'c8e7', 'h6g8', 'c8e7', 'h6g8']

fen_8 = "3k4/8/6R1/5Pr1/4K3/8/P3P3/N7 w - - 0 1"

Stockfish actions:['g6g5', 'd8e7', 'd6d2', 'g5g8', 'a1c2', 'g3e3', 'f5f6', 'b8c7', 'e2e4', 'g3b3']
MCTS actions: ['g6g8', 'd8e8', 'd6d8', 'c7c8', 'd6d8', 'c7c8', 'd6d8', 'b8c8', 'f6f7', 'a7b8']

fen_4 = "7r/8/7P/1K1k4/8/8/8/8 w - - 0 1"

Stockfish actions:['b5b4', 'd5c4', 'a4a3', 'h8h6', 'a3b4', 'd4c3', 'a2a3', 'd4d3', 'b2b3', 'd3c2']
MCTS actions: ['b5b6', 'h8g8', 'a4a5', 'h8g8', 'a3b4', 'h8g8', 'a2b3', 'd8h8', 'b2b3', 'd8h8']

```

The search speed for the state with only 4 pieces left on the board is approximately 40 iterations per second in contrast with the 26 pieces state which is 10 iterations per second

MCTS manages to find the best move 2 times, 1 time and 2 times for the `fen_26`, `fen_16` and `fen_4` starting states respectively 

MCTS cannot find the optimal move 

It takes around 7 seconds per search iteration, because it starts from initial state and the simulation takes longer to terminate


