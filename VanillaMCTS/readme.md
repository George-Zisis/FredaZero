### Vanilla MCTS

Chose six random fens for the evaluation, with different numbers of pieces for each fen
```
fen_26 = "2R2n2/Nqrp4/P2PP2R/pP2b3/2Bp3p/1pp1P1k1/2PP1N1p/2K3B1 w - - 0 1"
fen_22 = "2R5/P1r3p1/2bq1PQ1/7K/P1p4b/BPk1pr1P/1p1N2P1/3B4 w - - 0 1"
fen_16 = "8/2P2p2/2P1P3/1k3P2/1p4B1/2q3Q1/Kp2r1P1/R3N3 w - - 0 1"
fen_12 = "2N5/2KBp1P1/7n/4B3/3N4/1R4r1/6kP/8 w - - 0 1"
fen_8 = "3k4/8/6R1/5Pr1/4K3/8/P3P3/N7 w - - 0 1"
fen_4 = "7r/8/7P/1K1k4/8/8/8/8 w - - 0 1"
```
The evaluation was performed with 100, 1000 and 10000 number of tree searches

The tree search returns a list that maps a probability to each valid action for the specific fen

Even for 10000 number of searches, the probabilities given for all the different fens were almost uniformly distributed

The search was also really slow, it took from 15 to 3 minutes to perform a 10000 depth search depending on the number of pieces on the board at given fen

Given the mean number of turns for a random action game is approximately 330 turns, and Vanilla MCTS follows an almost random policy at 10000 number of searches, it would take around two days to finish one complete chess match

In conclusion, the game of chess is too complex of a game to solve with vanilla MCTS

### Random chess games

Run 100 games of chess, from the starting fen to a terminal state, choosing random actions

Took approximately 14 seconds for all games to run and a mean of 335 turns for each game
85% of the games ended in a draw 

When comparing actions to stockfish's best action only 13.12% of the actions chosen randomly were the optimal actions

When comparing actions it took much longer to train ~ 2 minutes


> Written with [StackEdit](https://stackedit.io/).
