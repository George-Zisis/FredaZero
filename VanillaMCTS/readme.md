### Random chess games

Run 100 games of chess, from the starting fen to a terminal state, choosing random actions

It took approximately 14 seconds for all games to run and a mean of 335 plies for each game

85% of the games ended in a draw 

When comparing actions to stockfish's best action only 13.12% of the actions chosen randomly were the optimal actions

When comparing actions it took much longer to train ~ 2 minutes

### Vanilla MCTS

Implementation of the vanilla MCTS algorithm for the game of chess

5 runs with number of searches set to 100 and C to 2 takes 120 minutes to run with each game taking around 24 minutes to finish, an average of 105 plies and all games ending in a draw

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

MCTS cannot find the optimal move 

It takes around 7 seconds per search iteration, because it starts from initial state and the simulation takes longer to terminate


