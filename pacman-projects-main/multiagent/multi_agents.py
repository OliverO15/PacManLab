# multi_agents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattan_distance
from game import Directions, Actions
from pacman import GhostRules
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states]
        
        "*** YOUR CODE HERE ***"
        # print(current_game_state)
        # print(successor_game_state)
        # print(new_pos)
        # print(get_food_positions(new_food))
        # print(successor_game_state.get_ghost_positions())
        # print(new_scared_times)
        
        if(min(new_scared_times) > 0):
            ghost_factor = 0
        else:
            ghost_factor = 5
        food_factor = -1
               
        if(action=="Stop"):
            return float('-inf')
        
        # give penalty for small distance to ghosts
        min_dist_ghosts = min([util.manhattan_distance(new_ghost_pos, new_pos) for new_ghost_pos in successor_game_state.get_ghost_positions()])
        
        # give bonus if nearer to food or food eaten
        if(len(get_food_positions(current_game_state.get_food())) > len(get_food_positions(successor_game_state.get_food()))):
            # bonus fo food eaten
            food_score = -1
        else:
            food_score = min([util.manhattan_distance(food_pos, new_pos) for food_pos in filter_foods(get_food_positions(new_food), current_game_state)], default=0)
        
        return   food_score * food_factor - 1/(min_dist_ghosts + 0.000001) * ghost_factor

def filter_foods(food_list, game_state):
    """filter out foods with direct path involving an illegal move as first move"""
    pos = game_state.get_pacman_position()
    ret = []
    for food in food_list:
        first_moves = set()
        if food[0] > pos[0]:
            first_moves.add('East')
        elif food[0] < pos[0]:
            first_moves.add('West')
        if food[1] > pos[1]:
            first_moves.add('North')
        elif food[1] < pos[1]:
            first_moves.add('South')
        if first_moves <= set(game_state.get_legal_actions()):
            ret.append(food)
            
    return ret

def get_food_positions(food_grid):
    """get positions of all foods as coordinates"""
    food_positions = []
    for x in range(food_grid.width):
        for y in range(food_grid.height):
            if food_grid[x][y]:
                food_positions.append((x, y))
    return food_positions

def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth) 

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generate_successor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.get_num_agents():
        Returns the total number of agents in the game

        game_state.is_win():
        Returns whether or not the game state is a winning state

        game_state.is_lose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        legal_actions = game_state.get_legal_actions(0)
        max_action = max(legal_actions, key=lambda action: self.minimax(game_state.generate_successor(0, action), 0, 1))
        return max_action

    def minimax(self, game_state, current_depth, agent_index):
        next_depth = current_depth
        next_agent = agent_index
        if agent_index == game_state.get_num_agents() -1:
            # last eval for this depth => next depth in next call
            next_agent = 0
            next_depth += 1
        else:
            next_agent += 1
        if current_depth == self.depth or game_state.is_win() or game_state.is_lose():
            return self.evaluation_function(game_state)
        action_evals = [self.minimax(game_state.generate_successor(agent_index, action), next_depth, next_agent) for action in game_state.get_legal_actions(agent_index)]
        if agent_index == 0:
            # pacman moves (MAX)
            return max(action_evals)
        else:
            # one of the ghosts moves (MIN)
            return min(action_evals)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

def better_evaluation_function(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()
    


# Abbreviation
better = better_evaluation_function
