# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='SwitchingPatrolAgent', second='SwitchingPatrolAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class DynamicPatrolAgent(CaptureAgent): 
    ''' 
    A base class for dynamic patrol agents that choose score-maximizing actions
    Based on the ReflexCaptureAgent from baselineTeam.py
    '''

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.has_food = False
    
    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
  
    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a) depending on the current strategy.
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]


        
        return random.choice(best_actions)

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}
    
    
    def get_features(self, game_state, action):
      '''Returns a counter of features for the state'''
      features = util.Counter()
      successor = self.get_successor(game_state, action)
      features['successor_score'] = self.get_score(successor)
      return features
    
    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights
    
    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor


class SwitchingPatrolAgent(CaptureAgent):
    """
    A class representing a switching patrol agent in a Pacman game.

    This agent switches between offensive and defensive strategies based on the game state and current conditions.
    It uses the DefensivePatrolling and OffensivePatrolling strategies to implement its behavior.

    Attributes:
    - index: The index of the agent.
    - time_for_computing: The time limit for computing an action.
    - start: The starting position of the agent.
    - current_strategy: The current strategy being used by the agent.
    - has_food: A flag indicating whether the agent has food.
    - is_in_lead: A flag indicating whether the agent is in the lead.
    - defensive_strategy: An instance of the DefensivePatrolling strategy.
    - offensive_strategy: An instance of the OffensivePatrolling strategy.
    - food_list: A list of food positions in the game.

    Methods:
    - register_initial_state(game_state): Registers the initial state of the agent.
    - choose_action(game_state): Chooses an action based on the current strategy.
    - get_weights(game_state, action): Returns weights depending on the current strategy.
    - get_features(game_state, action): Returns features depending on the current strategy.
    - evaluate(game_state, action): Computes a linear combination of features and feature weights.
    - get_successor(game_state, action): Finds the next successor which is a grid position.
    - check_if_in_lead(game_state): Returns True if the agent is in the lead.
    - check_for_invaders(successor): Returns True if there are any invaders, meaning enemies that are in the agent area.
    - enemies_near(game_state): Returns True if there are any enemies near the agent.
    - should_switch_strategy(game_state, action): Returns True if the agent should switch to another strategy.
    - switch_to_offensive(): Switches to the offensive strategy.
    - switch_to_defensive(): Switches to the defensive strategy.
    - switch_strategy(): Switches to the other strategy.
    """
    
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.current_strategy = None
        self.has_food = False
        self.is_in_lead = False
        self.defensive_strategy = DefensivePatrolling(self.index)
        self.offensive_strategy = OffensivePatrolling(self.index)

    def register_initial_state(self, game_state):
        """
        Registers the initial state of the agent. 
        With the instance of the defensive and offensive strategy.
        """
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.defensive_strategy.register_initial_state(game_state)
        self.offensive_strategy.register_initial_state(game_state)
        self.food_list = self.get_food(game_state).as_list()

    def choose_action(self, game_state):
        """
        Picks among the actions depending on the current strategy.
        """
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)

        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # check if we ate food as a result of the offensive strategy
        if len(self.food_list) > len(self.get_food(game_state).as_list()):
            self.has_food = True

        # Once we ate food we want to bring it back fast to our own area
        # Then we are in lead, if the enemy scores points we want to get another food pellet and bring it back
        # for that we save in a variable if we get in lead after bringing a food pellet back
        if self.has_food and self.check_if_in_lead(game_state) and not self.is_in_lead:
            self.is_in_lead = True

        # If we arent in lead anymore we want to get another food pellet and bring it back
        if self.has_food and not self.check_if_in_lead(game_state) and self.is_in_lead:
            self.is_in_lead = False
            self.has_food = False
            self.food_list = self.get_food(game_state).as_list()

        # if we ate food but got killed before we could bring it back we want to get another food pellet and bring it back
        # check if both agents are non pacman and if we have food
        team_indices = self.get_team(game_state)
        if not game_state.get_agent_state(team_indices[0]).is_pacman and not game_state.get_agent_state(
                team_indices[1]).is_pacman and self.has_food and not self.check_if_in_lead(game_state):
            self.has_food = False
            self.food_list = self.get_food(game_state).as_list()

        return random.choice(best_actions)

    def get_weights(self, game_state, action):
        """
        Returns weights depending on the current strategy.
        """
        return self.current_strategy.get_weights(game_state, action)

    def get_features(self, game_state, action):
        """
        Returns features depending on the current strategy.
        """
        return self.current_strategy.get_features(game_state, action)

    def evaluate(self, game_state, action):
        """
        Checks if the agent should switch to another strategy and computes a linear combination of features and feature weights.
        """
        if self.should_switch_strategy(game_state, action):  # Switch strategy if necessary
            self.switch_strategy()

        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def check_if_in_lead(self, game_state):
        """
        Returns true if the agent is in lead.
        """
        if self.get_score(game_state) > 0:
            return True
        return False

    def check_for_invaders(self, successor):
        """
        Returns true if there are any invaders, meaning enemies that are in the agent area.
        """
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        if len(invaders) > 0:
            return True
        return False

    def enemies_near(self, game_state):
        """
        Returns true if there are any enemies near the agent (within 10 noisy steps).
        """
        # If Agents are observable (within 5) return true
        for enemy in self.get_opponents(game_state):
            if game_state.get_agent_position(enemy) is not None:
                return True

        # If Agents are not observable (not within 5) check how far away they are with noisy distance
        # If distance is smaller than 10 return true

        noisy_distances = game_state.get_agent_distances()
        for enemy in self.get_opponents(game_state):
            if noisy_distances[enemy] < 10:
                return True
        return False

    def should_switch_strategy(self, game_state, action):
        """
        Returns true if the agent should switch to another strategy.
        """
        successor = self.get_successor(game_state, action)

        if self.current_strategy == None:
            # If no strategy is set, set to offensive strategy
            self.switch_to_offensive()
            return False

        if self.current_strategy == self.defensive_strategy and not self.check_if_in_lead(
                game_state) and not self.check_for_invaders(game_state) and not self.enemies_near(game_state) and not self.has_food:
            # Switch to offensive strategy if the agent is not in lead and there are no invaders and the enemies are not near
            return True

        if self.current_strategy == self.offensive_strategy and (
                self.check_for_invaders(successor) or self.enemies_near(game_state)):
            # Switch to defensive strategy if there are invaders or enemies are near

            return True

        return False

    def switch_to_offensive(self):
        self.current_strategy = self.offensive_strategy

    def switch_to_defensive(self):
        self.current_strategy = self.defensive_strategy

    def switch_strategy(self):
        if self.current_strategy == self.defensive_strategy:
         
            self.switch_to_offensive()
        else:
            self.switch_to_defensive()
    



class DefensivePatrolling(DynamicPatrolAgent): 
    """ DynamicPatrolAgent that implements a dynamic patrol strategy, where an agent is patrolling at the border, dividing the 
    border into two parts for the agent and its partner."""
   

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()


        # Feature: Being defensive
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = -50

        # Determine if there are any invaders that need to be hunted down
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        
        # Feature: Number of invaders
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
           if self.distancer is not None:
                dists= [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                features['invader_distance'] = min(dists)
         
               
        # Feature: Patrolling at border
        features['patrolling_at_border'] = self.patrolling_at_border(game_state, action)
        
        # Feature: Not stopping / reversing 
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features 
    
    def patrolling_at_border(self, game_state, action):
        '''Returns a high value if the agent is patrolling at the border, dividing the border into two parts for the agent and its partner'''
        border_threshold = 5
       
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # To divide the border into two parts for the two agents we need to know which inidices are in our team
        team_indices = self.get_team(game_state)
        

        full_y_area = int(game_state.get_walls().height)
        first_agent_y_area = int(full_y_area / 2) 

        # get border position
        if self.red:
            border_x = int(game_state.get_walls().width / 2) - 1
        else:
            border_x = int(game_state.get_walls().width / 2)

        # Determin the distance to the border
        # Only the x coordinate is relevant
        border_pos = ()

        # Include all the y coordinates that are in the border area and create a list of all possible border positions
        if self.index == team_indices[0]:
            for y in range(first_agent_y_area):
                if not game_state.has_wall(border_x, y):
                    border_pos = border_pos + ((border_x, y),)
        else:
            for y in range(first_agent_y_area , full_y_area):
                if not game_state.has_wall(border_x, y):
                    border_pos = border_pos + ((border_x, y),)
        

        # Check if the agent is near to any of the border positions and return the distance to the nearest border position
        distance_to_border = min([self.get_maze_distance(my_pos, border) for border in border_pos])
     
        # If the agent is near the border return a high value
        if distance_to_border < border_threshold:
            return 5
        
        return -distance_to_border # Make it negative so that the agent is rewarded for being near the border

    def get_weights(self, game_state, action):
        return {'num_invader' : -1000, 
                'on_defense': 100, 
                'invader_distance': -10, 
                'patrolling_at_border': 500, 
                'stop': -100, 
                'reverse': -2}



class OffensivePatrolling(DynamicPatrolAgent): 
    """
    A dynamic patrol agent that carefully tries to get only a few pellets at a time and then returns to the border to patrol again."""

     

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
      

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # Feature: Distance to nearest enemy
        nearest_enemy_distance = self.calculate_nearest_enemy_distance(game_state, my_pos, action)
        features['nearest_enemy_distance'] = nearest_enemy_distance

        # Feature: Careful offense
        features['careful_offense'] = 1
        if not my_state.is_pacman: features['careful_offense'] = 0

        # Feature: Not stopping / reversing
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # Feature: Eating closest food
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)


        if not self.has_food:
            my_pos = successor.get_agent_state(self.index).get_position()
            if len(food_list) > 0:  
                min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
                features['distance_to_food'] = min_distance
        else:
            # Get back home (star_position) if we have food
            features['distance_to_home'] = self.get_maze_distance(my_pos, self.start)

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000,
                'nearest_enemy_distance': -100, 
                'enemies_in_own_area': -1000, 
                'careful_offense': 100, 
                'stop': -100, 'reverse': -2, 
                'distance_to_food': -1, 
                'distance_to_home': -1000, 
                'successor_score': 100}
    
    def calculate_nearest_enemy_distance(self, game_state, my_pos, action):
        '''Calculates the distance to the nearest enemy agent'''
        successor = self.get_successor(game_state, action)

        # When enemies are near we know their exact position
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        near_enemies = [a for a in enemies if a.get_position() is not None]

        # If enemies are not near we only know their noisy distance
        enemy_indices = self.get_opponents(game_state)
        noisy_distances = game_state.get_agent_distances()
        noisy_enemies = [noisy_distances[i] for i in enemy_indices]
        
        # Check if enemies are near and return noise distance if not 
        if len(near_enemies) > 0:
            distances = [self.get_maze_distance(my_pos, a.get_position()) for a in near_enemies]
        else :
            distances = noisy_enemies
        return min(distances)
    
      
