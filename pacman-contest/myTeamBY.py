# myTeamBY.py
# ---------
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


from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game

from util import nearestPoint

#################
# Team creation #
#################

# BY: Example command line: "args": ["-r", "myTeamBY", "-b", "baselineTeam",  "-n", "2", "-x", "1"]

def createTeam(firstIndex, secondIndex, isRed,
               first = 'QLearningOffensive', second = 'ReflexCaptureAgent',
               numTraining=None):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions

  This is similar to the one in 'baselineTeam'.
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
   
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}



class QLearningAgent(ReflexCaptureAgent):

  def __init__(self, index, timeForComputing = .1):

    print('========================================== QLearningAgent.__init__ was called !!!!!')
    # Agent index for querying state
    self.index = index

    # Whether or not you're on the red team
    self.red = None

    # Agent objects controlling you and your teammates
    self.agentsOnTeam = None

    # Maze distance calculator
    self.distancer = None

    # A history of observations
    self.observationHistory = []

    # Time to spend each turn on computing maze distances
    self.timeForComputing = timeForComputing

    # Access to the graphics
    self.display = None

    ###########################
    #  Keep track of the Q-table
    ###########################
    self.qTable = util.Counter() # this maps (state, action) pair to real numbers

    # Keep track of the learning process
    self.episodesSoFar = 0
    '''numTraining - number of training episodes, i.e. no learning after these many episodes'''
    self.numTraining = 1
    self.accumTrainRewards = 0.0
    self.accumTestRewards = 0.0

    # Whether print the result or not:
    self.print = False
    self.printMore = False




  def registerInitialState(self, gameState): # This will be called everytime a new game started.
                                             # Whereas __init__ will be called only once.
    # Keep track of the learning process
    self.episodesSoFar += 1
    if self.episodesSoFar > self.numTraining:
      self.printMore = True
              
    print("看这里！: registerInitialState for QLearningAgent")
    ReflexCaptureAgent.registerInitialState(self, gameState)

    # BY: TODO: load qTable
    self.usefulStates = None # BY: perhaps extract useful info from the humongous GameState

    # Store the last (state, action) for updating Q-Table
    self.lastState = None
    self.lastAction = None 
    self.lastInfo = util.Counter() # this keeps track of some useful stuff (num_food, num_food_enemy...)

    self.currentInfo = util.Counter()

    # learning parameters:
    ''' alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor'''
    self.alpha = 0.2
    self.gamma = 0.9
    self.epsilon = 0.1

    

    print()
    print('Length of state at the beginning', len(self.qTable))



  def chooseAction(self, gameState):
    """
    BY: This should be quite straight forward, in exploitation, best action is the one with highest qValue.
        Otherwise, in exploration, some stochasticity involved
    """
      
    actions = gameState.getLegalActions(self.index)
    qValues = [self.getQValue(gameState, action) for action in actions]
    bestActions = [a for a, v in zip(actions, qValues) if v == max(qValues)]
    
    

    # self.getMazeDistance(self.start, pos2)
    
    ## Some fun stats to keep track of  
    self.pos = gameState.getAgentPosition(self.index)
    self.food = self.getFood(gameState).asList()
    self.food_defend = self.getFoodYouAreDefending(gameState).asList()

    dist_to_foods = [self.getMazeDistance(self.pos, foodPos) for foodPos in self.food]
    self.minFoodDist = min(dist_to_foods)

    self.currentInfo['score'] = self.getScore(gameState)
    self.currentInfo['num_food'] = len(self.food)
    self.currentInfo['num_food_defend'] = len(self.food_defend)
    self.currentInfo['pos'] = self.pos
    self.currentInfo['minFoodDist'] = self.minFoodDist


    # BY: TODO 以后考虑exploration/exploitation， 现在只是argmax
    # HINT: You might want to use util.flipCoin(prob)
    # HINT: To pick randomly from a list, use random.choice(list)

    action_chosen = random.choice(bestActions)
    self.updateQValue(gameState, action_chosen)

    # Store data from this state as <last_data>
    self.lastState = gameState
    self.lastInfo = self.currentInfo.copy()
    self.lastAction = action_chosen

    # BY: Print to Console
    if self.printMore:
      print("--------------")
      print("Actions: {}\nQvalues: {}\nAction Chosen: {}".format(actions, qValues, action_chosen))
      print("Min Food Dist: ", self.minFoodDist)

    return action_chosen
  

  def formatGameState(self, gameState):
    """ This method is not useful for now"""
    state_formated = util.Counter()
    return state_formated


  def getReward(self, gameState, action):
    """
    Get the immediate reward R(s) / R(S,a).
    Calculated by comparing state s with the last state 
    """
    # Data for current state s
    state = gameState
    score = self.getScore(state)
    position = gameState.getAgentPosition(self.index)
    myState = gameState.getAgentState(self.index)

    # Data for last state s_t-1
    last_state = self.getPreviousObservation()
    if last_state:
      last_score = self.getScore(last_state)
    else:
      last_score = 0

            # # Data for next state s_t+1
            # next_state = state.generateSuccessor(self.index, action) # BY: ? 这里return的是一个 包含所有agent数据的gamestate吧 # Not Fair？？？ 
            # # next_state_for_agent 应该只包含了自己这个agent的信息    
            # # reward的算法其实有点’我站在state s，决定用了a，然后模拟了下一步的s‘，并在这一步更新了Q(s,a)
            # # 实际上应该的等到下一轮，等我站在s', 然后说 这个(s',s)给了我一个reward，然后用这个reward来更新上一步的Q(s,a)                                           
            # next_state_for_agent = self.observationFunction(next_state)
            # next_score = self.getScore(next_state) 
    
    # BY: TODO 现在是score相减，过会儿按sketch book上的改。

    reward = -1 + score - last_score
    return reward


  def updateQValue(self, gameState, action):
    """
    Update Q value here
    """
    reward = self.getReward(gameState, action) # this is the reward of (s,a) going into next state s‘
    qValue = self.getQValue(gameState, action)
    
    next_state = self.getSuccessor(gameState, action)
    next_actions = next_state.getLegalActions(self.index)
    next_qValues = [self.getQValue(next_state, action) for action in next_actions]
    next_qValue = max(next_qValues)

    self.qTable[(gameState, action)] = qValue + self.alpha * (reward + self.gamma * next_qValue - qValue)
    
    if self.print:
      print("QQQQQQQQQQ: ", qValue, next_qValue, qValue + self.alpha * (reward + self.gamma * next_qValue - qValue))

    return


  def getQValue(self, gameState, action):
    """ Return the Q value for a (state, action) pair
        Initialize Q value to 0.0 if (s,a) pair not in the dict
    """
    if not self.qTable[(gameState, action)]:
      self.qTable[(gameState, action)] = 0.0

    return self.qTable[(gameState, action)]

  
  def getValue(self, gameState):
    """ Returns max_action's Q(state,action)
    """
    actions = gameState.getLegalActions(self.index)
    
    if actions:
      qValues = (self.getQValue(gameState, action) for action in actions)
      return max(qValues)
    
    print("For FUN: Logically, this should never be printed... we'll see")
    return 0.0


  def getCompressedState(self, gameState):
    """ extract useful information from GameState Object, and """

    usefulState = ()

    return usefulState


  ####################################
  # Helper functions for fun
  ####################################

  def getOldObservation(self, numStepForward):
    """
    Returns the GameState object corresponding to the N-Step-forward state the agent saw.
    (the observed state of the game last time this agent moved - this may not include
    all of your opponent's agent locations exactly).

    Intuitions:
    numStepForward = 0 returns observationHistory[-1], which is current state (OR 0-step-forward)
    """
    n = len(self.observationHistory)

    if numStepForward >= n:
      return None
    else:
      return self.observationHistory[ -1 * numStepForward - 1 ] 

  def observationFunctionGodLike(self, gameState): # BY: TODO 这里实验玩玩 
    """ observe everything fully (including emeny agent)
        ONLY for fun purpose. This should remove non-deterministic aspect of the state
    """
    return super().observationFunction(gameState)

  def final(self, gameState):
    # import json

    # BY: TODO: Save the Q-Table into json
    
    print()
    print('Length of state in the end', len(self.qTable))
    print("========== Final ========== Final ========== Final ========== Final ========== Final ========== Final ==========")

    print()
    # table = self.qTable
    # new_table = {}
    # for k,v in table.items():
    #   new_table[str(k)] = v

    # with open('my_dict.json', 'w+') as f:
    #   json.dump(new_table, f)

    return



class QLearningOffensive(QLearningAgent):
  """ Basically the same as normal Q-agent
  """

  def getReward(self, gameState, action):
    """
    Get the immediate reward R(s) / R(S,a).
    Calculated by comparing state s with the last state 
    """
    
    action_cost = -1

    # Data for current state s
    score = self.currentInfo['score']
    num_food = self.currentInfo['num_food']
    minFoodDist = self.currentInfo['minFoodDist']

    # Data for last state s_t-1
    # if self.lastInfo:
    last_score = self.lastInfo['score']
    last_num_food = self.lastInfo['num_food']
    last_minFoodDist = self.lastInfo['minFoodDist']
      # last_num_food_defend = self.lastInfo['num_food_defend']


    
    # reward = action + score + num_food + dist_food
    a = score - last_score
    b = num_food - last_num_food
    c = minFoodDist - last_minFoodDist

    if self.print:
      print("REWARD CALCULATION: ", a,b,c, action_cost + 100 * a - 10 * b - 5 * c)
    return action_cost + 100 * a - 10 * b - 5 * c


class QLearningDeffensive(QLearningAgent):
  """ Basically the same as normal Q-agent
  """

  def getReward(self, gameState, action):
    """
    Get the immediate reward R(s) / R(S,a)
    Calculated by comparing state s with the last state 
    """
    action_cost = -1

    # Data for current state s
    score = self.currentInfo['score']
    # num_food = self.currentInfo['num_food']
    num_food_defend = self.currentInfo['num_food_defend']


    # Data for last state s_t-1
    if self.lastInfo:
      last_score = self.lastInfo['score']
      # last_num_food = self.lastInfo['num_food']
      last_num_food_defend = self.lastInfo['num_food_defend']

    
    # BY: TODO 现在是score相减，过会儿按sketch book上的改。

    return action_cost + 100*(score - last_score) + 10*(num_food_defend - last_num_food_defend)
























