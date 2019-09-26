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

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

# by KB
import operator

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'OffensiveReflexAgent'):
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
  behavior is what you want for the nightly contest..
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
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
    start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))

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
    Computes Q values
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

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}











  class KBQLearningAgent(CaptureAgent):

    def __init__(self, index, timeForComputing = .1, numTraining = 0, epsilon = 0.6, alpha = 0.8, discount = 0.8):
      CaptureAgent.__init__(self,index,timeForComputing)

      #Q-value calculation parameters
      self.epsilon = float(epsilon) # exploration rate
      self.alpha = float(alpha) # learning rate
      self.discount = float(discount) # discount rate

      self.qvalues = util.Counter()  # Q-values, key: (gameState,action)

      # episode
      self.numTraining = int(numTraining)
      self.currentEpisode = 0
      self.currentEpisodeReward = 0.0
      self.totalTrainRewards = 0.0
      self.totalTestRewards = 0.0



    def registerInitialState(self, gameState):
      CaptureAgent.registerInitialState(self, gameState)
      self.start = gameState.getAgentPosition(self.index)
      
      # startTime = time.time()

      # episode start
      self.lastState = None
      self.lastAction = None
      self.currentEpisodeReward = 0.0
     

      # get team mate index
      teamIndex = self.getTeam(gameState)
      if self.index == teamIndex[0]:
        self.teamMateIndex = teamIndex[1]
      else:
        self.teamMateIndex = teamIndex[0]

    
    #overriding, this func is called on every movement 
    def observationFunction(self, currentGameState):
      if self.lastState:
        # find the score change between each step
        rewardChange = (currentGameState.getScore() - self.lastState.getScore())
        # update 1-step Q values
        self.observeTransition(self.lastState, self.lastAction, currentGameState, rewardChange)
      return CaptureAgent.observationFunction(self.index, currentGameState)

    def observeTransition(self, gameState, action, nextState, newReward):
      self.currentEpisodeReward += newReward
      self.update(gameState, action, nextState,newReward)
    
    def update(self, gameState, action, nextState, reward):
        self.qvalues[(gameState, action)] = (1 - self.alpha) * self.qvalues[(gameState, action)] \
                    + self.alpha * (reward + self.discount * self.compValueFromQ(nextState))

    def compValueFromQ(self, gameState):
      actions = gameState.getLegalActions(self.index)
      
      if actions:
        values = [self.getQValue(gameState,a) for a in actions]
        return max(values)
      else:
        return 0.0
      
    def getQValue(self, gameState, action):  
      return self.qvalues[(gameState,action)]


      
    # Overiding
    def chooseAction(self, gameState):
      legalActions = gameState.getLegalActions(self.index)
      action = None
      if legalActions:
        if util.flipCoin(self.epsilon):
          action = random.choice(legalActions)
        else:
          action = self.compActionFromQ(gameState)
      
      # save it as the last snapshot
      self.snapshot(gameState,action)
      return action

    def compActionFromQ(self, gameState): #Compute the best action   
      legalActions = gameState.getLegalActions(self.index)

      if legalActions == None:
         return None
      
      actionList = {}
      for action in legalActions:
        actionList[action] = self.getQValue(gameState,action)
      
      sorted_actions = sorted(actionList.items(),key = operator.itemgetter(1))
      sorted_actions.reverse()

      return sorted_actions[0][0]


    def snapshot(self, gameState, action):
      self.lastState = gameState
      self.lastAction = action
    



    #overriding, this func is called at the end of each game
    def final(self, gameState):
      # update on last movement 
      rewardChange = (gameState.getScore() - self.lastState.getScore())
      self.observeTransition(self.lastState, self.lastAction, gameState, rewardChange)
    
      #clear observation history list
      CaptureAgent.final(self, gameState)

      # stop episode
      self.endOneEpisode()




    def endOneEpisode(self):
      self.currentEpisode += 1
      # collect rewards for training/testing
      if self.currentEpisode < self.numTraining:
        self.totalTrainRewards += self.currentEpisodeReward
      else:
        self.totalTestRewards += self.currentEpisodeReward
        self.epsilon = 0.0 # no exploration
        self.alpha = 0.0 # no learning

      


    def getTeamMateInfo(self,gameState):
            # get team mate index
      teamIndex = self.getTeam(gameState)
      teamMateIndex = teamIndex[0]
      if self.index == teamIndex[0]:
        teamMateIndex = teamIndex[1]
     
      teamMateLocation = gameState.getAgentPosition(teamMateIndex)
      return teamMateIndex, teamMateLocation

    def getPossibleActions(position, walls):
      possible = []
      x, y = position
      for dir, vec in Actions._directionsAsList:
          dx, dy = vec
          next_y = y + dy
          next_x = x + dx
          if not walls[next_x][next_y]: possible.append(dir)
      return possible
       
 


      
      

# # get probability distribution
      # if self.index == teamIndex[0]:
      #   self.directionProb = [0.3, 0.4, 0.2, 0.1] # north, east, south, west
      # else:
      #   self.directionProb = [0.2, 0.4, 0.3, 0.1] # north, east, south, west

      # # set initial Q values for the first 15 seconds
      # self.startMap = initializeMap(gameState)

      # # iterate across all grid cell to calculate Q values
      # walls = gameState.getWalls()
      # while time.time() - startTime < 14:
      #   for i in len(startMap):
      #     for j in len(startMap[0]):
      #       values = []
      #       loc = (i,j)
            
      #       # for each possible action
      #       for action in getLegalActions(loc, walls):
      #         values.append(computeQValue(loc,action))
            
      #       next_values[loc] = max(values)

    

      # def initializeMap(self, gameState):
      #   w= self.gameState.data.layout.mapWidth
      #   l = len(self.gameState.data.layout)
      #   map =  [[0 for col in range(w)] for row in range(l)]

      #   # assign intial reward
      #   foods = self.getFood(gameState)
      #   capsules = self.getCapsules(gameState)
      #   for i in l:
      #     for j in w:
      #       if food[i][j]:
      #         map[i][j] = self.foodReward
      #       if capsules[i][j]:
      #         map[i][j] = self.capsReward
      #   return map


