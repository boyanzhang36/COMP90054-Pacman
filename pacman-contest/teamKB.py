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
from game import Actions

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'QLearningAgent', second = 'OffensiveReflexAgent'):
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

    # get team mate index
    teamIndex = self.getTeam(gameState)
    if self.index == teamIndex[0]:
      self.teamMateIndex = teamIndex[1]
    else:
      self.teamMateIndex = teamIndex[0]

    # get opponent index
    if self.red:
      self.opponentIndexes = gameState.getBlueTeamIndices()
    else:
      self.opponentIndexes = gameState.getRedTeamIndices()


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

class QLearningAgent(CaptureAgent):

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
    return CaptureAgent.observationFunction(self, currentGameState)

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
      


class OurAgent(ReflexCaptureAgent):

    def registerInitialState(self, gameState):
      self.start = gameState.getAgentPosition(self.index)
      CaptureAgent.registerInitialState(self, gameState)

      start_x, start_y = self.start  # TODO: Assuming only west vs east layout?
      if start_x < gameState.data.mapWidth/2:
        self.homeDirection = Directions.WEST  
        self.foodDirection = Directions.EAST
      else:
        self.homeDirection = Directions.EAST
        self.foodDirection = Directions.WEST
    
    def chooseAction(self, gameState):

      answer = Directions.STOP

      # Scenario - i am scared, then scape from ghost and not home
      if self.iScared(gameState):
        return self.escapeBack(gameState, false)



      # Scenario - eat capsule 
      elif not self.myCapsEaten(gameState): 
         
        # ghost nearby?
        if self.whoChaseMeNearby(gameState) is not None:
          return self.escapeBack(gameState,false) # scape but not go home

        # player 1 to eat capsule, player 2 to eat food
        if not self.opponentScaredTimeUtil[0]:
          if self.index == teamIndex[0]: # it is Player 1 
            caps = self.getMyCap(gameState)
            pathToCap = self.bfs(gameState,caps[0])
            return pathToCap[0]
          else: 
            pathToFood = self.pathToClosestFood(gameState)
            return pathToFood[0]



      # Scenario -  both player eat food as many as possible during capsule time
      elif self.myCapsEaten(gameState) \
        and self.opponentScaredTimeUtil[0] \
          and self.opponentScaredTimeUtil[1] > 3:
        pathToFood = self.pathToClosestFood(gameState)
        return pathToFood[0]



      # Scenario - deposite food 3 sec before their scared time ends 
      elif self.myCapsEaten(gameState) \
        and self.opponentScaredTimeUtil[0] \
          and self.opponentScaredTimeUtil[1] <= 3:
        return self.escapeBack(gameState, true)
  


      #Scenario  - scored outnumber others by 10 after capsule
      elif self.myCapsEaten(gameState)\
        and not self.opponentScaredTimeUtil(gameState)[0]\
          and self.getScore() > 10:
        # guard my foods 
        if gameState.getAgentState(self.index).isPacman: # go back home
          return escapeBack(gameState)

        else:
          if self.whoEatMyFoodNearby(gameState) is not None: 
            dist,loc,index  = self.whoEatMyFoodNearby(gameState)[0]
            pathToOppPac = bfs(gameState,loc)
            return pathToOppPac[0]
          else:
            if self.whereAreLostFoods(gameState) is not None:
              pathToOppPac = bfs(gameState, self.whereAreLostFoods(gameState)[0])
              return pathToOppPac[0]
            else:
              return Directions.STOP  # TODO: re-think of it
      


      #Scenario  - scored outnumber others by < 10 after capsule
      elif self.myCapsEaten(gameState) \
        and not self.opponentScaredTimeUtil(gameState)[0]\
          and self.getScore() < 10:
        pathToFood = self.pathToClosestFood(gameState)
        return pathToFood[0]
            
            
        
    def opponentScaredTimeUtil(self, gameState):
      theyScared = False
      timeLeft = gameState.getAgentState(self.opponentIndexes[0]).scaredTimer
      if timeLeft > 0:
        theyScared = True
      return (theyScared, timeLeft)

    def iScared(self, gameState):
      if gameState.getAgentState(self.index).scaredTimer >0:
        return True
      else: 
        return False

    def myCapsEaten(self, gameState):
      if len(self.getMyCap(gameState)) > 0:
        return False
      else
        return True


    # return a list of the capsules we can eat
    def getMyCap(self, gameState):
      if self.red: 
        return gameState.getRedCapsules()
      else:
        return gameState.getBlueCapsules()

    def getTeamMatePosition(self):
      teamIndex = self.getTeam(gameState)
      if (self.index == teamIndex[0]):
          teamMatePos=gameState.getAgentState(teamIndex[1]).getPosition()
      else:
          teamMatePos = gameState.getAgentState(teamIndex[0]).getPosition()
      return teamMatePos
    
    def whoEatMyFoodNearby(self, gameState):
      opponentStates = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      pacmans = [a for a in opponentStates if a.isPacman and a.getPosition() != None] # check postion to find the closest
      if len(pacmans) < 1:
        return None
      
      myPos = gameState.getAgentPosition(self.index)
      answer = [(self.getMazeDistance(myPos,a.getPosition()), a.getPosition(), a) for a in pacmans] # [(dis, pos, index)]e.g. [(6,(2,2),1),(12,(4,5),3)]
      answer.sort() # smallest first
      return answer
    
    def whoChaseMeNearby(self, gameState):
      if gameState.getAgentState(self.index).isPacman:  # I am eating others' food
        opponentStates = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        whoChaseMe = [a for a in opponentStates if not a.isPacman and a.getPosition() != None]

        myPos = gameState.getAgentPosition(self.index)
        answer = [(self.getMazeDistance(myPos,a.getPosition()),a) for a in whoChaseMe] # [(componnet index, dist)]
        answer.sort() # smallest dist first, e.g. [(6,1),(12,3)]
        return answer
      else:
        return None
    
    def whereAreLostFoods(self):
      lastState = self.getPreviousObservation()
      currentState = self.getCurrentObservation()
      oldFoods = self.getFoodYouAreDefending(lastState).aslist()
      newFoods = self.getFoodYouAreDefending(currentState).aslist()
      foodGone = list(set(oldFoods).difference(set(newFoods)))
      return foodGone # if no missing then return None
                
    def bfs(self,gameState,endLoc): #bfs
        
      toBeVisited = util.Queue()
      visited = []

      # add the initial state and current empty path to the openlist
      startLoc = gameState.getAgentPosition(self.index)
      toBeVisited.push((startLoc, []))

      while toBeVisited:
        # pop the most recently pushed item
        currentItem = toBeVisited.pop()
        currentLoc = currentItem[0]
        pathFromStart = currentItem[1]
        if currentLoc == endLoc:
          return pathFromStart
        # update visited list
        visited.append(currentLoc)
        # explore the open list of current node
        walls = gameState.getWalls()
        x, y = currentLoc
        for dir, vec in Actions._directionsAsList:
          dx, dy = vec
          next_x = x + dx
          if next_x < 0 or next_x == walls.width: continue
          next_y = y + dy
          if next_y < 0 or next_y == walls.height: continue
          if not walls[next_x][next_y]: 
            newLoc = (next_x,next_y)
            nextDirection = dir

          if newLoc not in visited:
            toBeVisited.push((newLoc,pathFromStart+[nextDirection]))

    def closestFoodPos(self, gameState):
      myPos = gameState.getAgentPosition(self.index)

      dists = [self.getMazeDistance(myPos,food) for food in self.getFood(gameState)]
      closestFood = [food for food in self.getFood(gameState) if self.getMazeDistance(myPos,food) == min(dists)]
      return random.choice(closestFood)
      
    def pathToClosestFood(self, gameState):
      food = self.closestFoodPos(gameState)
      path = self.bfs(gameState,food)
      return path # return ["north", "south"]
    
    def escapeBack(self, gameState, back = True):

      # find the direction with the farest 
      actions = gameState.getLegalActions(self.index)

      # check how many ghost approaching
      ghosts = self.whoChaseMeNearby(gameState)
      if ghosts is None:
        return random.choice(actions)
      else:
        ghost = ghosts[0] # TODO how to handle multiple ghosts
      
      maxDist = 0
      farestAction = []

      for action in actions:
        successor = gameState.getSuccessor(self.index, action)
        dist = successor.getMazeDistance(self.getPosition(), ghost)
        if dist > =  maxDist:
          maxDist = dist
          farestAction.append(action)
      

      # west/east vs north/south
      if len(farestAction) == 1:
        return farestAction[0]
      else:
        if back:  # escape back to 
          if self.homeDirection in farestAction: 
            return self.homeDirection
          elif self.foodDirection in farestAction:
            farestAction.remove(self.foodDirection)
            return random.choice(farestAction)
          else: 
            return random.choice(farestAction)
        else:  # escape to get more food
          if self.foodDirection in farestAction: 
            return self.foodDirection
          elif self.homeDirection in farestAction:
            farestAction.remove(self.homeDirection)
            return random.choice(farestAction)
          else:
            return random.choice(farestAction)
          
    def isDeadend(self, gameState, action):
      
      successorState = gameState.generateSuccessor(self.index, action)
      actions = successorState.getLegalActions(self.index)
      if len(actions) = 1:
        return True
      return false

    def isCorner(self, gameState, action):
      successorState = gameState.generateSuccessor(self.index, action)
      actions = successorState.getLegalActions(self.index)
      if len(actions) = 2:
        return True
      return false







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


