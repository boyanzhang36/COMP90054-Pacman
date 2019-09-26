# myTeam.py
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
from game import Directions

import random, time, util, sys
from util import nearestPoint
from game import Actions
import pickle

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OurAgent', second='OurAgent'): #OffensiveAgent
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

    # The following line is an example only; feel free to change it.
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
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)

        bestActions = [a for a, v in zip(actions, values) if v == maxValue] # according to the maxValue choose the best action(s)

        foodLeft = len(self.getFood(gameState).asList()) # get the number of food we are going to eat

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)

                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions) # randomly choose one of the best actions

    def getSuccessor(self, gameState, action): # return successor which can used to choose action
        """
    Finds the next successor which is a grid position (location tuple).
    """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action): # get the value of the state (f*w)
        """
    Computes a linear combination of features and feature weights
    """
        features = self.getFeatures(gameState, action)
        print('actions:',action)
        print('features:',features)
        weights = self.getWeights(gameState, action)
        print("weights",weights)
        print("value",features * weights)

        return features * weights

    def getFeatures(self, gameState, action):
        """
    Returns a counter of features for the state
    """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)  # feature == score
        return features

    def getWeights(self, gameState, action):
        """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
        return {'successorScore': 1.0}


class OurAgent(ReflexCaptureAgent):

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        myState = gameState.getAgentState(self.index)
        actions = gameState.getLegalActions(self.index)
        myPos = myState.getPosition()


        start = time.time()
        Enemies = self.whoChaseMeNearby(self,gameState)
        Invaders = self.whoEatMyFoodNearby(self,gameState)
        teamMatePos = self.getTeamMatePosition(self, gameState)
        foodGone = self.whereAreLostFoods(self)


        #foodList = self.getFoodYouAreDefending(gameState).asList()

        values = [self.evaluate(gameState, a) for a in actions]

        maxValue = max(values)

        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        # print(values,bestActions,maxValue)

        # for a, v in zip(actions, values):
        #     print(a, v)
        #     print("----")
        # print("bestactions")
        # print(bestActions)
        foodLeft = len(self.getFood(gameState).asList())
        score = gameState.getScore()
        # if(self.red and score>0)
        if foodLeft <= 2 or ((gameState.data.timeleft) / 4 <= 22):
            if myState.isPacman:
                bestDist = 9999
                for action in actions:
                    successor = self.getSuccessor(gameState, action)
                    pos2 = successor.getAgentPosition(self.index)
                    dist = self.getMazeDistance(self.start, pos2)
                    if dist < bestDist:
                        bestAction = action
                        bestDist = dist
                return bestAction
            if len(self.whereLost) > 0:
                self.actionsGoLost = self.aaStar(gameState, self.whereLost[0])
                if len(self.actionsGoLost) > 0:
                    return self.actionsGoLost[0]
                if myPos == self.whereLost[0]:
                    self.whereLost = []
            else:
                return random.choice(gameState.getLegalActions(self.index))

        print(self.index)
        print("bestactions")
        print(bestActions)
        print('----')
        # if bestAction =="Stop"
        return random.choice(bestActions)

    def getTeamMatePosition(self,gameState):
        teamIndex = self.getTeam(gameState)
        if (self.index == teamIndex[0]):
            teamMatePos=gameState.getAgentState(teamIndex[1]).getPosition()
        else:
            teamMatePos = gameState.getAgentState(teamIndex[0]).getPosition()
        return teamMatePos
    
    def whoEatMyFoodNearby(self, gameState):
        
        opponentStates = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        pacmans = [a for a in opponentStates if a.isPacman and a.getPosition() != None] # check postion to find the closest
        myPos = gameState.getAgentPosition(self.index)
        answer = [(a, self.getMazeDistance(myPos,a.getPosition())) for a in pacmans] # e.g. [(1,6),(2,12)]
        answer.sort() # smallest first
        return answer
     

    def whoChaseMeNearby(self, gameState):
        #if gameState.getAgentState(self.index).isPacman:  # I am eating others' food
            opponentStates = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            whoChaseMe = [a for a in opponentStates if not a.isPacman and a.getPosition() != None]

            myPos = gameState.getAgentPosition(self.index)
            answer = [(a, self.getMazeDistance(myPos,a.getPosition())) for a in whoChaseMe] # e.g. [(1,6),(2,12)]
            answer.sort() # smallest first
            return answer
        #else:
            #return None
    
    
    def whereAreLostFoods(self):
        lastState = self.getPreviousObservation()
        currentState = self.getCurrentObservation()
        oldFoods = self.getFoodYouAreDefending(lastState).aslist()
        newFoods = self.getFoodYouAreDefending(currentState).aslist()
        foodGone = list(set(oldFoods).difference(set(newFoods)))
        return foodGone # if no missing then return None

    def aaStar(self, gameState, whereLostFood):
        from util import PriorityQueue
        from util import manhattanDistance

        closed = []
        directions = {}
        costs = {}

        priorityQ = PriorityQueue()
        priorityQ.push(gameState, 0)

        startState = gameState.getAgentState(self.index)
        startPosition = startState.getPosition()
        closed.append(startPosition)

        directions[startPosition] = []
        costs[startPosition] = 0

        while not priorityQ.isEmpty():
            cur_State = priorityQ.pop()
            cur_Position = cur_State.getAgentState(self.index).getPosition()
            actions = cur_State.getLegalActions(self.index)

            if cur_Position == whereLostFood:
                print(directions[cur_Position])
                return directions[cur_Position]

            for action in actions:
                nextState = self.getSuccessor(cur_State, action)
                nextPosition = nextState.getAgentState(self.index).getPosition()
                Actions = directions[cur_Position][:]
                nextCost = costs[cur_Position] + 1

                if nextPosition not in closed or nextCost < costs[nextPosition]:
                    costs[nextPosition] = nextCost
                    directions[nextPosition] = []
                    Actions.append(action)
                    directions[nextPosition].extend(Actions)
                    closed.append(nextPosition)
                    g = nextCost
                    h = manhattanDistance(nextPosition, whereLostFood)
                    f = g + h
                    priorityQ.push(nextState, f)

        return []
                

        
        


