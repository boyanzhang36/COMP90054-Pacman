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
               first='PrimaryAgent', second='SecondaryAgent'):

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


class DummyAgent(CaptureAgent):
    """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

    def registerInitialState(self, gameState):
        """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

        '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
    Picks among actions randomly.
    """
        actions = gameState.getLegalActions(self.index)

        '''
    You should change this in your own agent.
    '''

        return random.choice(actions)


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
        start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]

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


        return random.choice(bestAnctions) # randomly choose one of the best actions

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
        print("***************After changes from {} is {}".format(self.index, action))
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


class BaseAgent(ReflexCaptureAgent):

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        self.weights = util.Counter()
        self.weights['foodNum'] = -100
        self.weights['distanceToFood'] = -1
        self.weights['invaderDistance'] = -30
        self.weights['numInvaders'] = -1000
        self.weights['distanceToCap'] = -10
        self.weights['capNum'] = -120
        self.weights['oppScared'] = -1

        self.originalFood = self.getFoodYouAreDefending(gameState).asList()
        self.whereLostFood = []

    def chooseAction(self, gameState):
        """
    Picks among the actions with the highest value.
    """
        myState = gameState.getAgentState(self.index)
        actions = gameState.getLegalActions(self.index)
        myPos = myState.getPosition()
        start = time.time()

        oppositeGhost = self.getOppositeGhost(gameState)

        invadingPacman,_,_ = self.getInvadingPacman(gameState)


        self.whereLostFood = self.whereAreLostFoods(self.originalFood,gameState)

        teamIndex = self.getTeam(gameState)
        teamMatePos = self.getTeamMatePosition(gameState)

        if self.index == teamIndex[0]:
            teamMateIndex = teamIndex[1]
            eneDist = 3
        else:
            teamMateIndex = teamIndex[0]
            eneDist = 2

        if len(oppositeGhost) > 0 and oppositeGhost[0].scaredTimer <= 1:
            if (not oppositeGhost[0].isPacman) and (not myState.isPacman) and self.getMazeDistance(
                    oppositeGhost[0].getPosition(),
                    myPos) <= eneDist:  #defending opponents become pacman
                actKeepGhost = []
                for a in actions:
                    succ = self.getSuccessor(gameState, a).getAgentState(self.index)
                    dis = self.getMazeDistance(myPos, succ.getPosition())
                    if not succ.isPacman and dis == 1:
                        actKeepGhost.append(a)

                bestDist = 0
                best_action = []
                for action in actKeepGhost:
                    successor = self.getSuccessor(gameState, action)
                    pos2 = successor.getAgentPosition(self.index)  # get my successor's position
                    dist = self.getMazeDistance(successor.getAgentState(teamMateIndex).getPosition(),
                                                pos2)  # get my teammate's successor's position ,then calculate thedistance between them
                    if dist >= bestDist:
                        if dist == bestDist:  
                            best_action.append(action)
                            bestDist = dist
                        else:
                            best_action = [action]
                            bestDist = dist
                return random.choice(best_action)

        if len(oppositeGhost) > 0 and oppositeGhost[0].scaredTimer <= 0:
            answer = self.actionOnDuty(gameState, oppositeGhost, myPos, myState, actions)
            if answer is not None:
                return answer

        if len(invadingPacman) > 0: # the oppenent team has one or two invading pacman
            distToPacman = self.getMazeDistance(myPos, invadingPacman[0].getPosition()) # invader[0]: represents the nearest pacman, the distance between me and the nearest pacman
            allyToPacman = self.getMazeDistance(teamMatePos, invadingPacman[0].getPosition())
            if distToPacman < 15 and (not myState.isPacman) and (distToPacman <= allyToPacman): # my current state is ghost: defending our food
                if myState.scaredTimer == 0: # i am not scared, so i can eat the invading pacman
                    return self.findClosestAction(gameState, invadingPacman[0].getPosition())
                if myState.scaredTimer != 0:
                    if self.getMazeDistance(myPos, invadingPacman[0].getPosition()) >= 3: # i am scared, but i am far away from the invading pacman
                        return self.findClosestAction(gameState, invadingPacman[0].getPosition())
                    else:  # i am scared and i am near the invading pacman
                        # return self.findFarthestAction(gameState,invadingPacman[0].getPosition())
                        bestDist = 0
                        for action in actions:
                            successor = self.getSuccessor(gameState, action)
                            pos2 = successor.getAgentPosition(self.index)
                            dist = self.getMazeDistance(invadingPacman[0].getPosition(), pos2)
                            if self.getMazeDistance(pos2, myPos) > 1:  # i dont need to take action, because i will not be eaten ,bu lang fei bushu
                                continue
                            if dist > bestDist: #need to take a action
                                bestAction = action
                        return bestAction


        values = [self.evaluate(gameState, a) for a in actions ]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2 or ((gameState.data.timeleft)/4<=22):
            if myState.isPacman:
                #return self.findClosestAction(gameState, self.start)
                bestDist = 9999
                for action in actions:
                    successor = self.getSuccessor(gameState, action)
                    pos2 = successor.getAgentPosition(self.index)
                    dist = self.getMazeDistance(self.start, pos2)
                    if dist < bestDist:
                        bestAction = action
                        bestDist = dist
                return bestAction
            if len(self.whereLostFood) > 0:
                self.actionsGoLost = self.aaStar(gameState, self.whereLostFood[0]) # find the eaten food , goal recognition , find the invading pacman
                if len(self.actionsGoLost) > 0:
                    return self.actionsGoLost[0]
                if myPos == self.whereLostFood[0]:
                    self.whereLostFood = []
            else:
                return random.choice(gameState.getLegalActions(self.index))
        
        answer = random.choice(bestActions)
        print("==========================================ACTION TAKEN by {} is {} at {}".format(self.index, answer, myPos))
        return answer

    def actionOnDuty(self,gameState, oppositeGhost, myPos, myState, actions):
        return Directions.STOP

    def findClosestAction(self, gameState, pos):

        actions = gameState.getLegalActions(self.index)
        dists = [9999]
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            dist = self.getMazeDistance(successor.getAgentPosition(self.index), pos)
            dists.append(dist)

        closestActions = [action for action, dist in zip(actions,dists) if dist == min(dists)]
        if closestActions is None or len(closestActions) ==0:
            return Directions.STOP
        else:
            return random.choice(closestActions)

    def findFarthestAction(self, gameState, pos):

        actions = gameState.getLegalActions(self.index)
        dists = [0]

        for action in actions:
            successor = self.getSuccessor(gameState, action)
            dist = self.getMazeDistance(successor.getAgentPosition(self.index), pos)
            dists.append(dist)

        farthestActions = [action for action, dist in zip(actions,dists) if dist == max(dists)]
        if farthestActions is None or len(farthestActions)==0:
            return Directions.STOP
        else:
            return random.choice(farthestActions)

    def manhattanDistance(self, p1, p2):  # calculate manhattan distance of two given points
        "The Manhattan distance heuristic for a PositionSearchProblem"
        xy1 = list(p1)
        xy2 = list(p2)
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

    def aaStar(self, gameState, whereLostFood):
        from util import PriorityQueue
        from util import manhattanDistance

        closed = []
        directions = {}
        costs = {}

        priorityQ = PriorityQueue()
        priorityQ.push(gameState, 0)

        startState = gameState.getAgentState(self.index)
        startPosition= startState.getPosition()
        closed.append(startPosition)

        directions[startPosition] = []
        costs[startPosition] = 0

        while not priorityQ.isEmpty():
            cur_State = priorityQ.pop()
            cur_Position = cur_State.getAgentState(self.index).getPosition()
            actions = cur_State.getLegalActions(self.index)

            if cur_Position == whereLostFood:
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

    def getTeamMatePosition(self,gameState):
        teamIndex = self.getTeam(gameState)
        if (self.index == teamIndex[0]):
            teamMatePos=gameState.getAgentState(teamIndex[1]).getPosition()
        else:
            teamMatePos = gameState.getAgentState(teamIndex[0]).getPosition()
        return teamMatePos

    def getInvadingPacman(self, gameState):

        opponentStates = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        pacmans = [a for a in opponentStates if
                   a.isPacman and a.getPosition() != None]  # check postion to find the closest
        myPos = gameState.getAgentPosition(self.index)
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in pacmans]  # e.g. [(1,6),(2,12)]
        minDist = 0
        if len(pacmans) > 0:
            minDist = min(dists)
            invadingPacman = [a for a in pacmans if self.getMazeDistance(myPos, a.getPosition()) == min(dists)]
            return invadingPacman, minDist, pacmans
        else:
            return [], minDist, pacmans
      
    def getOppositeGhost(self, gameState):
        # if gameState.getAgentState(self.index).isPacman:  # I am eating others' food
        opponentStates = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        whoChaseMe = [a for a in opponentStates if not a.isPacman and a.getPosition() != None]

        myPos = gameState.getAgentPosition(self.index)
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in whoChaseMe]  # e.g. [(1,6),(2,12)]
        oppositeGhost = [a for a in whoChaseMe if self.getMazeDistance(myPos, a.getPosition()) == min(dists)]  # smallest first
        return oppositeGhost

    def whereAreLostFoods(self,originalFood, gameState):
        foodList = self.getFoodYouAreDefending(gameState).asList()
        foodGone = list(set(self.originalFood).difference(set(foodList)))
        return foodGone # if no missing then return None

    def getFeatures(self, gameState, action):

        # game state after taking the next action
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = successor.getAgentPosition(self.index)
        teamIndex=self.getTeam(successor)
        

        features = util.Counter()

        # food number feature
        foodList = self.getFood(successor).asList()
        features['foodNum'] = len(foodList) 
        # food distance feature
        if len(foodList) > 0:
            # get closest food position
            dists_food = [(self.getMazeDistance(myPos, food),food) for food in foodList]
            dists_food.sort()
            minDist2Food = dists_food[0][0]
            closestFood = dists_food[0][1]

            if not myState.isPacman:

                if minDist2Food < 10:
                    features['distanceToFood'] = minDist2Food
                else:
                    from operator import itemgetter
                    foodList.sort(key=itemgetter(1))
                    minY = foodList[0][1]
                    maxY = foodList[-1][1]
                    mostNorthFoods = [food for food in foodList if food[1] == maxY] # find the northmost food
                    mostSouthFoods = [food for food in foodList if food[1] == minY] # find the southmost food

                    #find the closest in the most northern/southern foods
                    distsNorth = [(self.getMazeDistance(myPos, food), food) for food in mostNorthFoods]
                    distsNorth.sort()
                    minDist2mostNorth = distsNorth[0][0]

                    distsSouth = [(self.getMazeDistance(myPos, food), food) for food in mostSouthFoods]
                    distsSouth.sort()
                    minDist2mostSouth = distsSouth[0][0]

                    if self.index == teamIndex[0]:
                        features['distanceToFood'] = minDist2mostSouth
                    else:
                        features['distanceToFood'] = minDist2mostNorth

            else:
                teamMatePos = self.getTeamMatePosition(successor)
                closestFood2TeamMate = self.getMazeDistance(teamMatePos, closestFood)

                # get the second closest distance
                dists = [self.getMazeDistance(myPos, food) for food in foodList]
                dists.sort()
                distsSet = set(dists)

                distsSet.pop()
                if len(distsSet) >0:
                    secondMinDist2Food = distsSet.pop()
                else:
                    secondMinDist2Food = minDist2Food

                # compare who is closer to the cloestd
                if minDist2Food < closestFood2TeamMate:
                    features['distanceToFood'] = minDist2Food
                else:
                    features['distanceToFood'] = secondMinDist2Food
        else:
            features['distanceToFood'] = 9999




         # invader number and invader instance feature
        if not myState.isPacman:
            _, minDist,invaders = self.getInvadingPacman(gameState)
            features['invaderDistance'] = minDist
            features['numInvaders'] = len(invaders)


        # capsule features
        capsulePos = self.getCapsules(successor)
        features['capNum'] = len(capsulePos)

        # get nearest capsules
        distsCap = [self.getMazeDistance(myPos, a) for a in capsulePos]
        nearestCap = [cap for cap in capsulePos if self.getMazeDistance(myPos, cap) == min(distsCap)]

        if len(nearestCap) > 0 and self.getMazeDistance(myPos, nearestCap[0]) <= 4:
            features['distanceToCap'] = 9999*(len(capsulePos)-1) + self.getMazeDistance(myPos, nearestCap[0])
        else:
            features['distanceToCap'] = 9999*len(capsulePos)



        # opponent scared feature
        opponentStates = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        if opponentStates[0].scaredTimer > 0: # if they scared
            opponentStates = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            scaredVisible = [a for a in opponentStates if a.getPosition() != None]  # check postion to find the closest
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in scaredVisible]  # e.g. [(1,6),(2,12)]
            if len(dists) > 0:
                features['oppScared'] = min(dists)
    

        return features


    def getWeights(self,gameState,action):
        return self.weights

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"

        if state.getAgentState(self.index).isPacman:
            return False
        else:
            return True

    def heuristicOfGreedy(self, state, enemyPos):

        myPos = state.getAgentState(self.index).getPosition()

        return -2*self.getMazeDistance(myPos, enemyPos)+self.getMazeDistance(myPos, self.start)

    def greedyBestFirstSearch(self, gameState, enemyPosition):
        """Search the node that has the lowest combined cost and heuristic first."""
        from util import PriorityQueue
        closed, directions = [],[]
        myState = gameState.getAgentState(self.index)
        myPosition = myState.getPosition()
        cost = 0
        preCost = 0
        priorityQ = PriorityQueue()
        priorityQ.push((gameState, directions, cost, preCost, myPosition), cost)
        while not priorityQ.isEmpty():
            cur_state = priorityQ.pop()
            if self.isGoalState(cur_state[0]):
                if (len(cur_state[1]) == 0):
                    return 'random'
                else:
                    return cur_state[1]

            if cur_state[4] in closed:
                continue
            else:
                closed.append(cur_state[4])
                actions = cur_state[0].getLegalActions(self.index)
                for action in actions:
                    my_curState = cur_state[0].getAgentState(self.index)
                    my_curPos = my_curState.getPosition()
                    my_nextState = self.getSuccessor(cur_state[0], action).getAgentState(self.index)
                    my_nextPos = my_nextState.getPosition()

                    nextDirection = list(cur_state[1])
                    nextDirection.append(action)
                    nextCost = self.heuristicOfGreedy(cur_state[0], enemyPosition)

                    if self.getMazeDistance(my_nextPos, my_curPos) == 1:
                        if self.getMazeDistance(enemyPosition, my_nextPos) <= 1 and len(list(nextDirection)) == 1:
                            continue
                        elif len(cur_state[1]) < 15:
                            preCost = len(cur_state[1]) * 1.8
                            priorityQ.push((self.getSuccessor(cur_state[0], action), nextDirection, nextCost + preCost, preCost, my_nextPos),nextCost + preCost)
                        else:
                            preCost = len(cur_state[1]) * len(cur_state[1]) / 5
                            priorityQ.push((self.getSuccessor(cur_state[0], action), nextDirection, nextCost + preCost, preCost, my_nextPos),nextCost + preCost)
        return directions





class PrimaryAgent(BaseAgent):

    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    def actionOnDuty(self, gameState, oppositeGhost,myPos,myState, actions):
        """
    Picks among the actions with the highest value.
    """

        if self.getMazeDistance(oppositeGhost[0].getPosition(), myPos) <= 2:
            actToGhost = self.greedyBestFirstSearch(gameState, oppositeGhost[0].getPosition())
            if (len(actToGhost) > 0):
                if(actToGhost=='random' and self.getMazeDistance(oppositeGhost[0].getPosition(), myPos) <= 4):
                    actions = gameState.getLegalActions(self.index)
                    act= [a for a in actions if a != 'West' and a!= 'East' ]
                    return random.choice(act)
                else:
                    return actToGhost[0]
            else:
                return self.findFarthestAction(gameState, oppositeGhost[0].getPosition())

    def getWeights(self, gameState, action):

        weights = BaseAgent.getWeights(self, gameState, action)

        weights['distanceToCap'] = -10000
        
                       
        return weights

class SecondaryAgent(BaseAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    def actionOnDuty(self,gameState, oppositeGhost, myPos,myState, actions):
        if len(oppositeGhost) > 0 and self.getMazeDistance(oppositeGhost[0].getPosition(), myPos) <= 2: # i am the pacman
            foodLeft = len(self.getFood(gameState).asList())
            actToGhost = self.greedyBestFirstSearch(gameState, oppositeGhost[0].getPosition())
            if foodLeft <= 2: # need to avoid the ghost
                if (len(actToGhost) > 0):
                    if (actToGhost == 'random' and self.getMazeDistance(oppositeGhost[0].getPosition(), myPos) <= 4):

                        actions = gameState.getLegalActions(self.index)
                        act = [a for a in actions if a != 'West' and a != 'East']
                        return random.choice(act)
                    else:
                        return actToGhost[0]
                else:
                    return self.findFarthestAction(gameState,oppositeGhost[0].getPosition())
            if actToGhost!=[]:
                legalActs=[]
                for acts in gameState.getLegalActions(self.index):
                    myCurrentState = gameState.getAgentState(self.index)
                    myNextState = self.getSuccessor(gameState, acts).getAgentState(self.index)  # my successor's state
                    myCurrentPos = myCurrentState.getPosition()
                    myNextPos = myNextState.getPosition() # my successor's position
                    if self.getMazeDistance(myNextPos, myCurrentPos) == 1:
                        if self.getMazeDistance(oppositeGhost[0].getPosition(), myNextPos) > 1:
                            enemyPos= myCurrentPos
                            eneToSuccessor=self.greedyBestFirstSearch(self.getSuccessor(gameState, acts), enemyPos)
                            if len(eneToSuccessor) > 0:
                                legalActs.append(acts)
                for act in legalActs:
                    myNextState = self.getSuccessor(gameState, act).getAgentState(self.index)
                    nextFood = self.getFood(self.getSuccessor(gameState, act)).asList()
                    originFood = self.getFood(gameState).asList()
                    nextCaps =self.getCapsules(self.getSuccessor(gameState, act))
                    originCaps = self.getCapsules(gameState)

                    if  myState.numCarrying > 0 and not myNextState.isPacman: #after eating i change to the ghost
                        return act

                    if len(originCaps)-len(nextCaps )==1: # eat capsule
                        return act

                    if len(originFood)-len(nextFood)==1: # eat food
                        return act
                if (len(legalActs)!=0):
                    actionNow = []
                    if len(legalActs) >= 2:
                        actionNow = [a for a in legalActs if self.getSuccessor(gameState, a).getAgentState(self.index).isPacman]
                        return random.choice(actionNow)
                    return random.choice(legalActs)
                elif (len(legalActs)==0):
                    return self.findFarthestAction(gameState, oppositeGhost[0].getPosition())
            else:
                bestDist = 0
                bestAction="Stop"
                for action in actions:

                    successor = self.getSuccessor(gameState, action)
                    pos2 = successor.getAgentPosition(self.index)
                    if self.getMazeDistance(myPos, pos2) > 1:
                        continue
                    dist = self.getMazeDistance(oppositeGhost[0].getPosition(), pos2)

                    if dist > bestDist:
                        bestAction = action
                        bestDist = dist
                return bestAction


