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

#  Group:  Tantan
#  Author: Bowen Rao   raob@student.unimelb.edu.au
#          Zizhe Ruan  zizher@student.unimelb.edu.au
#          Tenglun Tan tenglunt@student.unimelb.edu.au

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import copy
from game import Actions

#################
# Team creation #
#################



def createTeam(firstIndex, secondIndex, isRed,
               first = 'concreteAgent', second = 'concreteAgent'):
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

class AdversarialAgent(CaptureAgent):
  """
   A base class for  agents that chooses score-maximizing actions
   """

  def registerInitialState(self, gameState):
      util.raiseNotDefined()

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    util.raiseNotDefined()

  def calculateCloseGhost(self, gameState, threshold):
      opponentList = self.getOpponents(gameState)
      closeGhost = []
      for opponent in opponentList:
          if gameState.getAgentState(opponent).isPacman or gameState.getAgentState(opponent).scaredTimer > 0:
              continue
          if gameState.getAgentPosition(opponent) != None:
              ghostPos = gameState.getAgentState(opponent).getPosition()
              currentPos = gameState.getAgentState(self.index).getPosition()

              if self.getMazeDistance(ghostPos, currentPos) <= threshold:
                  closeGhost.append(opponent)
      return len(closeGhost)

  def calculateClosePacmanAsDefender(self, gameState, threshold):
      opponentList = self.getOpponents(gameState)
      closePacmam = []
      for opponent in opponentList:
          if not gameState.getAgentState(opponent).isPacman or gameState.getAgentState(opponent).scaredTimer > 0:
              continue
          if gameState.getAgentPosition(opponent) != None:
              pacmanPos = gameState.getAgentState(opponent).getPosition()
              currentPos = gameState.getAgentState(self.index).getPosition()

              if self.getMazeDistance(pacmanPos, currentPos) <= threshold:
                  closePacmam.append(opponent)
      return len(closePacmam)

  def getSuccessor(self, gameState, action, index):
    successor = gameState.generateSuccessor(index, action)
    postion = successor.getAgentState(index).getPosition()
    if postion != util.nearestPoint(postion):

        return successor.generateSuccessor(index, action)
    else:
        return successor

  def evaluate(self, gameState, action):

    util.raiseNotDefined()


class concreteAgent(AdversarialAgent):
    def registerInitialState(self, gameState):
        # TODO swicth between attack and defence rely on the food number of the layout

        self.foodSize = len(self.getFood(gameState).asList())

        self.start = gameState.getAgentPosition(self.index)
        self.searchDepth = 3
        self.bagCapacity = 14
        CaptureAgent.registerInitialState(self, gameState)
        self.estimateEnemyPos = {}
        self.halflayoutHeight = gameState.data.layout.height/2
        self.isAttacking = True
        self.walls = gameState.getWalls().asList()
        self.defenceToAttack = False


        for enemy in self.getOpponents(gameState):
            self.estimateEnemyPos[enemy] = inferringGoalProbability(gameState, self.index, enemy)

    def chooseAction(self, gameState):
        for opponent in self.getOpponents(gameState):
            self.estimateEnemyPos[opponent].observe(gameState)

        foodLeft = len(self.getFood(gameState).asList())

        ourfoodLeft = self.getFoodYouAreDefending(gameState).asList()


        # if (self.getScore(gameState)<3 or self.defenceToAttack)  :
        # print self.getScore(gameState)

        # if (len(ourfoodLeft)> 16 or self.getScore(gameState)==0) or (self.index < 2 and len(ourfoodLeft) > 10):

        closeGhostAsGhost = self.calculateCloseGhost(gameState,2)

        # if ((len(ourfoodLeft) > 16 or self.getScore(gameState) == 0) or (self.index < 2 and len(ourfoodLeft) > 10)) \
        #         and not (not gameState.getAgentState(self.index).isPacman and closeGhostAsGhost>0) :

        if ((len(ourfoodLeft) > 5 or self.getScore(gameState) == 0) or (self.index < 2 and len(ourfoodLeft) > 3)) \
                and not (not gameState.getAgentState(self.index).isPacman and closeGhostAsGhost > 0):

            self.isAttacking = True
        else:
            self.isAttacking = False

        if self.isAttacking:

            # TODO: while attacking, as a ghost,  if enemy near us at the border, stop attacking avoid be eaten on the border

            closeGhost = self.calculateCloseGhost(gameState, 4)

            # "agent is pacman and ghost closing, return to own land"

            # if gameState.getAgentState(self.index).isPacman and (foodLeft <= 2 or \
            #                 gameState.getAgentState(self.index).numCarrying > self.bagCapacity or closeGhost>0):
            if gameState.getAgentState(self.index).isPacman and (foodLeft <= 2 or closeGhost > 0):
                return self.searchEscapePath([self.start])

            # "on my way to attack but enemy pacman closing in, defend"

            if not gameState.getAgentState(self.index).isPacman:
                closeGhostAsGhost = self.calculateClosePacmanAsDefender(gameState,5)
                if closeGhostAsGhost>0:
                    self.isAttacking = False

                    actions = gameState.getLegalActions(self.index)
                    values = [self.evaluate(gameState, a) for a in actions]
                    maxValue = max(values)
                    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

                     # "change tp defence mode"

                    return random.choice(bestActions)



            actions = gameState.getLegalActions(self.index)
            values = [self.evaluate(gameState, a) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]

            return random.choice(bestActions)


        else:
            self.defenceToAttack = False

            actions = gameState.getLegalActions(self.index)
            values = [self.evaluate(gameState, a) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            return random.choice(bestActions)


    def evaluate(self, gameState, action):


        "evaluate when in attack mode"

        if self.isAttacking:
            successor = self.getSuccessor(gameState, action, self.index)
            evaluatePoint = self.alphabetaSearch(0, successor)


        else:
            # "evaluate when in defence mode"
            features = self.getFeaturesAsDefender(gameState, action)
            weights = self.getWeightsAsDefender(gameState, action)
            evaluatePoint = features * weights

        return evaluatePoint

    def searchEscapePath(self, problem):
        #  Using Astar algorithm to search a escaping path when encounter enemy ghost as Pacman
        for sub_problem in problem:
            gameState = self.getCurrentObservation()
            currentPos = gameState.getAgentPosition(self.index)
            openList = util.PriorityQueue()
            initialCost = util.manhattanDistance(currentPos,sub_problem)
            openList.push((gameState.getAgentPosition(self.index),[]),initialCost)
            closedList = set()
            while not openList.isEmpty():
                currentNode, currentMove=openList.pop()
                if not currentNode in closedList:
                    if currentMove!=[] and currentNode==sub_problem:
                        return currentMove[0]
                    closedList.add(currentNode);

                    for child in self.getSuccessors(currentNode,self.getCurrentObservation()):
                        updatedCost = util.manhattanDistance(child[0], sub_problem)+len(currentMove+[child[0]])

                        openList.push((child[0], currentMove+[child[1]]), updatedCost)



        #cannot find way then go home
        if not problem:
            # self.searchEscapePath([self.start])
            return 'Stop'
        else:
            if problem[0] != self.start:
                self.searchEscapePath([self.start])

        # if goals[0] != self.start:
        #     self.aStarSearch([self.start])
        #cannot find way go home then wait for die
        return 'Stop'

    def getSuccessors(self, curPos, gameState):
        successors = []

        wallList = copy.copy(self.walls)

        if gameState.getAgentState(self.index).isPacman:
            # get defenders position
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            enemyGhost = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]
            if len(enemyGhost) > 0:
                defendersPos = [i.getPosition() for i in enemyGhost]
                wallList.extend(defendersPos)

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = curPos
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if (nextx, nexty) not in wallList:
                nextState = (nextx, nexty)
                successors.append((nextState, action))
        return successors

    def alphabetaSearch(self,depth,gameState):
        return self.min_value(gameState,-999999,999999,depth)

    def max_value(self,gameState, alpha,beta,depth):
        if depth == self.searchDepth:
            return self.evaluateCurrentSuccessor(gameState)
        v = -999999
        actions = gameState.getLegalActions(self.index)

        for action in actions:
            successor = self.getSuccessor(gameState, action, self.index)
            v = max(v,self.min_value(successor,alpha,beta,depth+1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v




    def min_value(self,gameState,alpha,beta,depth):
        if depth ==self.searchDepth:
            return self.evaluateCurrentSuccessor(gameState);
        v = 999999
        opponentList = self.getOpponents(gameState)
        opponent1 = opponentList[0]
        opponent2 = opponentList[1]
        opponentCount = 0
        for opponent in opponentList:
            if gameState.getAgentPosition(opponent):
                opponentCount = opponentCount + 1

        if opponentCount == 2 :
            # print opponentCount
            # for opponent in opponentList:
            #     print opponent,"!!!"

            for action in gameState.getLegalActions(opponentList[opponentCount-1]):
                if opponentCount == 0 :
                    successor = self.getSuccessor(gameState,action,opponentList[opponentCount-1])
                    # successor = gameState.generateSuccessor(opponentList[opponentCount-1],action)
                    v = min(v,self.max_value(successor,alpha,beta,depth + 1))
                else:
                    opponentCount = opponentCount-2;
                    # successor = gameState.generateSuccessor(opponentList[opponentCount - 1], action)
                    successor = self.getSuccessor(gameState, action,opponentList[opponentCount - 1])
                    v = min(v,self.max_value(successor,alpha,beta,depth+1))

                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        if gameState.getAgentPosition(opponent1) != None:
            actions = gameState.getLegalActions(opponent1)
            for action in actions:
                successor = self.getSuccessor(
                    gameState, action, opponentList[0])
                v = min(v, self.max_value(successor, alpha, beta, depth + 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        if gameState.getAgentPosition(opponent2) != None:
            actions = gameState.getLegalActions(opponent2)
            for action in actions:
                successor = self.getSuccessor(
                    gameState, action, opponentList[1])
                v = min(v, self.max_value(successor, alpha, beta, depth + 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        v = min(v, self.max_value(gameState, alpha, beta, depth + 1))

        if v <= alpha:
            return v
        beta = min(beta, v)

        return v


    def evaluateCurrentSuccessor(self,gameState):
        foodList = self.getFood(gameState).asList()

        capsules = self.getCapsules(gameState)
        if capsules is not None:
            for e in capsules:
                foodList.append(e)


        foodList1 = []
        foodList2 = []
        for food in foodList:
            if food[1] <=self.halflayoutHeight:
                foodList1.append(food)
            else:
                foodList2.append(food)


        gameScore = -len(foodList)
        minDistance = None
        ghostScore = None



        # "two food list each for each agent to attack"
        if len(foodList) > 0:
            myPos = gameState.getAgentState(self.index).getPosition()
            if self.index < 2:
                if len(foodList1) > 0:
                    minDistance = min([self.getMazeDistance(
                        myPos, food) for food in foodList1])
                else:
                    minDistance = min([self.getMazeDistance(
                        myPos, food) for food in foodList])
            else:
                if len(foodList2) > 0:
                    minDistance = min([self.getMazeDistance(
                        myPos, food) for food in foodList2])
                else:
                    minDistance = min([self.getMazeDistance(
                        myPos, food) for food in foodList])

        ghostScore = self.calculateCloseGhost(gameState,4)

        if ghostScore>0:
            ghostScore = 1
        else:
            ghostScore = 0

        if minDistance is not None:
            return 100*gameScore+(-1*minDistance)+(-10000*ghostScore)
        else:
            return 100 * gameScore  + (-10000 * ghostScore)


# -----------------defence mode--------------------------

    def getFeaturesAsDefender(self, gameState, action):
        features = util.Counter()

        successor = self.getSuccessor(gameState, action,self.index)

        nextState = gameState.generateSuccessor(self.index, action)
        nextStatePos = nextState.getAgentPosition(self.index)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()


        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemyPacman = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(enemyPacman)

        # "here use the inference module to get the estimate oppoent distance"

        likelyPos = {}
        for enemy in self.getOpponents(gameState):
            likelyPos[enemy] = self.estimateEnemyPos[enemy].belief.argMax()

        minDistanceToPacman = 9999999
        minDistanceToGhost  = 9999999


        for enemy in self.getOpponents(gameState):
            distanceToEnemy = self.getMazeDistance(myPos,likelyPos[enemy])
            if successor.getAgentState(enemy).isPacman:
                minDistanceToPacman = min (distanceToEnemy,minDistanceToPacman)
            else:
                minDistanceToGhost = min (distanceToEnemy,minDistanceToGhost)

        if minDistanceToPacman != 9999999:
            features['minDistanceToPacman'] = minDistanceToPacman
        else:  # all ghost
            features['minDistanceToPacman'] = minDistanceToGhost


        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeightsAsDefender(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'minDistanceToPacman': -10, 'stop': -100,'reverse': -2}


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

    '''
    Your initialization code goes here, if you need any.
    '''


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
          dist = self.getMazeDistance(self.start, pos2)
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
      if pos != util.nearestPoint(pos):
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



class inferringGoalProbability:
    def __init__(self,gameState,myIndex,enemyIndex):
        self.index = myIndex
        self.enemyIndex = enemyIndex
        self.belief = util.Counter()

        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                if gameState.hasWall(x,y):
                    self.belief[(x,y)] = 0
                else:
                    self.belief[(x,y)] = 1

        self.belief.normalize()


    def observe(self,gameState):
        enemyPos = gameState.getAgentPosition(self.enemyIndex)
        myPos = gameState.getAgentPosition(self.index)
        noiseDistance = gameState.getAgentDistances()[self.enemyIndex]

        # noiseDistance[self.enemyIndex] = gameState.getAgentDistances(self.enemyIndex)

        if enemyPos!=None:
            for pos in self.belief:
                self.belief[pos] = 0
            self.belief[enemyPos] = 1
        else:
            for pos in self.belief:
                distance = util.manhattanDistance(myPos,pos)

                "get the probability of a noisy distance given the true distance"
                self.belief[pos] = self.belief[pos]*gameState.getDistanceProb(distance,noiseDistance)
        self.belief.normalize()

        if self.belief.totalCount() ==0:
            for x in range(gameState.data.layout.width):
                for y in range(gameState.data.layout.height):
                    if gameState.hasWall(x, y):
                        self.belief[(x, y)] = 0
                    else:
                        self.belief[(x, y)] = 1

            self.belief.normalize()








