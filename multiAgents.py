# multiAgents.py
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


from util import manhattanDistance
from game import Directions
import random, util
import math
from game import Agent


def randomPolicy(state):
    while not (state.isWin() or state.isLose):
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getScore()


class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isWin() or state.isLose()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def __str__(self):
        s = []
        s.append("totalReward: %s" % (self.totalReward))
        s.append("numVisits: %d" % (self.numVisits))
        s.append("isTerminal: %s" % (self.isTerminal))
        s.append("possibleActions: %s" % (self.children.keys()))
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))


class MCTAgent():
    def __init__(self, timeLimit=None, iterationLimit=500, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy
        # self.rollout = evaluationPolicy

    def search(self, initialState, needDetails=False):
        self.root = treeNode(initialState, None)  # 开始树只是个空树
        # 完成4步，构造出了树
        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        action = (action for action, node in self.root.children.items() if node is bestChild).__next__()
        self.action = action
        if needDetails:
            return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
        else:
            return action

    def getAction(self, gameState):
        action = self.search(gameState)
        return action

    def executeRound(self):
        """
            执行一次模拟
        """
        node = self.selectNode(self.root)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    # 这步就不抽象了，而是和具体的游戏有关
    def expand(self, node):
        actions = node.state.getLegalActions()  # 拓展节点
        for action in actions:
            if action not in node.children:
                # ??这句话我不知道对不对
                newNode = treeNode(node.state.getNextState(0, action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode
        raise Exception("Should never reach here")

    # 回溯，更改每个节点的值
    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    # 找到某一节点的最好子节点
    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)  # 多个最优中随机选择一个


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # 采集合法状态
        legalMoves = gameState.getLegalActions()

        # “一步推断”
        successorGameState = [gameState.getPacmanNextState(action) for action in legalMoves]
        scores = [EvaluationFunction(successorGameState) for successorGameState in successorGameState]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # 在所有最佳方向中随机选一个

        return legalMoves[chosenIndex]


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can
      if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.depth = int(depth)
        # 每次找两层


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        Here are some method calls that might be useful when implementing minimax.
        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
        gameState.getNextState(agentIndex, action):
        Returns the successor game state after an agent takes an action
        gameState.getNumAgents():
        Returns the total number of agents in the game
        gameState.isWin():
        Returns whether or not the game state is a winning state
        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimaxSearch(gameState, agentIndex=0, depth=self.depth)[1]
        # 在抽象类中定义默认深度为2，返回两个值，[0]是节点估计值，[1]是动作

    def minimaxSearch(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            # 递归最后一层：达到深度限制或游戏结束
            action = EvaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:  # Pacman
            action = self.maximizer(gameState, agentIndex, depth)
        else:  # Ghost
            action = self.minimizer(gameState, agentIndex, depth)
        return action

    def minimizer(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        # 得到合法的动作
        # 下一状态准备
        if agentIndex == gameState.getNumAgents() - 1:
            next_agent = 0
            next_depth = depth - 1
            # 若为最后一个鬼，则下一个index为pacman，深度-1
        else:
            next_agent = agentIndex + 1
            next_depth = depth
            # 若非最后一个鬼，则本层搜索未完，继续搜索下一个ghost
        min_score = float("inf")
        min_action = Directions.STOP
        # 初始化
        for action in actions:
            successor_game_state = gameState.getNextState(agentIndex, action)
            new_score = self.minimaxSearch(successor_game_state, next_agent, next_depth)[0]
            # 递归调用 [0]为评估函数
            if new_score < min_score:
                min_score = new_score
                min_action = action
        return min_score, min_action

    def maximizer(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        # 由于为pacman，不需要分类
        next_agent = agentIndex + 1
        next_depth = depth
        max_score = float("-inf")
        max_action = Directions.STOP
        print('_______________________')
        a = input()
        print('actions:', actions)
        for action in actions:
            successor_game_state = gameState.getNextState(agentIndex, action)
            new_score = self.minimaxSearch(successor_game_state, next_agent, next_depth)[0]
            print(action, new_score)
            if new_score > max_score:
                max_score = new_score
                max_action = action
        return max_score, max_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def maximizer(agent, depth, game_state, a, b):  # maximizer function
            v = float("-inf")
            for newState in game_state.getLegalActions(agent):
                v = max(v, alphabetaprune(1, depth, game_state.getNextState(agent, newState), a, b))
                if v > b:  # α-β剪枝算法
                    return v
                a = max(a, v)  # 更新α值
            return v

        def minimizer(agent, depth, game_state, a, b):  # minimizer function
            v = float("inf")

            next_agent = agent + 1  # 计算下一个Agent
            if game_state.getNumAgents() == next_agent:
                next_agent = 0
            if next_agent == 0:
                depth += 1

            for newState in game_state.getLegalActions(agent):
                v = min(v, alphabetaprune(next_agent, depth, game_state.getNextState(agent, newState), a, b))
                if v < a:  # α-β剪枝算法
                    return v
                b = min(b, v)  # 更新β值
            return v

        def alphabetaprune(agent, depth, game_state, a, b):
            if game_state.isLose() or game_state.isWin() or depth == self.depth:  # return the utility in case the defined depth is reached or the game is won/lost.
                return EvaluationFunction(game_state)

            if agent == 0:  # maximize for pacman
                return maximizer(agent, depth, game_state, a, b)
            else:  # minimize for ghosts
                return minimizer(agent, depth, game_state, a, b)

        """Performing maximizer function to the root node i.e. pacman using alpha-beta pruning."""
        utility = float("-inf")
        action = Directions.WEST
        alpha = float("-inf")
        beta = float("inf")
        for agentState in gameState.getLegalActions(0):
            ghostValue = alphabetaprune(1, 0, gameState.getNextState(0, agentState), alpha, beta)
            if ghostValue > utility:
                utility = ghostValue
                action = agentState
            alpha = max(alpha, utility)  # 处理根节点

        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  # return the utility in case the defined depth is reached or the game is won/lost.
                return EvaluationFunction(gameState)  ##如果游戏结束或遍历到depth深度 返回估价值
            if agent == 0:  # maximizing for pacman
                return max(expectimax(1, depth, gameState.getNextState(agent, newState)) for newState in
                           gameState.getLegalActions(agent))
            else:  # performing expectimax action for ghosts/chance nodes.
                nextAgent = agent + 1  # calculate the next agent and increase depth accordingly.
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                return sum(expectimax(nextAgent, depth, gameState.getNextState(agent, newState)) for newState in
                           gameState.getLegalActions(agent)) / float(len(gameState.getLegalActions(agent)))
                # 和minimax区别在这里 不是min 是sum 求期望

        """Performing maximizing task for the root node i.e. pacman"""
        maximum = float("-inf")
        action = Directions.WEST
        for agentState in gameState.getLegalActions(0):  # 根节点 取max
            utility = expectimax(1, 0, gameState.getNextState(0, agentState))
            if utility > maximum or maximum == float("-inf"):
                maximum = utility
                action = agentState

        return action


def EvaluationFunction(currentGameState):
    """
      My extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function.
    """

    newPos = currentGameState.getPacmanPosition()  # 获取当前吃豆人的位置
    newFood = currentGameState.getFood()  # 获取当前状态食物的分布
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    GhostPos = currentGameState.getGhostPositions()  # 获取幽灵的位置
    originalScore = currentGameState.getScore()  # 获取此时的分数
    aroundfood = []  # 存放的是点坐标，表示该点有食物

    # 均采取曼哈顿距离
    dist = min([manhattanDistance(each, newPos) for each in GhostPos])
    if dist != 0 and dist < 4:
        # 计算鬼怪给吃豆人带来的负反馈，当吃豆人与其距离小于4的时候才考虑负反馈
        ghostScore = -11 / dist
    else:
        ghostScore = 0

    Width = newFood.width
    Height = newFood.height

    # #豆子是分散的，当自己所处的位置附近没有豆子时，优先找距离最近的另一堆豆子，这里用用曼哈顿距离计算启发值
    # # 暴力穷举
    # for i in range(Width):
    #     for j in range(Height):
    #         if newFood[i][j]:
    #             aroundfood.append((i, j))

    # if dist >= 2 and len(aroundfood) > 0:
    #     # 周围有豆子且吃豆人距离幽灵要大于等于2
    #     # 求出与吃豆人最近的食物距离
    #     nearestFood = min([manhattanDistance(each, newPos) \
    #                        for each in aroundfood])
    #     foodScore = 10 / nearestFood  # 反比，设置影响因子
    # else:
    #     foodScore = 0

    x_pacman, y_pacman = currentGameState.getPacmanPosition()
    # 搜索距离吃豆人最近的豆子的第二种实现方式: 'BFS'
    nearestFood = float('inf')  # 设为无穷大
    if dist >= 2:  # 如果吃豆人距离幽灵大于等于2
        # 右下左上
        dx = [1, 0, -1, 0]
        dy = [0, 1, 0, -1]
        List = []  # 用列表模拟队列
        d = {}  # 用来标记某一状态是否之前遍历过
        dis = {}  # 用来记录到每个点的迷宫距离
        List.append(newPos)
        d.update({(x_pacman, y_pacman): 1})
        dis.update({(x_pacman, y_pacman): 0})
        while List:
            temp = List[0]
            List.pop(0)
            tx, ty = temp
            if newFood[tx][ty]:
                nearestFood = min(nearestFood, dis[temp])
                break
            for i in range(len(dx)):
                pretemp = (tx, ty)
                x = tx + dx[i]
                y = ty + dy[i]
                if 0 <= x < Width and 0 <= y < Height:  # 判断该点是否出界
                    temp = (x, y)
                    if temp not in d and not currentGameState.hasWall(x, y):  # 之前没有遍历过的状态，就添加到队列末尾
                        d[temp] = 1
                        dis[temp] = dis[pretemp] + 1
                        List.append(temp)

    if nearestFood != float('inf'):
        foodScore = 10 / nearestFood
    else:
        foodScore = 0
    print('total:', originalScore + foodScore + ghostScore, ' origin:', originalScore, ' food:', foodScore, ' ghost:',
          ghostScore, ' near:', nearestFood)
    return originalScore + foodScore + ghostScore

