[【人工智能】UC Berkeley 2021春季 CS188 Project 2: Multi-Agent Search Pacman吃豆人游戏_ucb人工智能pacman_夹小汁的博客-CSDN博客](https://blog.csdn.net/weixin_45942927/article/details/120315999)

https://blog.csdn.net/weixin_45771864/article/details/117004460

#### ==一、大致介绍对抗算法（PPT1）==

**减少特征深度m**方法有:

**函数近似**。我们构造一个所谓的估值函数（evaluation function），它的目的是估计某个局面的 Minimax 评分，这样我们我们就不用搜索到终盘才能得知结果。一般而言，越接近终盘估计会越准确（因为游戏往往越到残局越简单），所以估值函数不能完全替代搜索
**模拟估计**。我们可以从某个局面开始，使用一个相对简单的对弈系统（如果和主程序一样复杂就没有意义了），快速下到终局，用得到的结果作为这个局面的 Minimax 评分的近似。显然这个方法也往往越到终盘越准确。此方法又称为 rollout
上面两种方法的混合，比如两个的结果加权平均。这是 AlphaGo-Fan 和 AlphaGo-Lee使用的方式
可以看出，减少特征深度 m 的核心思想就是直接近似得到<u>某个盘面 Minimax 评分</u>，而不是依赖后续的搜索，以截断 Minimax 的搜索深度

而**减少分支因子b**的方法有:

**剪枝**。剪枝就是去除掉某些明显劣势的走法，最著名的便是 Alpha-beta 剪枝。它充分利用了 Minimax 算法的特点，并且仍然可以得到和 Minimax相同的结果（也就是不是近似），是首选的优化
**采样估计**。其思路是蒙特卡洛(monte-carlo)模拟。我们可以先预估出一个先验概率分布 P，然后再根据 P 采样部分路径用作估计。这种手段对分支因子的减少比较激进，在围棋等分支因子较大的棋类中较常用。Alpha系列采用的就是此类方法

（随便拉了一段到时候看能不能用吧）

#### 二、介绍项目背景？pacman（讲清楚题目 ghost pacman 流程图？）

如何把pacman建模成对抗算法问题

##### 互动× 找拖√

整个项目基本结构我觉得可以提提，gamestate,rule等

#### 三、介绍常见算法（MINIMAX\ALPHABETA\蒙特卡洛）

###### 算法 程序  可放实例（备注命令行）

估价函数选择

REFLEX（代码）

==MINIMAX\ALPHABETA（讲代码）==

蒙特卡洛（先介绍井字棋 + 代码）

#### 四、结果呈现

图形化界面（onlyone the best），命令行是：

对比（数据）：胜率，得分，分析

1. smallClassic跑一百次

   贪婪比较稳定，得分和胜率低于后续算法；αβ胜率不稳，胜率和minmax基本持平；优化算法最好

结果不好 - 解释原因（分算法） 引出优化

#### 四、优化-Q4补充-解决GHOST不聪明问题-

<img src="C:\Users\叶\AppData\Roaming\Typora\typora-user-images\image-20230406212941370.png" alt="image-20230406212941370" style="zoom:50%;" />

期望平均

自己选择

#### 五、总结这几种方案  代码心得

#### 命令行：



###### 测试（供选择）：

// 题1：Reflex智能体
python pacman.py    // 通过键盘方向键控制
python pacman.py -p ReflexAgent -l testClassic    // 在默认布局测试Reflex智能体
python pacman.py -p ReflexAgent -l testClassic -g DirectionalGhost    // -g DirectionalGhost 使游戏中的幽灵更聪明和有方向性
python pacman.py -p ReflexAgent -l testClassic -g DirectionalGhost -q -n 10    // -q 关闭图形化界面使游戏更快运行  -n 让游戏运行多次，这段命令是运行了十次
python pacman.py --frameTime 0 -p ReflexAgent -k 2    // 设置2个幽灵，同时提升加速显示
python pacman.py --frameTime 0 -p ReflexAgent -k 1    // 设置1个幽灵，同时提升加速显示

python pacman.py -p ReflexAgent -l openClassic    // 在openClassic布局测试Reflex智能体
python pacman.py -p ReflexAgent -l openClassic -q -n 15
python pacman.py -p ReflexAgent -l openClassic -g DirectionalGhost
python pacman.py -p ReflexAgent -l openClassic -g DirectionalGhost -q -n 10
...... 暴力穷举测试通过



// 题2：Minimax
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3——？
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=3 -q -n 10——大输特输
python pacman.py -p MinimaxAgent -l openClassic -a depth=3 -g DirectionalGhost
python pacman.py -p MinimaxAgent -l mediumClassic -a depth=3 -g DirectionalGhost
python pacman.py -p MinimaxAgent -l smallClassic -a depth=3 -g DirectionalGhost

// 题3: Alpha-Beta
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic -k 1 -q -n 10 全win
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic -k 1 -g DirectionalGhost -q -n 10	 0.90
python pacman.py -p AlphaBetaAgent -a depth=2 -l smallClassic -g DirectionalGhost 
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic -g DirectionalGhost
......



python pacman.py -p ExpectimaxAgent -a depth=3 -l smallClassic -k 1 -q -n 100

#### 代码心得：

1. 在pacman规则中去掉了STOP，不停

#### 剩余工作：

~~修改估价函数 增大豆子的诱惑 pacman_stop~~

~~修改minimax/alphabeta算法  改错~~

逐行注释

数据对比 分数 胜率

新代码 Q4

讲稿 PPT
