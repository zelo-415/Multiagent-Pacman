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

python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3——？
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=3 -q -n 10——大输特输
python pacman.py -p MinimaxAgent -l openClassic -a depth=3 -g DirectionalGhost
python pacman.py -p MinimaxAgent -l mediumClassic -a depth=3 -g DirectionalGhost
python pacman.py -p MinimaxAgent -l smallClassic -a depth=3 -g DirectionalGhost

python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic -k 1 -q -n 10 全win
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic -k 1 -g DirectionalGhost -q -n 10	 0.90
python pacman.py -p AlphaBetaAgent -a depth=2 -l smallClassic -g DirectionalGhost 
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic -g DirectionalGhost
......



python pacman.py -p ExpectimaxAgent -a depth=3 -l smallClassic -k 1 -q -n 100
