'''
遠心分離機×4　ON OFF control
'''
+
import sys

import gym
import numpy as np
import gym.spaces


class CentrifugeEnv(gym.Env):
    '''
    燃料消費：0_max_syouhi_ryou l/mini
    燃料消費変動：up_prob=0.2 keep_prob= 0.6 down_prob=0.2
    input 液面高さ：h(t-1) , 燃料消費量：syouhi_ritsu , 液面速度：v(t-1) , 出力ポジション：p
    output 液面高さ：h(t) , 液面速度：v(t) , 液面加速度：a
    constants 低面積：s = 10000mm2  , 時間刻み：time_increments= 5
    state 出力：p [0,1,2,3,4]の5段階　処理量：p*40 l/min

    obsevation
        Type : Box(5)
        Num     Observation         Min                  Max
        0       液面位置            0                    600
        1       液面速度            処理0,消費max        処理max,消費0
        2       液面加速度          -(max_v-min_v)/time  (max_v-min_v)/time_increments
        3       燃料消費率          0                    100
        4       遠心分離機稼働数    0                    5

    Actions:
        Type: Discrete(3)
        Num     Action
        0       遠心分離機稼働数　一台休止
        1       遠心分離機稼働数　現状維持
        2       遠心分離機稼働数　一台追加

    Reward:
        枠の中に液面が入っていれば keep_rewards
        新しい遠心分離機を稼働したら start_new_machiine（マイナス）

    Starting state:
        液面位置：300
        燃料消費率：0
        遠心分離機稼働数：0

    Episode Termination:
        液面高さ= 0 or 600
        step == 60


    '''
    
    #----------------------------------------------
    # constants
    #----------------------------------------------
    # 遠心分離機稼働率変動
    max_syouhi_ryou = 100
    up_prob=0.2
    keep_prob= 0.6
    down_prob=0.2

    # 液面計算
    syori_ryou = 40 #l/min
    ss = 10000 # mm2：低面積
    time_increments = 5 # 時間刻み
    max_steps = 60 # エピソード

    # state
    syori_pos = 5 # 出力ポジション 0,1,2,3,4
    max_acrion = 3 # 分離器の増減　 +1,0,-1

    # rewards
    keep_rewards = 0.1
    start_new_machine = -0.5
 
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
   # metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        # 燃料消費：0_max_syouhi_ryou l/mini
        # 出力変動：up_prob=0.2 keep_prob= 0.6 down_prob=0.2
        self.max_syori_ryou = max_syori_ryou
        self.pre_syori_ritsu = pre_syori_ritsu
        self.up_prob = up_prob
        self.keep_prob = keep_prob
        self.down_prob = 1-down_prob

        max_v = syori_pos * syori_ryou/ss
        min_v = -mmax_syouhi_ryou*1000/ss

        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Discrete(max_action) 
        self.observation_space = gym.spaces.Box(
            # 液面高さ,液面速度,液面加速度,消費率
            low=[0,min_v,-(max_v-min_v)/time_increments,0],
            high=[600,max_v,(max_v-min_v)/time_increments,1]
            )
        # self.reward_range = [-1., 100.]
        self.viewer = None
        self.state = None
        self._reset()

    def _syouhi_ryou(self,max_syouhi_ryou, pre_syori_ritsu,up_prob, keep_prob, down_prob):

        p = ranfom.random()

        if p <= self.up_prob:
            syouhi_ritsu = self.pre_state + max_syouhi_ritsu*p
            if syouhi_ritsu >= max_syouhi_ritsu :
                syouhi_ritsu = max_syouhi_ritsu

        if self.up_prob < p and p < self.down_prob :
            syouhi_ritsu = self.pre_state

        if self.down_prob <= p :
            syouhi_ritsu = pre_state - max_syouhi_ritsu*p
            if 0 >= syouhi_ritsu : 
                syouhi_ritsu = 0

        return syouhi_ritsu , syouhi_ritsu * max_syouhi_ryou

    def _step_ekimen(self, action):

        # input 液面高さ：h(t-1) , 燃料消費量：syouhi , 液面速度：v(t-1) , 出力ポジション：p
        # output 液面高さ：h(t) , 液面速度：v(t) , 液面加速度：a
        # constants 低面積：ss 

        h_p, v_p, a_t, syouhi_ritsu_p, kado_su_p = self.state

        syori_ryou = []
        for i in range(time_increments):
            if i == 0:
                tmp_syori_ryou = _syouhi_ryou(self, syori_ritsu_p)
                syori_ryou = tmp_syori_riyou
            else:
                tmp_syori_ryou = _syouhi_ryou(self,max_syouhi_ryou, tmp_syori_ryou[0],up_prob, keep_prob, down_prob)
                syori_ryou.append(tmp_syori_ryou)
                                    
        one_step_ryou = sum(syori_ryou[1]) 
        v_p = p_p,syori_ryou/ss - syori_ryou[0][1]
        v_t = p_t*syori_ryou/ss - syori_ryou[4][1]
        a_t = (v_t - v_p)/time_increments
        h_t = h_p + one_step_ryou/ss
        state = [h_t,v_t,a_t]

        reward = self._get_reward(self.p, state)

        self.done = _is_done(self)

        return state,reward,done


    def _reset(self):
        # 諸々の変数を初期化する
        self.pos = self._find_pos('S')[0]
        self.goal = self._find_pos('G')[0]
        self.done = False
        self.damage = 0
        self.steps = 0
        return self._observe()

    def _render(self, mode='human', close=False):
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('\n'.join(' '.join(
                self.FIELD_TYPES[elem] for elem in row
                ) for row in self._observe()
            ) + '\n'
        )
        return outfile

    def _close(self):
        pass

    def _seed(self, seed=None):
        pass

    def _get_reward(self, p, state):
        # 報酬を返す。報酬の与え方が難しいが、ここでは
        # 枠の中に液面が入っていれば keep_rewards
        # 新しい遠心分離機を稼働したら start_new_machiine（マイナス）

        if state.h>0 and 600>state.h :
            r += keep_rewards
        if action == 1 :
            r += start_new_machine

        return r

    def _is_done(self):
        # 今回は最大で self.MAX_STEPS までとした
        if h<0 or 600<h :
            return True
        elif self.steps >= self.max_steps:
            return True
        else:
            return False

class CentrifugeObserver(observer):
    def transform(self,state) :
        return np.array(state)


