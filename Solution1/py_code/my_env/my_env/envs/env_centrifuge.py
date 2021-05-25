'''
遠心分離機×4　ON OFF control
'''

import sys
import random
import gym
import numpy as np
import gym.spaces
import math

class CentrifugeEnv(gym.Env):
    '''
    燃料消費：0_max_syouhi_ryou l/mini
    燃料消費変動：up_prob=0.2 keep_prob= 0.6 down_prob=0.2   syouhi_hendo = 0.2
    input 液面高さ：h(t-1) , 燃料消費量：syouhi_ryou , 液面速度：v(t-1) , 出力ポジション：p
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
    syori_per_centrifuge = 40 #l/min
    syori_pos = 5 # 出力ポジション 0,1,2,3,4
    max_actions = 3 # 分離器の増減　 +1,0,-1
    centrifuge_params = [syori_per_centrifuge, syori_pos, max_actions ]

    # 燃料消費
    max_syouhi_ryou = 100 # l/min,
    syouhi_hendo_ritsu = 0.1
    up_prob = 0.2,
    keep_prob = 0.6,
    down_prob = 0.2,
    syouhi_params = [max_syouhi_ryou, syouhi_hendo_ritsu, up_prob , keep_prob, down_prob ]

    # 液面計算
    ss = 10000 # mm2：低面積
    h_max = 600 # タンク高さ
    tank_params = [h_max,ss]
    
    # episode_params
    time_increments = 5 # 時間刻み
    max_steps = 60 # エピソード
    episode_params = [time_increments, max_steps]

    # state

    # rewards
    keep_reward = 0.1
    start_new_contrifuge = -0.5
    rewards = [keep_reward, start_new_contrifuge]

    #----------------------------------------------

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
   # metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, syouhi_params, episode_params, tank_params, centrifuge_params, rewards):
        # 燃料消費：0_max_syouhi_ryou l/mini
        # 出力変動：up_prob=0.2 keep_prob= 0.6 down_prob=0.2
        self.max_syouhi_ryou, self.up_prob, self.keep_prob, self.down_prob,\
             = syouhi_params
        self.time_increments, self.max_steps = episode_params
        self.h_max, self.ss = tank_params
        self.syori_per_centrifuge, self.syori_pos, self.max_actions = centrifuge_params
        self.keep_reward, self.start_new_contrifuge = rewards

        max_v = self.syori_pos * self.syori_per_centrifuge /self.ss
        min_v = -self.max_syouhi_ryou*1000/self.ss

        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Discrete(self.max_action) 
        self.observation_space = gym.spaces.Box(
            # 液面高さ,液面速度,液面加速度,消費率
            low=[0,min_v,-(max_v-min_v)/self.time_increments,0],
            high=[600,max_v,(max_v-min_v)/self.time_increments,1]
            )
        # self.reward_range = [-1., 100.]
        self.viewer = None
        self.state = None
        self._reset()


    def _syouhi_ryou(self, syouhi_ritsu):

        max_syouhi_ryou, up_prob, keep_prob, down_prob = self.syouhi_params
        self.syouhi_ritsu = syouhi_ritsu

        syouhi_hendo = random.random()
        if syouhi_hendo <= self.up_prob:
            next_syouhi_ritsu = self.syouhi_ritsu + syouhi_hendo
            syouhi_ryou = self.max_syouhi_ryou * next_syouhi_ritsu
            if syouhi_ryou >= self.max_syouhi_ryou :
                syouhi_ryou = self.max_syouhi_ryou
                next_syouhi_ritus = 1

        if self.up_prob < syouhi_hendo and syouhi_hendo < down_prob :
            syouhi_ryou = self.syouhi_ritsu_p * max_syouhi_ryou
            next_syouhi_ritsu = syouhi_ritsu

        if (1 - down_prob) <= syouhi_hendo :
            next_syouhi_ritsu = self.syouhi_ritsu - syouhi_hendo
            syouhi_ryou =  max_syouhi_ryou*next_syouhi_ritsu
            if 0 >= syouhi_ryou : 
                syouhi_ryou = 0
                next_syouhi_ritsu = 0

        return syouhi_ryou, next_syouhi_ritsu

    def _step_ekimen(self, action):
        self.action = action

        # input 液面高さ：h(t-1) , 燃料消費量：syouhi , 液面速度：v(t-1) , 遠心分離機稼働数：n

        # output 液面高さ：h(t) , 液面速度：v(t) , 液面加速度：a
        # constants 低面積：ss 

        h_p, v_p, a_p, syouhi_ritsu_p, kado_su_p = self.state

        '''
        1step：5secの燃料消費を積み上げる
        燃料消費は毎秒毎に変動率分だけ下記の確率で増減、または維持する
            燃料消費変動：up_prob=0.2 keep_prob= 0.6 down_prob=0.2   
        その確率は左記の確率で変動する     syouhi_hendo = 0.2
        '''
        one_step_shouhi_ryou = []
        for i in range(self.time_increments):
            if i == 0:
                tmp_syouhi_ryou = self._syouhi_ryou(self, syouhi_ritsu_p)
                one_step_syouhi_ryou = tmp_syouhi_ryou[0]
            else:

                tmp_syouhi_ryou = self._syouhi_ryou(self,syouhi_ritsu_p)
                one_step_syouhi_ryou.append(tmp_syouhi_ryou[0])
                                    
        step_ryou = sum(one_step_syouhi_ryou) 
        next_syouhi_ritsu = tmp_syouhi_ryou[1]

        v_t = (kado_su_p * self.syori_per_contrifuge - step_ryou/self.time_increments)/self.ss
        a_t = (v_t - v_p)/self.time_increments
        h_t = h_p + step_ryou/self.ss

        kado_su_t = kado_su_p + action

        next_state = [h_t,v_t,a_t,next_syouhi_ritsu,kado_su_t]

        reward = self._get_reward(self, h_t, action)

        done = self.is_done(self, h_t)

        print(self.steps)
        return next_state,reward,done


    def _reset(self):
        # 諸々の変数を初期化する
        p_0 = random.random()
        self.h_0 = p_0 * self.h_max
        self.syouhi_ritsu_0 = p_0
        if p_0 != 1 :
            self.kado_su_0 = math.floor(p_0 * self.syori_pos)
        else:
            self.kado_su_0 = self.syori_pos - 1
        self.v_0 = (self.kado_su_0 * self.syori_per_contrifuge - \
            self.syouhi_ritsu * self.max_syouhi_ryou) / self.ss * self.time_increments
        self.a_0 =0
        self.steps = 0
        self.state = self.h_0, self.v_0, self.a_0, self.syouhi_ritsu_0, self.kado_su_0
        return self.state
 
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

    def _get_reward(self, h, action):
        # 報酬を返す。報酬の与え方が難しいが、ここでは
        # 枠の中に液面が入っていれば keep_rewards
        # 新しい遠心分離機を稼働したら start_new_machiine（マイナス）

        if h>0 and 600>h :
            r = self.keep_reward
        if action == 1 :
            r = self.start_new_contrifuge

        return r

    def _is_done(self, h):
        # 今回は最大で self.MAX_STEPS までとした
        if h<0 or self.h_max < h :
            return True
        elif self.steps >= self.max_steps:
            return True
        else:
            return False



