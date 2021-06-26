'''
遠心分離機×4　ON OFF control
'''

import sys
import random
import gym
import numpy as np
# import gym.spaces
from gym import spaces, logger
import math

'''
燃料消費：0_MAX_SYOUHI_RYOU l/mini
燃料消費変動：UP_PROB=0.1 KEEP_PROB= 0.8 DOWN_PROB=0.1   syouhi_hendo = 0.2
input 液面高さ：h(t-1) , 燃料消費量：syouhi_ryou , 液面速度：v(t-1) , 出力ポジション：p
output 液面高さ：h(t) , 液面速度：v(t) , 液面加速度：a
constants 低面積：s = 10000mm2  , 時間刻み：TIME_INCREMENTS= 5
state 出力：p [0,1,2,3,4]の5段階　処理量：p*40 l/min

obsevation
    Type : Box(5)
    Num     Observation         Min                  Max
    0       液面位置            0                    600
    1       液面速度            処理0,消費max        処理max,消費0
    2       液面加速度          -(MAX_V-MIN_V)/time  (MAX_V-MIN_V)/TIME_INCREMENTS
    3       燃料消費率          0                    100
    4       遠心分離機稼働数    0                    5

Actions:
    Type: Discrete(3)
    Num     Action
    0       遠心分離機稼働数　一台休止
    1       遠心分離機稼働数　現状維持
    2       遠心分離機稼働数　一台追加

Reward:
    枠の中に液面が入っていれば KEEP_REWARDs
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
SYORI_PER_CENTRIFUGE = 30*1000 #l/min
SYORI_POS = 5 # 出力ポジション 0,1,2,3,4
MAX_ACTIONS = 3 # 分離器の増減　 +1,0,-1
# centrifuge_params = (SYORI_PER_CENTRIFUGE, SYORI_POS, MAX_ACTIONS )

# 燃料消費
MAX_SYOUHI_RYOU = 100*1000 # l/min,
SYOUHI_HENDO_RITSU = 0.1
UP_PROB = 0.1
KEEP_PROB = 0.8
DOWN_PROB = 0.1
# syouhi_params = (MAX_SYOUHI_RYOU, SYOUHI_HENDO_RITSU, UP_PROB , KEEP_PROB, DOWN_PROB )

# 液面計算
SS = 10000 # cm**2：低面積
H_MAX = 1000 # cmタンク高さ
# tank_params = (H_MAX,SS)

# episode_params
TIME_INCREMENTS = 5 # 時間刻み
MAX_STEPS = 60 # エピソード
# episode_params = (TIME_INCREMENTS, MAX_STEPS)

# state

# rewards
KEEP_REWARD = 1.5
OVER_FLOW = -3
START_NEW_CONTRIFUGE = -2
EMPTY = -4
# rewards = (KEEP_REWARD, START_NEW_CONTRIFUGE)

# 燃料消費：0_MAX_SYOUHI_RYOU l/mini
# 出力変動：UP_PROB=0.2 KEEP_PROB= 0.6 DOWN_PROB=0.2

MAX_V = SYORI_POS * SYORI_PER_CENTRIFUGE /SS
MIN_V = -MAX_SYOUHI_RYOU/SS

# action_space, observation_space, reward_range を設定する
# 液面加速度はstep内の出力変動がなければ0だが、出力変動があるため、加速度が発生する
# 5secで 消費率が最大振れ幅の時の速度変動/TIME_INCREMENTS
A_MAX = MAX_SYOUHI_RYOU*SYOUHI_HENDO_RITSU*TIME_INCREMENTS/TIME_INCREMENTS

SCREEN_WIDTH = 1300
SCREEN_HEIGHT = 800
#----------------------------------------------

# def _syouhi_ryou(self, syouhi_ritsu):

    # self.syouhi_ritsu = syouhi_ritsu

    # tmp_p = random.random()

    # if tmp_p <= UP_PROB:
        # next_syouhi_ritsu = self.syouhi_ritsu + SYOUHI_HENDO_RITSU
        # if next_syouhi_ritsu >= 1:
            # next_syouhi_ritus = 1

    # if UP_PROB < tmp_p and tmp_p < UP_PROB+KEEP_PROB :
        # next_syouhi_ritsu = self.syouhi_ritsu

    # if UP_PROB + KEEP_PROB <= tmp_p :
        # next_syouhi_ritsu = syouhi_ritsu - SYOUHI_HENDO_RITSU
        # if 0 >= next_syouhi_ritsu : 
            # next_syouhi_ritsu = 0

    # syouhi_ryou = next_syouhi_ritsu*MAX_SYOUHI_RYOU

    # return syouhi_ryou, next_syouhi_ritsu



class CentrifugeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    # 1clock分の燃料消費量
    def _next_syouhi_ritsu(self, syouhi_ritsu):

        self.syouhi_ritsu = syouhi_ritsu

        tmp_p = random.random()

        if tmp_p <= UP_PROB:
            next_syouhi_ritsu = self.syouhi_ritsu + SYOUHI_HENDO_RITSU
            if next_syouhi_ritsu >= 1:
                next_syouhi_ritsu = 1

        if UP_PROB < tmp_p and tmp_p < UP_PROB+KEEP_PROB :
            next_syouhi_ritsu = self.syouhi_ritsu

        if UP_PROB + KEEP_PROB <= tmp_p :
            next_syouhi_ritsu = syouhi_ritsu - SYOUHI_HENDO_RITSU
            if 0 >= next_syouhi_ritsu : 
                next_syouhi_ritsu = 0


        return next_syouhi_ritsu

    # 1step分の燃料消費の積分
    def _one_step_syouhi(self,syouhi_ritsu):
        self.syouhi_ritsu_p = syouhi_ritsu
        self.one_step_syouhi = []
        for i in range(TIME_INCREMENTS):
            if i == 0:
                tmp_syouhi_ryou = self.syouhi_ritsu_p * MAX_SYOUHI_RYOU
                self.one_step_syouhi.append([self.syouhi_ritsu_p,tmp_syouhi_ryou])
                tmp_next_syouhi_ritsu = self._next_syouhi_ritsu(self.syouhi_ritsu_p)
            else:
                tmp_syouhi_ryou = tmp_next_syouhi_ritsu * MAX_SYOUHI_RYOU
                self.one_step_syouhi.append([tmp_next_syouhi_ritsu, tmp_syouhi_ryou])
                tmp_next_syouhi_ritsu = self._next_syouhi_ritsu(tmp_next_syouhi_ritsu)
                                    
        # 最終clockの情報だけreturn
        # one_step_ryou = sum(np.array(one_step_syouhi)[:,0])
        # one_step_ritsu = np.array(one_step_syouhi)[-1,1]               

        # return one_step_ryou, one_step_ritsu 
        return self.one_step_syouhi

    # metadata = {'render.modes': ['human', 'ansi']}
#    def __init__(self, syouhi_params, episode_params, tank_params, centrifuge_params, rewards):
    def __init__(self):
        self.action_space = spaces.Discrete(MAX_ACTIONS) 
        self.observation_space = spaces.Box(
            # 液面高さ,液面速度,液面加速度,消費率
            low=np.array([0,MIN_V,A_MAX,0,0]),
            high=np.array([600,MAX_V,-A_MAX,1,1])
            )
        # self.reward_range = [-1., 100.]
        self.syouhi_ritsu_random_walk = []
        self.action = None
        self.steps_beyond_done = None
        self.steps = 0
        self.viewer = None
        self.state = None
        self.continuous = None
        self.reset()

    def step(self, action):
        self.action = action - 1

        # input 液面高さ：h(t-1) , 燃料消費量：syouhi , 液面速度：v(t-1) , 遠心分離機稼働数：n
        # output 液面高さ：h(t) , 液面速度：v(t) , 液面加速度：a
        # constants 低面積：SS 

        h_p, v_p, a_p, syouhi_ritsu_p, kado_su_p = self.state

        '''
        1step：5secの燃料消費を積み上げる
        燃料消費は毎秒毎に変動率分だけ下記の確率で増減、または維持する
            燃料消費変動：UP_PROB=0.2 KEEP_PROB= 0.6 DOWN_PROB=0.2   
        その確率は左記の確率で変動する     syouhi_hendo = 0.2
        '''
        tmp_one_step_syouhi = np.array(self._one_step_syouhi(syouhi_ritsu_p)) 
        # print("tmp_one_step_syouhi : ",tmp_one_step_syouhi)
        self.step_ryou = sum(tmp_one_step_syouhi[:,1])
        # print("step_ryou : ",self.step_ryou)
        self.syouhi_ritsu_random_walk.append(tmp_one_step_syouhi[:,0])
        next_syouhi_ritsu = tmp_one_step_syouhi[-1,0]
        # print("next_syouhi_ritsu : ",next_syouhi_ritsu)

        v_t = (kado_su_p * SYORI_PER_CENTRIFUGE * 5 - self.step_ryou/TIME_INCREMENTS)/SS
        a_t = (v_t - v_p)/TIME_INCREMENTS
        tmp_h_t = h_p + (kado_su_p * SYORI_PER_CENTRIFUGE * 5 - self.step_ryou)/SS
        if tmp_h_t >= H_MAX:
            h_t = H_MAX
        elif 0 >= tmp_h_t :
            h_t = 0
        else:
            h_t = tmp_h_t

        tmp_kado_su = kado_su_p + self.action
        if tmp_kado_su >= 4:
            kado_su_t = 4
        elif 0 >= tmp_kado_su:
            kado_su_t = 0
        else:
            kado_su_t = tmp_kado_su

        self.state = np.array([h_t,v_t,a_t,next_syouhi_ritsu, int(kado_su_t)])

        self.done = self._is_done(h_t)

        # rewardとepsodeの途中で終わった時のrwardの処理
        if not self.done:
            self.reward = self._get_reward(h_t, self.action)
        elif self.steps_beyond_done is None:
            # Pole just fell!!
            self.steps_beyond_done = 0
            self.reward = self._get_reward(h_t, self.action)
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            self.reward = 0.0

        self.steps += 1
        # print(self.steps)
        # return self.state,reward,done
        return self.state,self.reward,self.done,{}
        # return self.state,self.reward,self.done,tmp_one_step_syouhi
        # return self.state,reward,done,self.steps


    def reset(self):
        # 諸々の変数を初期化する
        p_t = random.random()
        p_s = random.random()
        p_k = random.random()
        
        self.rener_state = []
        self.done = False
        self.steps_beyond_done = None
        self.steps = 0
        self.episode_rewards = 0

        # self.viewer = None
        if self.continuous:
            return self.state
        else:
            self.h_0 = p_t * H_MAX

            self.syouhi_ritsu_0 = p_s

            if p_k != 1 :
                self.kado_su_0 = math.floor(p_k * SYORI_POS)
            else:
                self.kado_su_0 = SYORI_POS - 1

            self.v_0 = (self.kado_su_0 * SYORI_PER_CENTRIFUGE - \
                self.syouhi_ritsu_0 * MAX_SYOUHI_RYOU) / SS * TIME_INCREMENTS
            tmp_one_step_syouhi = np.array(self._one_step_syouhi(self.syouhi_ritsu_0))
            self.tmp_syouhi_ritsu , self.tmp_ryou = tmp_one_step_syouhi[-1,:]
            self.a_0 = self.tmp_ryou / (TIME_INCREMENTS**2)

            self.state = [self.h_0, self.v_0, self.a_0, self.tmp_syouhi_ritsu, int(self.kado_su_0)]

            return self.state


###############################################
# render
###############################################
    def render(self, mode='human'):
    # def render(self, mode='human', close=False):

        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        # if mode != 'console':
            # raise NotImplementedError

        self.rener_state = []
        self.h, self.v, self.a, self.s_r, self.kado_su = self.state
        
        # self.syouhi_ritsu_random_walk = []
        # self.syouhi_ritsu_random_walk.append(np.array(self.one_step_syouhi)[:,1])

        self.render_kado_su = []
        self.render_kado_su.append(self.kado_su)

        self.render_h = []
        self.render_h.append(self.h)
        graphic_h = 200*self.h/H_MAX

        self.render_reward = []
        self.render_reward.append(self.reward)

        self.render_episode_rewards = []
        self.render_episode_rewards.append(self.episode_rewards)
        
        # print("self.render_state : ",self.render_state)
        tmp_syouhi_ritsu_random_walk = np.array(self.syouhi_ritsu_random_walk)
        len_random_walk = len(tmp_syouhi_ritsu_random_walk)*TIME_INCREMENTS
        tmp_syouhi_ritsu_random_walk = tmp_syouhi_ritsu_random_walk.reshape([len_random_walk,1])
        # print("self.syouhi_ritsu_random_walk : ",tmp_syouhi_ritsu_random_walk.shape)
        
        from gym.envs.classic_control import rendering

        #----------------------------------------------
        # graphics
        #----------------------------------------------
        if self.viewer is None:
            self.viewer = rendering.Viewer(SCREEN_WIDTH, SCREEN_HEIGHT)
            # syouhi_graph
            self.syouhi_graph_x = rendering.Line((50, 650), (650, 650))
            self.syouhi_graph_x.set_color(0, 0, 0)
            self.viewer.add_geom(self.syouhi_graph_x)
            self.syouhi_graph_y = rendering.Line((50, 650), (50, 750))
            self.syouhi_graph_y.set_color(0, 0, 0)
            self.viewer.add_geom(self.syouhi_graph_y)

            # kado_su
            self.kado_su_graph_x = rendering.Line((50, 500), (650, 500))
            self.kado_su_graph_x.set_color(0, 0, 0)
            self.viewer.add_geom(self.kado_su_graph_x)
            self.kado_su_graph_y = rendering.Line((50, 500), (50, 600))
            self.kado_su_graph_y.set_color(0, 0, 0)
            self.viewer.add_geom(self.kado_su_graph_y)

            # ekimen h
            self.h_graph_x = rendering.Line((50, 350), (650, 350))
            self.h_graph_x.set_color(0, 0, 0)
            self.viewer.add_geom(self.h_graph_x)
            self.h_graph_y = rendering.Line((50, 350), (50, 450))
            self.h_graph_y.set_color(0, 0, 0)
            self.viewer.add_geom(self.h_graph_y)

            # reward
            self.reward_graph_x = rendering.Line((50, 200), (650, 200))
            self.reward_graph_x.set_color(0, 0, 0)
            self.viewer.add_geom(self.reward_graph_x)
            self.reward_graph_y = rendering.Line((50, 200), (50, 300))
            self.reward_graph_y.set_color(0, 0, 0)
            self.viewer.add_geom(self.reward_graph_y)

            # episode reward
            self.episode_reward_graph_x = rendering.Line((50, 50), (650, 50))
            self.episode_reward_graph_x.set_color(0, 0, 0)
            self.viewer.add_geom(self.episode_reward_graph_x)
            self.episode_reward_graph_y = rendering.Line((50, 50), (50, 150))
            self.episode_reward_graph_y.set_color(0, 0, 0)
            self.viewer.add_geom(self.episode_reward_graph_y)

            # animation
            self.ani_1 = rendering.Line((750, 550), (1150, 550))
            self.ani_1.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_1)
            self.ani_2 = rendering.Line((750, 550), (750, 350))
            self.ani_2.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_2)
            self.ani_26 = rendering.Line((1150, 550), (1150, 525))
            self.ani_26.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_26)
            self.ani_3 = rendering.Line((1050, 525), (1275, 525))
            self.ani_3.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_3)
            self.ani_4 = rendering.Line((1050, 525), (1050, 500))
            self.ani_4.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_4)
            self.ani_5 = rendering.Line((1125, 525), (1125, 500))
            self.ani_5.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_5)
            self.ani_6 = rendering.Line((1200, 525), (1200, 500))
            self.ani_6.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_6)
            self.ani_7 = rendering.Line((1275, 525), (1275, 500))
            self.ani_7.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_7)

            self.ani_12 = rendering.Line((1050, 425), (1275, 425))
            self.ani_12.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_12)
            self.ani_8 = rendering.Line((1050, 450), (1050, 425))
            self.ani_8.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_8)
            self.ani_9 = rendering.Line((1125, 450), (1125, 425))
            self.ani_9.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_9)
            self.ani_10 = rendering.Line((1200, 450), (1200, 425))
            self.ani_10.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_10)
            self.ani_11 = rendering.Line((1275, 450), (1275, 425))
            self.ani_11.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_11)
            self.ani_13 = rendering.Line((1150, 425), (1150, 200))
            self.ani_13.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_13)
            self.ani_22 = rendering.Line((1000, 200), (1150, 200))
            self.ani_22.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_22)

            self.ani_14 = rendering.Line((700, 350), (1000, 350))
            self.ani_14.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_14)
            self.ani_15 = rendering.Line((700, 350), (700, 150))
            self.ani_15.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_15)
            self.ani_16 = rendering.Line((800, 350), (800, 150))
            self.ani_16.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_16)
            self.ani_17 = rendering.Line((825, 350), (825, 150))
            self.ani_17.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_17)
            self.ani_18 = rendering.Line((1000, 350), (1000, 150))
            self.ani_18.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_18)
            self.ani_19 = rendering.Line((700, 150), (800, 150))
            self.ani_19.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_19)
            self.ani_20 = rendering.Line((825, 150), (1000, 150))
            self.ani_20.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_20)
            self.ani_21 = rendering.Line((800, 325), (825, 325))
            self.ani_21.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_21)
            self.ani_23 = rendering.Line((750, 150), (750, 75))
            self.ani_23.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_23)
            self.ani_24 = rendering.Line((750, 75), (1075, 75))
            self.ani_24.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_24)

            self.ani_geom_25 = rendering.Transform(translation = (1150,75))
            self.ani_motor_1 = rendering.make_circle(75,filled=False)
            self.ani_motor_1.add_attr(self.ani_geom_25)
            self.ani_motor_1.set_color(0, 0, 0)
            self.viewer.add_geom(self.ani_motor_1)

        #----------------------------------------------
        # character
        #----------------------------------------------
        syutsuryoku = math.floor(self.s_r*10)
        hight_of_ekimen = math.floor(self.h/H_MAX*10)

        if self.action != None:
            # print("self.kado_su : ",int(self.kado_su))
            if int(self.kado_su) == 0:
                on_off_1 = False
                on_off_2 = False
                on_off_3 = False
                on_off_4 = False
            if int(self.kado_su) == 1:
                on_off_1 = True
                on_off_2 = False
                on_off_3 = False
                on_off_4 = False
            if int(self.kado_su) == 2:
                on_off_1 = True
                on_off_2 = True
                on_off_3 = False
                on_off_4 = False
            if int(self.kado_su) == 3:
                on_off_1 = True
                on_off_2 = True
                on_off_3 = True
                on_off_4 = False 
            if int(self.kado_su) == 4:
                on_off_1 = True
                on_off_2 = True
                on_off_3 = True
                on_off_4 = True
            # print(int(self.kado_su))
            # self.ani_geom_30 = rendering.Transform()
            # self.ani_geom_30.set_translation(1050,475)
            self.ani_geom_30 = rendering.Transform(translation = (1050,475))
            self.ani_geom_31 = rendering.Transform(translation = (1125,475))
            self.ani_geom_32 = rendering.Transform(translation = (1200,475))
            self.ani_geom_33 = rendering.Transform(translation = (1275,475))
            # print("on_off_1 : ",on_off_1)
            self.ani_centrifuge_1_on = rendering.make_circle(25,filled = on_off_1)
            self.ani_centrifuge_1_on.add_attr(self.ani_geom_30)
            self.ani_centrifuge_1_on.set_color(0, 0, 0)
            self.viewer.add_onetime(self.ani_centrifuge_1_on)
            self.ani_centrifuge_2_on = rendering.make_circle(25, filled = on_off_2)
            self.ani_centrifuge_2_on.add_attr(self.ani_geom_31)
            self.ani_centrifuge_2_on.set_color(0, 0, 0)
            self.viewer.add_onetime(self.ani_centrifuge_2_on)
            self.ani_centrifuge_3_on = rendering.make_circle(25, filled = on_off_3)
            self.ani_centrifuge_3_on.add_attr(self.ani_geom_32)
            self.ani_centrifuge_3_on.set_color(0, 0, 0)
            self.viewer.add_onetime(self.ani_centrifuge_3_on)
            self.ani_centrifuge_4_on = rendering.make_circle(25, filled = on_off_4)
            self.ani_centrifuge_4_on.add_attr(self.ani_geom_33)
            self.ani_centrifuge_4_on.set_color(0, 0, 0)
            self.viewer.add_onetime(self.ani_centrifuge_4_on)

            tank = rendering.FilledPolygon([(700,150),(700,graphic_h+150),(800,graphic_h+150),(800,150)])
            tank.set_color(.1, .1, .8)
            self.tank_trans = rendering.Transform()
            tank.add_attr(self.tank_trans)
            self.viewer.add_onetime(tank)

            print("steps  : ",self.steps,end = "")
            print("   centrifuge : ",self.kado_su,"   ",end = "")
            print(" on "*int(self.kado_su), end = "")
            print("off "*int(SYORI_POS-self.kado_su-1),end = "")
            print("      ",end = "")
            print("syutsuryoku  :",syutsuryoku,"   ",end = "")
            print("*"*syutsuryoku,end = "")
            print("-"*(10-syutsuryoku),end = "")
            print("      ",end = "")
            print(syutsuryoku, "Hight_of_ekimen :",hight_of_ekimen,"  ",end = "")
            if self.h == H_MAX:
                print("++++++++++",end = "")
            elif self.h == 0:
                print("==========",end = "")
            else:
                print("*"*hight_of_ekimen,end = "")
                print("-"*(10-hight_of_ekimen),end = "")
            print("      ",end = "")
            print("reward :",round(self.reward,1),"   action : ",self.action, \
                "     episode_rewards : ",round(self.episode_rewards,1))
        # outfile = StringIO() if mode == 'ansi' else sys.stdout
        # outfile.write('\n'.join(' '.join(
                # self.FIELD_TYPES[elem] for elem in row
                # ) for row in self._observe()
            # ) + '\n'
        # )
        # return outfile

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
#----------------------------------------------

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _seed(self, seed=None):
        pass

    def _get_reward(self, h, action):
        # 報酬を返す。報酬の与え方が難しいが、ここでは
        # 枠の中に液面が入っていれば KEEP_REWARDs
        # 新しい遠心分離機を稼働したら start_new_machiine（マイナス）

        if H_MAX <= h:
            r = OVER_FLOW
        elif h <= 0:
            r = EMPTY
        else:
            r = KEEP_REWARD

        if action == 1 :
            r += START_NEW_CONTRIFUGE

        self.episode_rewards +=r
        return r

    def _is_done(self, h):
        # 今回は最大で self.MAX_STEPS までとした
        # if h<=0 or H_MAX <= h :
        # if h<=0:
            # return True
        # elif self.steps >= MAX_STEPS-1:
            # return True
        if self.steps >= MAX_STEPS-1:
            return True
        else:
            return False


# class Observer(Observer):
    # def transform(self,state) :
        # return np.array(state)



