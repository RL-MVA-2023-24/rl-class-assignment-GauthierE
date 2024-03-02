from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import gymnasium as gym
import random
import numpy as np
# from tqdm import tqdm # comment this line before pushing
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
# import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

###################################################################################
###################################################################################
# DQN
###################################################################################
###################################################################################

# def greedy_action(network, state):
#     with torch.no_grad():
#         Q = network(torch.Tensor(state).unsqueeze(0))
#         return torch.argmax(Q).item()
    
    
# state_dim = env.observation_space.shape[0]
# n_action = env.action_space.n 
# nb_neurons=64

# class DQNNetwork(nn.Module):
#     def __init__(self, state_dim, nb_neurons, n_action):
#         super(DQNNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dim, nb_neurons)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(nb_neurons, nb_neurons)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(nb_neurons, nb_neurons)
#         self.relu3 = nn.ReLU()
#         self.fc4 = nn.Linear(nb_neurons, nb_neurons)
#         self.relu4 = nn.ReLU()
#         self.fc5 = nn.Linear(nb_neurons, n_action)

#     def forward(self, x):
#         x = self.relu1(self.fc1(x))
#         x = self.relu2(self.fc2(x))
#         x = self.relu3(self.fc3(x))
#         x = self.relu4(self.fc4(x))
#         x = self.fc5(x)
#         return x

# # DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
# #                         nn.ReLU(),
# #                         nn.Linear(nb_neurons, nb_neurons),
# #                         nn.ReLU(), 
# #                         nn.Linear(nb_neurons, n_action))

# # DQN config
# config = {'nb_actions': env.action_space.n,
#           'learning_rate': 0.01,
#           'gamma': 0.95,
#           'buffer_size': 100000000,
#           'epsilon_min': 0.01,
#           'epsilon_max': 1.,
#           'epsilon_decay_period': 100000, #1000
#           'epsilon_delay_decay': 20,
#           'batch_size': 100} #20

# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.capacity = capacity # capacity of the buffer
#         self.data = []
#         self.index = 0 # index of the next cell to be filled
#     def append(self, s, a, r, s_, d):
#         if len(self.data) < self.capacity:
#             self.data.append(None)
#         self.data[self.index] = (s, a, r, s_, d)
#         self.index = (self.index + 1) % self.capacity
#     def sample(self, batch_size):
#         batch = random.sample(self.data, batch_size)
#         return list(map(lambda x:torch.Tensor(np.array(x)), list(zip(*batch))))
#     def __len__(self):
#         return len(self.data)


# # You have to implement your own agent.
# # Don't modify the methods names and signatures, but you can add methods.
# # ENJOY!
# class ProjectAgent:

#     def __init__(self):
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.gamma = config['gamma']
#         self.batch_size = config['batch_size']
#         self.nb_actions = config['nb_actions']
#         self.memory = ReplayBuffer(config['buffer_size'])
#         self.epsilon_max = config['epsilon_max']
#         self.epsilon_min = config['epsilon_min']
#         self.epsilon_stop = config['epsilon_decay_period']
#         self.epsilon_delay = config['epsilon_delay_decay']
#         self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
#         self.model = DQNNetwork(state_dim, nb_neurons, self.nb_actions).to(device)
#         # self.target_model = DQNNetwork(state_dim, config['nb_neurons'], self.nb_actions).to(device).eval()
#         self.criterion = torch.nn.MSELoss()
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])

#     def gradient_step(self):
#         if len(self.memory) > self.batch_size:
#             X, A, R, Y, D = self.memory.sample(self.batch_size)
#             QYmax = self.model(Y).max(1)[0].detach()
#             update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
#             QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
#             loss = self.criterion(QXA, update.unsqueeze(1))
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#     def train(self, env, max_episode):

#         episode_return = []
#         episode = 0
#         episode_cum_reward = 0
#         state, _ = env.reset()
#         epsilon = self.epsilon_max
#         step = 0

#         while episode < max_episode:


#             # update epsilon
#             if step > self.epsilon_delay:
#                 epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

#             # select epsilon-greedy action
#             if np.random.rand() < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 action = self.act(state, use_random=True)


#             # step
#             next_state, reward, done, trunc, _ = env.step(action)
#             self.memory.append(state, action, reward, next_state, done)
#             episode_cum_reward += reward


#             # train
#             self.gradient_step()


#             # next transition
#             step += 1
#             if done or step==200:
#                 step = 0
#                 episode += 1
#                 print("Episode ", '{:3d}'.format(episode),
#                       ", epsilon ", '{:6.2f}'.format(epsilon),
#                       ", batch size ", '{:5d}'.format(len(self.memory)),
#                       ", episode return ", '{:4.1f}'.format(episode_cum_reward),
#                       sep='')
#                 state, _ = env.reset()
#                 episode_return.append(episode_cum_reward)
#                 episode_cum_reward = 0
#             else:
#                 state = next_state

#             # print(f'episode = {episode}')
#             # if step%10==0:
#             #     print(f'step = {step}')


#         return episode_return

#     def act(self, observation, use_random=False):
#         if use_random and np.random.rand() < self.epsilon_max:
#             return np.random.choice(self.nb_actions)

#         state = torch.FloatTensor(observation).unsqueeze(0)
#         q_values = self.model(state)
#         action = torch.argmax(q_values).item()
#         return action

#     # def update_target_model(self):
#     #     self.target_model.load_state_dict(self.model.state_dict())

#     def save(self, path):
#         torch.save({
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict()
#         }, path)

#     def load(self):
#         checkpoint = torch.load("model_save")
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         # self.update_target_model()

#     # def act(self, observation, use_random=False):
#     #     return env.action_space.sample()

#     # def save(self, path):
#     #     pass

#     # def load(self):
#     #     pass


# # class dqn_agent:
# #     def __init__(self, config, model):
# #         device = "cuda" if next(model.parameters()).is_cuda else "cpu"
# #         self.gamma = config['gamma']
# #         self.batch_size = config['batch_size']
# #         self.nb_actions = config['nb_actions']
# #         self.memory = ReplayBuffer(config['buffer_size'], device)
# #         self.epsilon_max = config['epsilon_max']
# #         self.epsilon_min = config['epsilon_min']
# #         self.epsilon_stop = config['epsilon_decay_period']
# #         self.epsilon_delay = config['epsilon_delay_decay']
# #         self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
# #         self.model = model 
# #         self.criterion = torch.nn.MSELoss()
# #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
    
# #     def gradient_step(self):
# #         if len(self.memory) > self.batch_size:
# #             X, A, R, Y, D = self.memory.sample(self.batch_size)
# #             QYmax = self.model(Y).max(1)[0].detach()
# #             #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
# #             update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
# #             QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
# #             loss = self.criterion(QXA, update.unsqueeze(1))
# #             self.optimizer.zero_grad()
# #             loss.backward()
# #             self.optimizer.step() 
    
# #     def train(self, env, max_episode):
# #         episode_return = []
# #         episode = 0
# #         episode_cum_reward = 0
# #         state, _ = env.reset()
# #         epsilon = self.epsilon_max
# #         step = 0

# #         while episode < max_episode:
# #             # update epsilon
# #             if step > self.epsilon_delay:
# #                 epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

# #             # select epsilon-greedy action
# #             if np.random.rand() < epsilon:
# #                 action = env.action_space.sample()
# #             else:
# #                 action = greedy_action(self.model, state)

# #             # step
# #             next_state, reward, done, trunc, _ = env.step(action)
# #             self.memory.append(state, action, reward, next_state, done)
# #             episode_cum_reward += reward

# #             # train
# #             self.gradient_step()

# #             # next transition
# #             step += 1
# #             if done:
# #                 episode += 1
# #                 print("Episode ", '{:3d}'.format(episode), 
# #                       ", epsilon ", '{:6.2f}'.format(epsilon), 
# #                       ", batch size ", '{:5d}'.format(len(self.memory)), 
# #                       ", episode return ", '{:4.1f}'.format(episode_cum_reward),
# #                       sep='')
# #                 state, _ = env.reset()
# #                 episode_return.append(episode_cum_reward)
# #                 episode_cum_reward = 0
# #             else:
# #                 state = next_state

# #         return episode_return
    


# # # Train agent
# # agent = dqn_agent(config, DQN)
# # scores = agent.train(cartpole, 200)
# # plt.plot(scores)

###################################################################################
###################################################################################
# end DQN
###################################################################################
###################################################################################

###################################################################################
###################################################################################
# FQI
###################################################################################
###################################################################################

def collect_samples(env, horizon, disable_tqdm=False, print_done_states=False):
    s, _ = env.reset()
    #dataset = []
    S = []
    A = []
    R = []
    S2 = []
    D = []
    for _ in tqdm(range(horizon), disable=disable_tqdm):
        a = env.action_space.sample()
        s2, r, done, trunc, _ = env.step(a)
        #dataset.append((s,a,r,s2,done,trunc))
        S.append(s)
        A.append(a)
        R.append(r)
        S2.append(s2)
        D.append(done)
        if done or trunc:
            s, _ = env.reset()
            if done and print_done_states:
                print("done!")
        else:
            s = s2
    S = np.array(S)
    A = np.array(A).reshape((-1,1))
    R = np.array(R)
    S2= np.array(S2)
    D = np.array(D)

    # add a normalization step here for S and S2
    mS = np.min(S, axis=0)
    MS = np.max(S, axis=0)
    S_normalized = (S - mS) / (MS - mS)
    S2_normalized = (S2 - mS) / (MS - mS)

    # return S, A, R, S2, D
    return S_normalized, A, R, S2_normalized, D, mS, MS

def rf_fqi(S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
    # nb_samples = S.shape[0]
    Qfunctions = []
    SA = np.append(S,A,axis=1)
    for iter in tqdm(range(iterations), disable=disable_tqdm):
        if iter==0:
            value=R.copy()
        else:
            Q2 = np.zeros((nb_samples,nb_actions))
            for a2 in range(nb_actions):
                A2 = a2*np.ones((S.shape[0],1))
                S2A2 = np.append(S2,A2,axis=1)
                # dtest = xgb.DMatrix(S2A2)
                Q2[:,a2] = Qfunctions[-1].predict(S2A2)
                # Q2[:,a2] = Qfunctions[-1].predict(dtest)
            max_Q2 = np.max(Q2,axis=1)
            value = R + gamma*(1-D)*max_Q2

        ############################################# xgboost
        # dtrain = xgb.DMatrix(SA, label=value)
        # params = {
        #     'objective': 'reg:squarederror',
        #     'eval_metric': 'rmse',
        #     'max_depth': 3,
        #     'eta': 0.1  
        # }
        # bst = xgb.train(params, dtrain, num_boost_round=10) 

        # Save only the last Qfunction to save space
        # Qfunctions = [bst]
        # Qfunctions.append(bst)

        ############################################# random forest
        # Q = RandomForestRegressor()
        # Q.fit(SA,value)
        ## Qfunctions.append(Q) 
        ## save memory: save only the last Qfunction
        # Qfunctions = [Q]
        # Value of an initial state across iterations
        
        ############################################# ExtraTrees
        Q = ExtraTreesRegressor(n_estimators=100)
        Q.fit(SA, value)
        # # Qfunctions.append(Q)
        Qfunctions = [Q]
            
    # s0,_ = env.reset()
    # Vs0 = np.zeros(nb_iter)
    # for i in range(nb_iter):
    #     Qs0a = []
    #     for a in range(env.action_space.n):
    #         s0a = np.append(s0,a).reshape(1, -1)
    #         # s0a_d = xgb.DMatrix(s0a)
    #         # Qs0a.append(Qfunctions[i].predict(s0a_d))
    #         Qs0a.append(Qfunctions[i].predict(s0a))
    #     Vs0[i] = np.max(Qs0a)
    # plt.plot(Vs0)
    # return Qfunctions
        
    return Qfunctions
    # return [Qfunctions[-1]] # return only the last Qfunction to save space

def greedy_action(Q,s,nb_actions):
    Qsa = []
    for a in range(nb_actions):
        sa = np.append(s,a).reshape(1, -1)
        # sa_dmatrix = xgb.DMatrix(sa)
        Qsa.append(Q.predict(sa))
        # Qsa.append(Q.predict(sa_dmatrix))
    return np.argmax(Qsa)

gamma = 0.98
nb_iter = 50
nb_actions = env.action_space.n
nb_samples = 20000
# print('Calculating Qfunctions...')
# Qfunctions = rf_fqi(S, A, R, S2, D, max_episode, nb_actions, gamma)
# print('Qfunctions...')

### fqi2 : nb_iter=10000, nb_samples=15000
### fqi3 : nb_iter=7500, nb_samples=15000
### fqi4 : nb_iter=2000, nb_samples=10000
### fqi5: fqi4 with gamma = 1 istead of .9 => seems to lead to improvements
### fqi6 : gamma = 1, nb samples = 10000, nb_iter = 100 (trying something new), and normalization of S and S2 => not very good
### fqi7 : iter 2000 samples 10000 normalization + 0.1-greedy
### fqi8 : fqi7 without 0.1-greedy at the execution, and gamma = 0.98, and applciation of renormalization to new observation
### fqi9 : iter 7500 samples 10000 gamma = 0.98, renorm
### fqi_kaggle_xgb/rf : gamma 0.98 iter 2000 samples 10000
### fqi10 : fqi9 with iter 50 (easier to see impact of parameters)
### fqi 11 : fqi 10 without renorm. trying dump compress=9
### fqi 12 : trying xgb. gamme 0.98 iter 1000 samples 10000 max_depth 3 eta 0.1 num_boost_round 10 => not working, give up xgb
### fqi 13 : trying extratrees. gamme 0.98 iter 2000 samples 10000 n_estimators 100
### fqi 14 : trying extratrees. gamme 0.98 iter 2000 samples 10000 n_estimators 300 => no improvement
### fqi 15 : samples 30k iter 100 gamma 0.99 extratrees n_estimators 100
### fqi 16 : samples 30k iter 50 random forest

class ProjectAgent:

    def __init__(self):
        self.Qfunctions = []
        self.S = []
        self.A = []
        self.R = []
        self.S2 = []
        self.D = []
        #renormalization parameters
        self.mS = np.array((6,))
        self.MS = np.array((6,))

    def collect_samples(self):
        print('Collecting samples...')
        self.S,self.A,self.R,self.S2,self.D,self.mS,self.MS = collect_samples(env, nb_samples)
        # self.S,self.A,self.R,self.S2,self.D = collect_samples(env, nb_samples)
        print('Samples collected.')
        print("nb of collected samples:", self.S.shape[0])
        # for i in range(3):
        #     print("sample", i, "\n  state:", self.S[i], "\n  action:", self.A[i],
        #           "\n  reward:", self.R[i], "\n  next state:", self.S2[i], "\n terminal?", self.D[i])
        # print(f'renormalization parameters: [{self.mS},{self.MS}] for S and [{self.mS2},{self.MS2}] for S2')

    def train(self):
        print('Start training...')
        self.Qfunctions = rf_fqi(self.S, self.A, self.R, self.S2, self.D, nb_iter, nb_actions, gamma)
        print('Training completed.')

    def act(self, observation, use_random=False):
        observation = (observation - np.array(self.mS)) / (np.array(self.MS) - np.array(self.mS))
        # print('observation after preprocessing')
        # print(observation)
        return greedy_action(self.Qfunctions[-1],observation,env.action_space.n)

    def save(self, path):
        # dump(self.Qfunctions, path)
        data_to_save = {
            'Qfunctions': self.Qfunctions[-1],
            'mS': self.mS,
            'MS': self.MS
        }
        dump(data_to_save, path, compress=9)

    def load(self):
        # self.Qfunctions = load("model_save_fqi_11")
        loaded_data = load("src/model_save_fqi_17")
        self.Qfunctions = loaded_data['Qfunctions']
        self.mS = loaded_data['mS']
        self.MS = loaded_data['MS']

###################################################################################
###################################################################################
# end FQI
###################################################################################
###################################################################################

    # def act(self, observation, use_random=False):
    #     return env.action_space.sample()

    # def save(self, path):
    #     pass

    # def load(self):
    #     pass
