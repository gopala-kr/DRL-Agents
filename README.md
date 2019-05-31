
----------

##### Contents

- [RL Landscape](#rl-landscape)
- [RL History](#rl-history)
- [RL Agents Implementation](#rl-agents-implementation)
  - [Value Optimization Agents](#value-optimization-agents)
  - [Policy Optimization Agents](#policy-optimization-agents)
  - [General Agents](#general-agents)
  - [Imitation Learning Agents](#imitation-learning-agents)
  - [Hierarchical Reinforcement Learning Agents](#hierarchical-reinforcement-learning-agents)
  - [Memory Types](#memory-types)
  - [Exploration Techniques](#exploration-techniques)
- [RL Environments](#rl-environments)
- [RL Mechanisms](#rl-mechanisms)
- [RL Applications](#)
  - [RL Games](#rl-games)
  - [DRL applied to Robotics](#drl-applied-to-robotics)
  - [DRL applied to NLP](#drl-applied-to-nlp)
  - [DRL applied to Vision](#drl-applied-to-vision)
- [References](#references)
  - Reference Implementations
  - Review Papers
  - RL Platforms
  - Deep Reinforcement Learning Papers

[Back to top](#contents)



----------------

![deep_rl](https://github.com/gopala-kr/DRL-Agents/blob/master/resources/img/drl.PNG)

------------
#### RL Landscape

[Back to top](#contents)

![68747470733a2f2f706c616e73706163652e6f72672f32303137303833302d6265726b656c65795f646565705f726c5f626f6f7463616d702f696d672f616e6e6f74617465642e6a7067](https://camo.githubusercontent.com/9f59450ab0458e82c4d728415a4d0f1671ea8a48/68747470733a2f2f706c616e73706163652e6f72672f32303137303833302d6265726b656c65795f646565705f726c5f626f6f7463616d702f696d672f616e6e6f74617465642e6a7067)





--------------

#### RL Agents Implementation

[Back to top](#contents)


![algorithms](https://github.com/NervanaSystems/coach/blob/master/img/algorithms.png)

   - Value Optimization
       - [QR-DQN]
       - [DQN] - [[Slides](https://drive.google.com/file/d/0BxXI_RttTZAhVUhpbDhiSUFFNjg/view)]  [[Code](https://github.com/deepmind/dqn)] [[rainbow](https://github.com/hengyuan-hu/rainbow)]
       - [Bootstrapped DQN]
       - [DDQN]
       - [NEC]
       - [MMC]
       - [N-step Q Learning]
       - [PAL]
       - [Categorical DQN]
       - [NAF]
   - Policy Optimization
       - [Policy Gradient]
       - [Actor Critic]
         - [DDPG] [[Code](https://github.com/ghliu/pytorch-ddpg)]
           - [HAC DDPG]
           - [DDPG with HER]
         - [Clipped PPO]
         - [PPO]
   - [DFP]
   - Imitation
       - [Behavioural cloning]
       - [Inverse Reinforcement Learning] [[Code](https://github.com/MatthewJA/Inverse-Reinforcement-Learning)] [[irl-imitation-code](https://github.com/yrlu/irl-imitation)]
       - [Generative Adversarial Imitation Learning]
       
-----------

##### Value Optimization Agents

[Back to top](#contents)

* [Deep Q Network (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/dqn_agent.py))
* [Double Deep Q Network (DDQN)](https://arxiv.org/pdf/1509.06461.pdf)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/ddqn_agent.py))
* [Dueling Q Network](https://arxiv.org/abs/1511.06581)
* [Mixed Monte Carlo (MMC)](https://arxiv.org/abs/1703.01310)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/mmc_agent.py))
* [Persistent Advantage Learning (PAL)](https://arxiv.org/abs/1512.04860)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/pal_agent.py))
* [Categorical Deep Q Network (C51)](https://arxiv.org/abs/1707.06887)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/categorical_dqn_agent.py))
* [Quantile Regression Deep Q Network (QR-DQN)](https://arxiv.org/pdf/1710.10044v1.pdf)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/qr_dqn_agent.py))
* [N-Step Q Learning](https://arxiv.org/abs/1602.01783) | **Distributed**  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/n_step_q_agent.py))
* [Neural Episodic Control (NEC)](https://arxiv.org/abs/1703.01988)  ([code](rl_coach/agents/nec_agent.py))
* [Normalized Advantage Functions (NAF)](https://arxiv.org/abs/1603.00748.pdf) | **Distributed**  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/naf_agent.py))


##### Policy Optimization Agents

[Back to top](#contents)


* [Policy Gradients (PG)](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) | **Distributed**  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/policy_gradients_agent.py))
* [Asynchronous Advantage Actor-Critic (A3C)](https://arxiv.org/abs/1602.01783) | **Distributed**  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/actor_critic_agent.py))
* [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971) | **Distributed**  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/ddpg_agent.py))
* [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/ppo_agent.py))
* [Clipped Proximal Policy Optimization (CPPO)](https://arxiv.org/pdf/1707.06347.pdf) | **Distributed**  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/clipped_ppo_agent.py))
* [Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438) ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/actor_critic_agent.py#L86))

##### General Agents

[Back to top](#contents)

* [Direct Future Prediction (DFP)](https://arxiv.org/abs/1611.01779) | **Distributed**  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/dfp_agent.py))

##### Imitation Learning Agents

[Back to top](#contents)

* Behavioral Cloning (BC)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/bc_agent.py))

##### Hierarchical Reinforcement Learning Agents

[Back to top](#contents)

* [Hierarchical Actor Critic (HAC)](https://arxiv.org/abs/1712.00948.pdf) ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/ddpg_hac_agent.py))

##### Memory Types

[Back to top](#contents)

* [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495.pdf) ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/memories/episodic/episodic_hindsight_experience_replay.py))
* [Prioritized Experience Replay (PER)](https://arxiv.org/abs/1511.05952) ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/memories/non_episodic/prioritized_experience_replay.py))

##### Exploration Techniques

[Back to top](#contents)

* E-Greedy ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/exploration_policies/e_greedy.py))
* Boltzmann ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/exploration_policies/boltzmann.py))
* Ornstein–Uhlenbeck process ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/exploration_policies/ou_process.py))
* Normal Noise ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/exploration_policies/additive_noise.py))
* Truncated Normal Noise ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/exploration_policies/truncated_normal.py))
* [Bootstrapped Deep Q Network](https://arxiv.org/abs/1602.04621)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/bootstrapped_dqn_agent.py))
* [UCB Exploration via Q-Ensembles (UCB)](https://arxiv.org/abs/1706.01502) ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/exploration_policies/ucb.py))
* [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295) ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/exploration_policies/parameter_noise.py))

-------------

#### RL History

[Back to top](#contents)


- Temporal difference(TD) learning (1988)
- Q‐learning (1998)
- BayesRL (2002)
- RMAX (2002)
- CBPI (2002)
- PEGASUS (2002)
- Least‐Squares Policy Iteration (2003)
- Fitted Q‐Iteration (2005)
- GTD (2009)
- UCRL (2010)
- REPS (2010)
- DQN (2014) - DeepMind

----------

[Back to top](#contents)

![awesome](https://raw.githubusercontent.com/tigerneil/awesome-deep-rl/master/images/awesome-drl.png)


---------

[Back to top](#contents)

![landscape](https://raw.githubusercontent.com/tangzhenyu/Reinforcement-Learning-in-Robotics/master/images/landscape.jpeg)
---------
#### RL Environments

[Back to top](#contents)

- [Acrobot]
- [Bike]
- [Blackjack]
- [Cartpole]
- [ContextBandit]
- [Continuous Chain]
- [Corridor]
- [Discrete Chain]
- [Discretiser (for continuous environments)]
- [Double Loop]
- [Environment]
- [Gridworld]
- [Inventory management]
- [Linear context bandit]
- [Linear dynamic quadratic]
- [Mountaincar (2d and 3d)]
- [POMDP Maze]
- [Optimistic Task]
- [Puddleworld]
- [Random MDPs]
- [Riverswim]

----------

#### RL Mechanisms

[Back to top](#contents)


- [Attention and Memory]
- [Unsupervised learning ]
  - [GANs]
  - [GQN]
  - [UNREAL]
- [Hierarchical RL]
  - [FuNs]
  - [Option-Critic]
  - [STRAW]
  - [h-DQN]
  - [Stochastic Neural Networks]
- [Multi-agent RL]
- [Relational RL]
- [Learning to Learn, a.k.a. Meta-Learning]
  - [Few/One/Zero-shot Learning]
    - [MAML]
  - [Transfer and Multi-Task Learning]
  - [Learning to Optimize]
  - [Learning to Re-inforcement Learn]
  - [Learning Combinatorial Optimization]
  - [AutoML]
  
-------------------

#### RL Games

[Back to top](#contents)


- Chinook (1997;2007) for Checkers,
- Deep Blue (2002) for chess,
- Logistello (1999) for Othello,
- TD-Gammon (1994) for Backgammon,
- GIB (2001) for contract bridge,
- MoHex (2017) for Hex,
- DQN (2016)(2018) for Atari 2600 games,
- AlphaGo (2016a) and AlphaGo Zero (2017) for Go,
- Alpha Zero (2017) for chess, shogi, and Go,
- Cepheus (2015), DeepStack (2017), and Libratus (2017a;b) for heads-up Texas Hold’em Poker,
- Jaderberg et al. (2018) for Quake III Arena Capture the Flag,
- OpenAI Five, for Dota 2 at 5v5, https://openai.com/five/,
- Zambaldi et al. (2018), Sun et al. (2018), and Pang et al. (2018) for StarCraft II

-----------------


[Back to top](#contents)

- [Board Games]
  - [Computer Go]
  - [AlphaGo: Trainig pipeline with MCTS]
  - [AlphaGo Zero]
  - [Alpha Zero]
- [Card Games]
  - [DeepStack]
- [Video Games]
  - [Atari 2600 games]
  - [StarCraft]
  - [StarCraft
II mini-games]
  - [Quake III Arena]
  - [Minecraft]
  - [Super Smash Bros]
  - [Doom]
  - [ViZDoom]
  
------------

#### DRL applied to Robotics

[Back to top](#contents)


   - [Sim-to-Real]
     - [MuJoCo]
   - [Imitation Learning]
   - [Value-based Learning]
   - [Policy-based Learning]
   - [Model-based Learning]
   - [Autonomous Driving Vehicles]

-------------

#### DRL applied to NLP

[Back to top](#contents)

- [Sequence Generation]
- [Machine Translation]
- [Dialogue Systems]

------------

#### DRL applied to Vision

[Back to top](#contents)

- [Recognition]
- [Motion Analysis]
- [Scene Understanding]
- [Vision + NLP]
- [Visual Control]
- [Interactive Perception]


----------------

#### References

[Back to top](#contents)


- [reference implementations](https://github.com/gopala-kr/reinforce-tf/blob/master/ref-implementations.md)
- [review papers](https://github.com/gopala-kr/reinforce-tf/blob/master/review-papers.md)
- [RL platforms](https://github.com/gopala-kr/DRL-Agents/blob/master/platforms.md)
- [deep-reinforcement-learning-papers](https://github.com/junhyukoh/deep-reinforcement-learning-papers)
- [ICLR2019-RL-Papers](https://github.com/ewanlee/ICLR2019-RL-Papers)

-------------
- [deepmind.com/blog](https://deepmind.com/blog/)
- [blog.openai](https://blog.openai.com/)
- [openai/spinningup](https://github.com/openai/spinningup)
- [DeepRLHacks](https://github.com/williamFalcon/DeepRLHacks)-Deep RL Bootcamp (Aug 2017)
- [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)
- [UC Berkeley: Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/)
- [MIT 6.S094: Deep Learning for Self-Driving Cars](https://selfdrivingcars.mit.edu/)
- [Deep Reinforcement Learning and Control 
Spring 2017, CMU 10703](https://katefvision.github.io/#readings)
- [Sutton & Barto's: Reinforcement Learning: An Introduction](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
- [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)
- [reinforcejs](https://cs.stanford.edu/people/karpathy/reinforcejs/index.html)
- [Hands-On-Reinforcement-Learning-With-Python](https://github.com/sudharsan13296/Hands-On-Reinforcement-Learning-With-Python)
- [jetson-reinforcement](https://github.com/dusty-nv/jetson-reinforcement)
- [DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/1810.06339v1.pdf)
- [Reinforcement Learning Applications](https://medium.com/@yuxili/rl-applications-73ef685c07eb)
- [Lessons Learned Reproducing a Deep Reinforcement Learning Paper](http://amid.fish/reproducing-deep-rl)
- [Scalable Deep Reinforcement Learning for Robotic Manipulation](https://ai.googleblog.com/2018/06/scalable-deep-reinforcement-learning.html)
- [Closing the Simulation-to-Reality Gap for Deep Robotic Learning](https://ai.googleblog.com/2017/10/closing-simulation-to-reality-gap-for.html)
- [Intel AI : demystifying-deep-reinforcement-learning](https://ai.intel.com/demystifying-deep-reinforcement-learning/)
- [Intel AI : deep-reinforcement-learning-with-neon](https://ai.intel.com/deep-reinforcement-learning-with-neon/)
- [Intel AI : reinforcement-learning-coach-intel](https://ai.intel.com/reinforcement-learning-coach-intel/)
- [eecs.berkeley.deeprlcourse](http://rail.eecs.berkeley.edu/deeprlcourse/)
- [Deep Learning for Video Game Playing](https://arxiv.org/pdf/1708.07902.pdf)
- [Deep Reinforcement Learning that Matters](https://arxiv.org/pdf/1709.06560.pdf)
- [Adversarial Examples: Attacks and Defenses for Deep Learning](https://arxiv.org/pdf/1712.07107.pdf)
- [Reinforcement learning](https://yandexdataschool.com/edu-process/rl)
- [A Survey of Inverse Reinforcement Learning:
Challenges, Methods and Progress
](https://arxiv.org/pdf/1806.06877v1.pdf)
- [A Survey of Knowledge Representation and Retrieval
for Learning in Service Robotics](https://arxiv.org/pdf/1807.02192v1.pdf)
- [Applications of Deep Reinforcement Learning in
Communications and Networking: A Survey](https://arxiv.org/pdf/1810.07862v1.pdf)
- [An Introduction to Deep Reinforcement Learning](https://arxiv.org/abs/1811.12560v2)

----------------------------------------

_**Maintainer**_

Gopala KR / @gopala-kr

----------------------------------------
