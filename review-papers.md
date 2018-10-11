---------

- Temporal difference learning (1988)
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
- DQN (2014)

-----------
#### Surveys

 - Reinforcement Learning: A Survey, JAIR, 1996. [[arxiv]](https://arxiv.org/pdf/cs/9605103.pdf)
 - A Tutorial Survey of Reinforcement Learning, Sadhana, 1994. [[Paper]](http://www.cse.iitm.ac.in/~ravi/papers/keerthi.rl-survey.pdf)
 - Reinforcement Learning in Robotics, A Survey, IJRR, 2013. [[Paper]](http://www.ias.tu-darmstadt.de/uploads/Publications/Kober_IJRR_2013.pdf)
- A Brief Survey of Deep Reinforcement Learning 2017 [[arxiv](https://arxiv.org/abs/1708.05866)]
- A Survey of Deep Network Solutions for Learning Control in Robotics: From Reinforcement to Imitation. 2018 [[arxiv](https://arxiv.org/abs/1612.07139v4)]
- Universal Reinforcement Learning Algorithms: Survey and Experiments. 2017 [[arxiv](https://arxiv.org/abs/1705.10557v1)]
- [Bayesian Reinforcement Learning: A Survey. 2016](https://arxiv.org/pdf/1609.04436v1.pdf)
- [A Survey of Inverse Reinforcement Learning:
Challenges, Methods and Progress](https://arxiv.org/pdf/1806.06877v1.pdf)
- [Benchmarking Reinforcement Learning Algorithms
on Real-World Robots](https://arxiv.org/pdf/1809.07731v1.pdf)
- [DEEP REINFORCEMENT LEARNING: AN OVERVIEW](https://arxiv.org/pdf/1701.07274.pdf)
 
#### Foundational Papers

 - Steps toward Artificial Intelligence, Proceedings of the IRE, 1961. [[Paper]](http://staffweb.worc.ac.uk/DrC/Courses%202010-11/Comp%203104/Tutor%20Inputs/Session%209%20Prep/Reading%20material/Minsky60steps.pdf) (discusses issues in RL such as the "credit assignment problem")
 - An Adaptive Optimal Controller for Discrete-Time Markov Environments, Information and Control, 1977. [[Paper]](http://www.cs.waikato.ac.nz/~ihw/papers/77-IHW-AdaptiveController.pdf) (earliest publication on temporal-difference (TD) learning rule)
  
#### Methods

 - **Dynamic Programming (DP):**
 
   - Learning from Delayed Rewards, Ph.D. Thesis, Cambridge University, 1989. [[Thesis]](https://www.cs.rhul.ac.uk/home/chrisw/new_thesis.pdf)
   
 - **Monte Carlo:**
 
   - Monte Carlo Inversion and Reinforcement Learning, NIPS, 1994. [[Paper]](http://papers.nips.cc/paper/865-monte-carlo-matrix-inversion-and-reinforcement-learning.pdf)
   - Reinforcement Learning with Replacing Eligibility Traces, Machine Learning, 1996. [[Paper]](http://www-all.cs.umass.edu/pubs/1995_96/singh_s_ML96.pdf)
   
 - **Temporal-Difference:**
 
   - Learning to predict by the methods of temporal differences. Machine Learning 3: 9-44, 1988. [[Paper]](http://webdocs.cs.ualberta.ca/~sutton/papers/sutton-88-with-erratum.pdf)
   
 - **Q-Learning (Off-policy TD algorithm):**
 
   - Learning from Delayed Rewards, Cambridge, 1989. [[Thesis]](http://www.cs.rhul.ac.uk/home/chrisw/thesis.html)
   
 - **Sarsa (On-policy TD algorithm):**
 
   - On-line Q-learning using connectionist systems, Technical Report, Cambridge Univ., 1994. [[Report]](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&ved=0CDIQFjACahUKEwj2lMm5wZDIAhUHkg0KHa6kAVM&url=ftp%3A%2F%2Fmi.eng.cam.ac.uk%2Fpub%2Freports%2Fauto-pdf%2Frummery_tr166.pdf&usg=AFQjCNHz6IrgcaaO5lzC7t8oEIBY9epozg&sig2=sa-emPme1m5Jav7YmaXsNQ&cad=rja)
   - Generalization in Reinforcement Learning: Successful examples using sparse coding, NIPS, 1996. [[Paper]](http://webdocs.cs.ualberta.ca/~sutton/papers/sutton-96.pdf)
   
 - **R-Learning (learning of relative values)**
 
   -  A Reinforcement Learning Method for Maximizing Undiscounted Rewards, ICML, 1993. [[Paper-Google Scholar]](https://scholar.google.com/scholar?q=reinforcement+learning+method+for+maximizing+undiscounted+rewards&hl=en&as_sdt=0&as_vis=1&oi=scholart&sa=X&ved=0CBsQgQMwAGoVChMIho6p_MOQyAIVwh0eCh3XWAwM)
   
 - **Function Approximation methods (Least-Square Temporal Difference, Least-Square Policy Iteration)**
 
   - Linear Least-Squares Algorithms for Temporal Difference Learning, Machine Learning, 1996. [[Paper]](http://www-anw.cs.umass.edu/pubs/1995_96/bradtke_b_ML96.pdf)
   - Model-Free Least Squares Policy Iteration, NIPS, 2001. [[Paper]](http://www.cs.duke.edu/research/AI/LSPI/nips01.pdf) [[Code]](http://www.cs.duke.edu/research/AI/LSPI/)
   
 - **Policy Search / Policy Gradient**
 
   - Policy Gradient Methods for Reinforcement Learning with Function Approximation, NIPS, 1999. [[Paper]](http://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
   - Natural Actor-Critic, ECML, 2005. [[Paper]](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/NaturalActorCritic.pdf)
   - Policy Search for Motor Primitives in Robotics, NIPS, 2009. [[Paper]](http://papers.nips.cc/paper/3545-policy-search-for-motor-primitives-in-robotics.pdf)
   - Relative Entropy Policy Search, AAAI, 2010. [[Paper]](http://www.kyb.tue.mpg.de/fileadmin/user_upload/files/publications/attachments/AAAI-2010-Peters_6439%5b0%5d.pdf)
   - Path Integral Policy Improvement with Covariance Matrix Adaptation, ICML, 2012. [[Paper]](http://arxiv.org/pdf/1206.4621v1.pdf)
   - Policy Gradient Reinforcement Learning for Fast Quadrupedal Locomotion, ICRA, 2004. [[Paper]](http://www.cs.utexas.edu/~pstone/Papers/bib2html-links/icra04.pdf)
   -  PILCO: A Model-Based and Data-Efficient Approach to Policy Search, ICML, 2011. [[Paper]](http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf)
   - Learning Dynamic Arm Motions for Postural Recovery, Humanoids, 2011. [[Paper]](http://www-all.cs.umass.edu/pubs/2011/kuindersma_g_b_11.pdf)
   - Black-Box Data-efficient Policy Search for Robotics, IROS, 2017. [[Paper](https://arxiv.org/abs/1703.07261)]
   
 - **Hierarchical RL**
 
   - Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning, Artificial Intelligence, 1999. [[Paper]](https://webdocs.cs.ualberta.ca/~sutton/papers/SPS-aij.pdf)
   - Building Portable Options: Skill Transfer in Reinforcement Learning, IJCAI, 2007. [[Paper]](http://www-anw.cs.umass.edu/pubs/2007/konidaris_b_IJCAI07.pdf)
   
 - **Deep Learning + Reinforcement Learning (A sample of recent works on DL+RL)**
 
   - Human-level Control through Deep Reinforcement Learning, Nature, 2015. [[Paper]](http://www.readcube.com/articles/10.1038%2Fnature14236?shared_access_token=Lo_2hFdW4MuqEcF3CVBZm9RgN0jAjWel9jnR3ZoTv0P5kedCCNjz3FJ2FhQCgXkApOr3ZSsJAldp-tw3IWgTseRnLpAc9xQq-vTA2Z5Ji9lg16_WvCy4SaOgpK5XXA6ecqo8d8J7l4EJsdjwai53GqKt-7JuioG0r3iV67MQIro74l6IxvmcVNKBgOwiMGi8U0izJStLpmQp6Vmi_8Lw_A%3D%3D)
   - Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning, NIPS, 2014. [[Paper]](http://papers.nips.cc/paper/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning.pdf)
   - End-to-End Training of Deep Visuomotor Policies. ArXiv, 16 Oct 2015. [[ArXiv]](http://arxiv.org/pdf/1504.00702v3.pdf)
   - Prioritized Experience Replay, ArXiv, 18 Nov 2015. [[ArXiv]](http://arxiv.org/pdf/1511.05952v2.pdf)
   - Hado van Hasselt, Arthur Guez, David Silver, Deep Reinforcement Learning with Double Q-Learning, ArXiv, 22 Sep 2015. [[ArXiv]](http://arxiv.org/abs/1509.06461)
   - Asynchronous Methods for Deep Reinforcement Learning, ArXiv, 4 Feb 2016. [[ArXiv]](https://arxiv.org/abs/1602.01783)
   

#### Game Playing

Traditional Games
  - Backgammon - "TD-Gammon" game play using TD(λ) (Tesauro, ACM 1995) [[Paper]](http://www.bkgm.com/articles/tesauro/tdl.html)
  - Chess - "KnightCap" program using TD(λ) (Baxter, arXiv 1999) [[arXiv]](http://arxiv.org/pdf/cs/9901002v1.pdf)
  - Chess - Giraffe: Using deep reinforcement learning to play chess (Lai, arXiv 2015) [[arXiv]](http://arxiv.org/pdf/1509.01549v2.pdf)

Computer Games
  - Human-level Control through Deep Reinforcement Learning (Mnih, Nature 2015) [[Paper]](http://www.readcube.com/articles/10.1038%2Fnature14236?shared_access_token=Lo_2hFdW4MuqEcF3CVBZm9RgN0jAjWel9jnR3ZoTv0P5kedCCNjz3FJ2FhQCgXkApOr3ZSsJAldp-tw3IWgTseRnLpAc9xQq-vTA2Z5Ji9lg16_WvCy4SaOgpK5XXA6ecqo8d8J7l4EJsdjwai53GqKt-7JuioG0r3iV67MQIro74l6IxvmcVNKBgOwiMGi8U0izJStLpmQp6Vmi_8Lw_A%3D%3D) [[Code]](https://sites.google.com/a/deepmind.com/dqn/) [[Video]](https://www.youtube.com/watch?v=iqXKQf2BOSE)
  - [Flappy Bird Reinforcement Learning](https://github.com/SarvagyaVaish/FlappyBirdRL) [[Video]](https://www.youtube.com/watch?v=xM62SpKAZHU)
  - MarI/O - learning to play Mario with evolutionary reinforcement learning using artificial neural networks (Stanley, Evolutionary Computation 2002) [[Paper]](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) [[Video]](https://www.youtube.com/watch?v=qv6UVOQ0F44)

#### Robotics
  - Policy Gradient Reinforcement Learning for Fast Quadrupedal Locomotion (Kohl, ICRA 2004) [[Paper]](http://www.cs.utexas.edu/~pstone/Papers/bib2html-links/icra04.pdf)
  - Robot Motor SKill Coordination with EM-based Reinforcement Learning (Kormushev, IROS 2010) [[Paper]](http://kormushev.com/papers/Kormushev-IROS2010.pdf) [[Video]](https://www.youtube.com/watch?v=W_gxLKSsSIE)
  - Generalized Model Learning for Reinforcement Learning on a Humanoid Robot (Hester, ICRA 2010) [[Paper]](https://ccc.inaoep.mx/~mdprl/documentos/Hester_2010.pdf) [[Video]](https://www.youtube.com/watch?v=mRpX9DFCdwI&list=PL5nBAYUyJTrM48dViibyi68urttMlUv7e&index=12)
  - Autonomous Skill Acquisition on a Mobile Manipulator (Konidaris, AAAI 2011) [[Paper]](http://lis.csail.mit.edu/pubs/konidaris-aaai11b.pdf) [[Video]](https://www.youtube.com/watch?v=yUICAkSQTZY)
  - PILCO: A Model-Based and Data-Efficient Approach to Policy Search (Deisenroth, ICML 2011) [[Paper]](http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf)
  - Incremental Semantically Grounded Learning from Demonstration (Niekum, RSS 2013) [[Paper]](http://people.cs.umass.edu/~sniekum/pubs/NiekumRSS2013.pdf)
  - Efficient Reinforcement Learning for Robots using Informative Simulated Priors (Cutler, ICRA 2015) [[Paper]](http://markjcutler.com/papers/Cutler15_ICRA.pdf) [[Video]](https://www.youtube.com/watch?v=kKClFx6l1HY)
  - Robots that can adapt like animals (Cully, Nature 2015) [[Paper](https://arxiv.org/abs/1407.3501)] [[Video](https://www.youtube.com/watch?v=T-c17RKh3uE)] [[Code](https://github.com/resibots/cully_2015_nature)]
  - Black-Box Data-efficient Policy Search for Robotics (Chatzilygeroudis, IROS 2017) [[Paper](https://arxiv.org/abs/1703.07261)] [[Video](https://www.youtube.com/watch?v=kTEyYiIFGPM)] [[Code](https://github.com/resibots/blackdrops)]


#### Control
  - An Application of Reinforcement Learning to Aerobatic Helicopter Flight (Abbeel, NIPS 2006) [[Paper]](http://heli.stanford.edu/papers/nips06-aerobatichelicopter.pdf) [[Video]](https://www.youtube.com/watch?v=VCdxqn0fcnE)
  - Autonomous helicopter control using Reinforcement Learning Policy Search Methods (Bagnell, ICRA 2001) [[Paper]](http://repository.cmu.edu/cgi/viewcontent.cgi?article=1082&context=robotics)

#### Operations Research
  - Scaling Average-reward Reinforcement Learning for Product Delivery (Proper, AAAI 2004) [[Paper]](http://web.engr.oregonstate.edu/~proper/AAAI04SProper.pdf)
  - Cross Channel Optimized Marketing by Reinforcement Learning (Abe, KDD 2004) [[Paper]](http://www.research.ibm.com/people/n/nabe/kdd04AVAS.pdf)

#### Human Computer Interaction
  - Optimizing Dialogue Management with Reinforcement Learning: Experiments with the NJFun System (Singh, JAIR 2002) [[Paper]](http://web.eecs.umich.edu/~baveja/Papers/RLDSjair.pdf)
  
---------------

### References

  - [Mastering the game of Go with deep neural networks and tree search](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/papers/Mastering%20the%20game%20of%20Go%20with%20deep%20neural%20networks%20and%20tree%20search.md)
  - [Deep Successor Reinforcement Learning](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/papers/Deep%20Successor%20Reinforcement%20Learning.md)
  - [Action-Conditional Video Prediction using Deep Networks in Atari Games](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/papers/Action-Conditional%20Video%20Prediction%20using%20Deep%20Networks%20in%20Atari%20Games.md)
  - [Policy Distillation](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/papers/Policy%20Distillation.md)
  - [Learning Tetris Using the Noisy Cross-Entropy Method](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/papers/Learning%20Tetris%20Using%20the%20Noisy%20Cross-Entropy%20Method.md), with **code**
  - [Continuous Deep Q-Learning with Model-based Acceleration](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/papers/Continuous%20Deep%20Q-Learning%20with%20Model-based%20Acceleration.md)
  - [Value Iteration Networks](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/papers/Value%20Iteration%20Networks.md)
  - [Learning Modular Neural Network Policies for Multi-Task and Multi-Robot Transfer](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/papers/Learning%20Modular%20Neural%20Network%20Policies%20for%20Multi-Task%20and%20Multi-Robot%20Transfer.md) 
  - [Stochastic Neural Network For Hierarchical Reinforcement Learning](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/papers/Stochastic%20Neural%20Network%20For%20Hierarchical%20Reinforcement%20Learning.md)
  - [Noisy Networks for Exploration](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/papers/Noisy%20Networks%20for%20Exploration.md) 
  - [Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/papers/Improving%20Stochastic%20Policy%20Gradients%20in%20Continuous%20Control%20with%20Deep%20Reinforcement%20Learning%20using%20the%20Beta%20Distribution.md)
  - [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/papers/High-Dimensional%20Continuous%20Control%20Using%20Generalized%20Advantage%20Estimation.md) 
  - [Generalizing Skills with Semi-Supervised Reinforcement Learning](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/papers/Generalizing%20Skills%20with%20Semi-Supervised%20Reinforcement%20Learning.md)
  - [Unsupervised Perceptual Rewards for Imitation Learning](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/papers/Unsupervised%20Perceptual%20Rewards%20for%20Imitation%20Learning.md)
  - [Towards Deep Symbolic Reinforcement Learning](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/papers/Towards%20Deep%20Symbolic%20Reinforcement%20Learning.md)
- [Open Source](https://github.com/andrewliao11/Deep-Reinforcement-Learning-Survey/blob/master/Open-Source.md#open-source)
    
------------------------


- [Reinforcement Learning (RL)](https://hci-kdd.org/wordpress/wp-content/uploads/2016/07/3-REINFORCEMENT-Machine-Learning.pdf)
- [Simple Reinforcement Learning with Tensorflow Part 0-8](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)
- [Deep_reinforcement_learning_Course](https://github.com/simoninithomas/Deep_reinforcement_learning_Course)
- [Introduction to Various Reinforcement Learning Algorithms. Part I (Q-Learning, SARSA, DQN, DDPG)](https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287)


-----------------------
