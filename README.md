<h1 align="center">
  <b>RL Agents</b><br> 
  <b>Jax | Flux | PyTorch</b><br> 
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.8-ff69b4.svg" /></a>
       <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.10-2BAF2B.svg" /></a>
       <a href= "https://fluxml.ai/">
        <img src="https://img.shields.io/badge/Flux-v0.12.8-red" /></a>
       <a href= "https://github.com/google/jax">
        <img src="https://img.shields.io/badge/Jax-v0.1.75-yellow" /></a>
       <a href= "https://github.com/BeeGass/Agents/blob/master/LICENSE">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
         <a href= "http://twitter.com/intent/tweet?text=Readable-Agents:%20A%20Collection%20Of%20RL%20Agents%20Written%20In%20PyTorch%20And%20Jax%3A&url=https://github.com/BeeGass/Agents">
        <img src="https://img.shields.io/twitter/url/https/shields.io.svg?style=social" /></a>

</p>

A collection of Reinforcement Learning (RL) Methods I have implemented in [jax](https://github.com/google/jax)/[flax](https://github.com/google/flax), [flux](https://fluxml.ai/) and [pytorch](https://pytorch.org/) with particular effort put into readability and reproducibility. 

## Python 
### Requirements For Jax
- Python >= 3.8
- jax

#### Installation
```
$ git clone https://github.com/BeeGass/Agents.git
```

#### Usage
```
$ cd Agents/agents-jax
$ python main.py 
```

### Requirements For PyTorch
- PyTorch >= 1.10

#### Usage
```
$ cd Agents/agents-pytorch
$ python main.py 
```

## Julia
### Requirements For Flux
- TODO
- TODO

#### Usage
```
$ cd Agents/agents-flux
$ # TBA 
```
--- 

**Config File Template**
```yaml
TBA
```

**Weights And Biases Integration**
```
TBA
```

----
<h2 align="center">
  <b>Preliminary RL Implementations</b><br>
</h2>


| Model                                        | NumPy/Vanilla | Jax/Flax |  Flux   | Config  | Paper                                                                                        |
|:-------------------------------------------- |:-------------:|:--------:|:-------:|:-------:|:-------------------------------------------------------------------------------------------- |
| Policy Evaluation                            |    &#9745;    | &#9744;  | &#9744; | &#9744; | [DS595-RL-Projects](https://github.com/yingxue-zhang/DS595-RL-Projects/tree/master/Project1) |
| Policy Improvement                           |    &#9745;    | &#9744;  | &#9744; | &#9744; | [DS595-RL-Projects](https://github.com/yingxue-zhang/DS595-RL-Projects/tree/master/Project1) |
| Policy Iteration                             |    &#9745;    | &#9744;  | &#9744; | &#9744; | [DS595-RL-Projects](https://github.com/yingxue-zhang/DS595-RL-Projects/tree/master/Project1) |
| Value Iteration                              |    &#9745;    | &#9744;  | &#9744; | &#9744; | [DS595-RL-Projects](https://github.com/yingxue-zhang/DS595-RL-Projects/tree/master/Project1) |
| On-policy first visit Monte-Carlo prediction |    &#9745;    | &#9744;  | &#9744; | &#9744; | [DS595-RL-Projects](https://github.com/yingxue-zhang/DS595-RL-Projects/tree/master/Project2) |
| On-policy first visit Monte-Carlo control    |    &#9745;    | &#9744;  | &#9744; | &#9744; | [DS595-RL-Projects](https://github.com/yingxue-zhang/DS595-RL-Projects/tree/master/Project2) |
| Sarsa (on-policy TD control)                 |    &#9745;    | &#9744;  | &#9744; | &#9744; | [DS595-RL-Projects](https://github.com/yingxue-zhang/DS595-RL-Projects/tree/master/Project2) |
| Q-learing (off-policy TD control)            |    &#9745;    | &#9744;  | &#9744; | &#9744; | [DS595-RL-Projects](https://github.com/yingxue-zhang/DS595-RL-Projects/tree/master/Project2) |
 

----
<h2 align="center">
  <b> Off-Policy Results</b><br>
</h2>



| Model       | PyTorch | Jax/Flax |  Flux   | Config  | Paper                                      |
|:----------- |:-------:|:--------:|:-------:|:-------:|:------------------------------------------ |
| DQN         | &#9744; | &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/1312.5602)    |
| DDPG        | &#9744; | &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/1509.02971)   |
| DRQN        | &#9744; | &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/1507.06527)   |
| Dueling-DQN | &#9744; | &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/1511.06581)   |
| Double-DQN  | &#9744; | &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/1509.06461)   |
| PER         | &#9744; | &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/1511.05952)   |
| Rainbow     | &#9744; | &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/1710.02298v1) |


<h2 align="center">
  <b>Policy Results</b><br>
</h2>


| Model | PyTorch | Jax/Flax |  Flux   | Config  | Paper                                    |
|:----- |:-------:|:--------:|:-------:|:-------:|:---------------------------------------- |
| PPO   | &#9744; | &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/1312.6114)  |
| TRPO  | &#9744; | &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/1502.05477) |
| SAC   | &#9744; | &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/1801.01290) |
| A2C   | &#9744; | &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/1602.01783) |
| A3C   | &#9744; | &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/1602.01783) |
| TD3   | &#9744; | &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/1802.09477) |


<h2 align="center">
  <b>Fun Stuff</b><br>
</h2>

| Model               | PyTorch | Jax/Flax |  Flux   | Config  | Paper                                        |
|:------------------- |:-------:|:--------:|:-------:|:-------:|:-------------------------------------------- |
| World Models        | &#9744; | &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/pdf/1803.10122.pdf) |
| Dream to Control    | &#9744; | &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/1912.01603)     |
| Dream to Control v2 | &#9744; | &#9744;  | &#9744; | &#9744; | [Link](https://arxiv.org/abs/2010.02193)     |

### Citation
```bib
@software{Gass_Agents_2021,
  author = {Gass, B.A., Gass, B.A.},
  doi = {10.5281/zenodo.1234},
  month = {12},
  title = {{Agents}},
  url = {https://github.com/BeeGass/Agents},
  version = {1.0.0},
  year = {2021}
}
```
