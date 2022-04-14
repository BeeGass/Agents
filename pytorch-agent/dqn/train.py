

def run_episode(environment, the_agent, replay_buffer, state, epsilon, gamma, loss_fn, optimizer, scheduler):
    step_count = 0
    cumulative_reward = 0
    cumulative_loss = 0
    while True:
        
        #0. either explore or exploit
        action = epsilon_greedy(environment=environment,
                                the_agent=the_agent,
                                state=state,
                                step_count=step_count,
                                epsilon=epsilon)
        
        #1. render environment only when flag is set to True
        # if render:
        #     env_screen = environment.le_env.render(mode = 'rgb_array')
        #     images = wandb.Image(env_screen)
        #     wandb.log({"frame": images, "step_count": step_count}, step=step_count)
        #     time.sleep(0.001)
        
        #2. pass action to environment
        (next_state, reward, done, _) = environment.le_env.step(action)
        
        #3. get s' back from environment and preprocess (s' -> preprocessed_s')
        preprocessed_next_state = preprocess_two(next_state)
        # preprocessed_next_state = preprocess(next_state)
        
        #4. add transition (s, a, s', r) to replay buffer
        replay_buffer.add_to_rb((state, action, preprocessed_next_state, np.sign(reward), done))
        
        #5. if replay buffer is full, sample mini batch and update model
        if len(replay_buffer.rb) > replay_buffer.mini_batch_size and not epsilon <= 0.000001 and step_count % 4 == 0:
            loss = train(replay_buffer, the_agent, loss_fn, optimizer, scheduler, gamma)
            cumulative_loss += loss
            
        
        cumulative_reward += reward
        
        #6. check max number of time steps has been reached or if game is complete
        if step_count >= environment.max_steps or done:
            step_count += 1
            return cumulative_reward, step_count, cumulative_loss
        
        state = preprocessed_next_state
        
        step_count += 1
        
        
def run():
    cfg = vanilla_config()
    with wandb.init(project="BeeGass-Agents", entity="beegass", config=cfg, monitor_gym=True):
        config = wandb.config
        
        # initialize gym environment
        environment = Gym_Env(env_name='ALE/Breakout-v5', max_steps=config.max_steps, max_episodes=config.max_episodes)
        #environment = Gym_Env(env_name='CartPole-v1', max_steps=config.max_steps, max_episodes=config.max_episodes)
        
        # initialize prediction network
        #pred_net = Deep_Q_Network(environment.le_env.action_space.n).to(device)
        pred_net = DQN(4, 4).to(device)
        target_net = DQN(4, 4).to(device)
        
        # initialize agent that contains both prediction network and target network
        the_agent = Agent(pred_model=pred_net, target_model=target_net)
        the_agent.copy_pred_to_target()
        
        # define loss function
        loss_fn = nn.SmoothL1Loss() #nn.HuberLoss(reduction='mean', delta=config.delta)
        
        # define optimizer
        optimizer = build_optimizer(model=the_agent.prediction_net, 
                                    optimizer_name='adam', 
                                    learning_rate=config.lr,
                                    weight_decay=config.weight_decay)
        
        # define scheduler
        scheduler = build_scheduler(optimizer, 
                                    sched_name='reduce_lr', 
                                    patience=5, 
                                    verbose=True)
        
        # initialize replay buffer
        replay_buffer = Replay_Buffer(capacity=config.replay_buffer_size, mini_batch_size=config.batch_size)
        
        epsilon = config.p_init
        episode_cumulative_reward = 0
        episode_cumulative_loss = 0 
        total_steps = 0
        for e in range(environment.max_episodes):
            
            # 0. get initial state, s_{0}, and preprocess it (s_{0} -> preprocessed_s)
            state = environment.le_env.reset(seed=42, return_info=False)
            
            # 0.1: preprocess state
            preprocessed_state = preprocess_two(state)
            #preprocessed_state = preprocess(state)
            
            # 1. iterate over steps in episode
            cumulative_reward, step_count, cumulative_loss = run_episode(environment=environment, 
                                                                         the_agent=the_agent,
                                                                         replay_buffer=replay_buffer,
                                                                         state=preprocessed_state, 
                                                                         epsilon=epsilon,
                                                                         gamma=config.gamma,
                                                                         loss_fn=loss_fn,
                                                                         optimizer=optimizer, 
                                                                         scheduler=scheduler)
            
            environment.le_env.close()
            
            # 3. decay epsilon
            # epsilon = config.decay_rate * epsilon
            epsilon = epsilon_decay(environment, e+1, config.p_init, config.p_end, config.epsilon_decay_rate)
            
            if e % config.target_freq == 0:
                the_agent.copy_pred_to_target()
            
            if not e+1 <= 10:
                episode_cumulative_reward += cumulative_reward
                episode_cumulative_loss += cumulative_loss
                total_steps += step_count
                wandb.log({"episode": e, "mean episodic reward": (episode_cumulative_reward/(e+1))}, step=e)
                wandb.log({"episode": e, "reward per episode": cumulative_reward}, step=e)
                wandb.log({"episode": e, "step_count": step_count}, step=e)
                wandb.log({"episode": e, "loss per episode": cumulative_loss}, step=e)
                wandb.log({"episode": e, "mean episodic loss": episode_cumulative_loss/(e+1)}, step=e)
                wandb.log({"episode": e, "epsilon": epsilon}, step=e)