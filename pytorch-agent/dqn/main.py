

def main():
    with wandb.init(project="BeeGass-Agents", entity="beegass", config=cfg, monitor_gym=True):
        config = wandb.config

        # initialize gym environment
        environment = Gym_Env(env_name='ALE/Breakout-v5', max_steps=config.max_steps, max_episodes=config.max_episodes)
        
        # initialize prediction network
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



if __name__ == "__main__":
    main() 
