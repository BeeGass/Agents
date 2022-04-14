class Agent(nn.Module):
    def __init__(self, pred_model, target_model):
        super(Agent, self).__init__()
        self.prediction_net = pred_model
        self.target_net = target_model 
        
    def agent_policy(self, state, pred_model=True, grad=False):
        q_val = None
        state = state.to(device)
        if pred_model:
            if grad:
                q_val = self.agent(state)
            else:
                with torch.no_grad():
                    q_val = self.agent(state)
        else:
            with torch.no_grad():
                q_val = self.target(state)
        return q_val
    
    def copy_pred_to_target(self):
        self.target_net = load_state_dict(self.prediction_net.state_dict())
        self.target_net.eval()
