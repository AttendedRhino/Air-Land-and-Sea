import numpy as np
import random
from air_land_sea_env import AirLandSeaBaseEnv
from constants import GamePhase # etc.

class BaseOpponentPolicy:
    def __init__(self, opponent_player_idx):
        self.player_idx = opponent_player_idx

    def select_action(self, state, env: 'AirLandSeaBaseEnv'):
        """
        Selects an action for the opponent.
        'state' is the current observation for the opponent.
        'env' is the game environment instance.
        'self.player_idx' indicates which player this policy controls.
        """
        # Ensure it's this policy's turn before getting the mask
        if env.current_player_idx != self.player_idx:
            raise ValueError(f"Opponent policy P{self.player_idx} called when it's P{env.current_player_idx}'s turn.")
        
        action_mask = env._get_action_mask() # Opponent uses the env to get its own legal moves
        legal_actions = np.where(action_mask)[0]
        
        if len(legal_actions) == 0:
            # This should ideally be handled by the environment ensuring a valid mask
            # (e.g., withdraw is always an option in NORMAL_PLAY)
            print(f"Warning: Opponent P{self.player_idx} has no legal actions from mask. Phase: {env.game_phase.name}")
            if env.action_space.n > 0:
                return random.randrange(env.action_space.n) # Fallback, might be illegal by mask
            raise ValueError("Opponent has no legal actions and action space is zero.")
        
        return self._select_from_legal(legal_actions, state, env)

    def _select_from_legal(self, legal_actions, state, env):
        raise NotImplementedError

class RandomOpponentPolicy(BaseOpponentPolicy):
    def _select_from_legal(self, legal_actions, state, env):
        """Selects a random action from the legal actions."""
        return random.choice(legal_actions)

# Example for a future Heuristic Opponent (you'd fill in the logic)
class HeuristicOpponentPolicy(BaseOpponentPolicy):
    def _select_from_legal(self, legal_actions, state, env):
        # Implement heuristic logic here based on 'state' and 'env'
        # For now, let's make it also play randomly as a placeholder
        print(f"HeuristicOpponent P{self.player_idx} choosing (currently random)...")
        return random.choice(legal_actions)

# You could also have a DQNOpponentPolicy that loads a pre-trained DQN model
# class DQNOpponentPolicy(BaseOpponentPolicy):
#     def __init__(self, opponent_player_idx, model_path, n_observations, n_actions):
#         super().__init__(opponent_player_idx)
#         self.dqn_model = DQN(n_observations, n_actions).to(device)
#         self.dqn_model.load_state_dict(torch.load(model_path, map_location=device))
#         self.dqn_model.eval()
#
#     def _select_from_legal(self, legal_actions, state, env):
#         with torch.no_grad():
#             if not isinstance(state, torch.Tensor):
#                 state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#             else:
#                 state_tensor = state.to(device)
#             if state_tensor.dim() == 1:
#                 state_tensor = state_tensor.unsqueeze(0)
#
#             q_values = self.dqn_model(state_tensor)
#             masked_q_values = q_values.clone()
#             for i in range(env.action_space.n):
#                 if i not in legal_actions:
#                     masked_q_values[0, i] = -float('inf')
#             return masked_q_values.argmax().item()