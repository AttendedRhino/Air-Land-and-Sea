from constants import HAND_SIZE
from card import Card  # Assuming Card is in card.py


class Player:
    def __init__(self, player_id: int):
        self.id = player_id
        self.hand = []  # List of Card objects
        # Changed battle_vps to battle_reward_outcome and initialized to 0.0 for float consistency
        self.battle_reward_outcome = 0.0
        self.war_vps = 0  # Total victory points for the game/war

    def draw_to_hand(self, card: Card):
        if len(self.hand) < HAND_SIZE:
            self.hand.append(card)

    def play_card_from_hand(self, card_id_to_play: int) -> Card | None:
        card_found = None
        for card_in_hand in self.hand:
            if card_in_hand.id == card_id_to_play:
                card_found = card_in_hand
                break
        if card_found:
            self.hand.remove(card_found)
            return card_found
        return None

    # Renamed this method for clarity
    def clear_hand_and_battle_outcome(self):
        self.hand = []
        self.battle_reward_outcome = 0.0 # Reset to 0.0

    def __repr__(self):
        hand_str = ", ".join([c.name for c in self.hand])
        # Updated to show new attribute name
        return f"Player {self.id} (War VPs: {self.war_vps}, Last Battle Outcome: {self.battle_reward_outcome}) Hand: [{hand_str}]"