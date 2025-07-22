from constants import CardType


class Card:
    def __init__(self, card_id: int, name: str, strength: int, card_type: CardType, ability_text: str = ""):
        self.id = card_id
        self.name = name
        self.strength = strength
        self.card_type = card_type  # Its 'suit'
        self.ability_text = ability_text
        # In the future, we can add:
        # self.effect_type = None # e.g., "INSTANT", "ONGOING"
        # self.effect_details = {} # Parameters for the effect

    def __repr__(self):
        return f"{self.name}(S:{self.strength}, T:{self.card_type.name})"
