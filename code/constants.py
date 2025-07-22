from enum import Enum, auto


# --- Enums and Constants ---

class GamePhase(Enum):
    NORMAL_PLAY = auto()
    RESOLVING_MANEUVER = auto()
    RESOLVING_AMBUSH = auto()
    RESOLVING_DISRUPT_OPPONENT_CHOICE = auto() # ADDED
    RESOLVING_DISRUPT_SELF_CHOICE = auto()   # ADDED
    RESOLVING_TRANSPORT_SELECT_CARD = auto()    # ADDED
    RESOLVING_TRANSPORT_SELECT_DEST = auto() # ADDED
    # Add other effect phases here later, e.g., RESOLVING_AMBUSH

class TheaterType(Enum):
    LAND = auto()
    AIR = auto()
    SEA = auto()


class CardType(Enum):  # Maps to TheaterType for matching suits
    LAND = auto()
    AIR = auto()
    SEA = auto()


NUM_THEATERS = 3
HAND_SIZE = 6
TOTAL_CARDS_IN_DECK = 18


# VICTORY_POINTS_TO_WIN_WAR = 12 # For later multi-battle game structure