from constants import TheaterType
from card import Card
from played_card import PlayedCard


class Theater:

    def __init__(self, initial_theater_type: TheaterType):
        self.current_type = initial_theater_type
        self.player_zones = {0: [], 1: []}  # Stores PlayedCard objects

    def add_card_to_zone(self, player_id: int, card_to_play: Card, play_face_up: bool):
        """Adds a new card (from hand or similar source) to the player's zone."""
        for existing_played_card in self.player_zones[player_id]:
            existing_played_card.is_covered = True
        new_played_card_instance = PlayedCard(card_to_play, play_face_up)
        new_played_card_instance.is_covered = False
        self.player_zones[player_id].append(new_played_card_instance)

    def add_played_card_instance_to_zone(self, player_id: int, moved_played_card: PlayedCard):
        """Adds an existing PlayedCard instance (e.g., a moved card) to the top of the zone."""
        for existing_played_card in self.player_zones[player_id]:
            existing_played_card.is_covered = True
        moved_played_card.is_covered = False # The moved card is now on top and uncovered
        self.player_zones[player_id].append(moved_played_card)

    def remove_card_from_zone_at_index(self, player_id: int, card_index_in_stack: int) -> PlayedCard | None:
        """
        Removes and returns a specific PlayedCard instance from the player's zone
        by its index in the stack. Updates is_covered status of a newly uncovered card.
        The index is from bottom (0) to top (len-1).
        """
        zone = self.player_zones[player_id]
        if 0 <= card_index_in_stack < len(zone):
            removed_card = zone.pop(card_index_in_stack)
            # If the stack is not empty after removal, ensure the new top card is uncovered.
            if zone:
                zone[-1].is_covered = False
            return removed_card
        return None

    def get_uncovered_card(self, player_id: int) -> PlayedCard | None:
        if self.player_zones[player_id]:
            return self.player_zones[player_id][-1]
        return None

    def clear_cards(self):
        self.player_zones = {0: [], 1: []}

    def __repr__(self):
        # This repr will not show dynamically calculated strengths with effects
        # For accurate strength, use env._get_player_strength_in_theater_with_effects
        p0_cards_repr = str(self.player_zones[0])
        p1_cards_repr = str(self.player_zones[1])
        return (f"Theater({self.current_type.name}): "
                f"P0_Zone: {p0_cards_repr}, "
                f"P1_Zone: {p1_cards_repr}")