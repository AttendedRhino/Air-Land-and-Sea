from card import Card


class PlayedCard:
    def __init__(self, card_ref: Card, is_face_up: bool):
        self.card_ref = card_ref
        self.is_face_up = is_face_up
        self.is_covered = False

    def get_current_strength(self, is_escalation_active_for_owner: bool = False,
                             is_covered_by_active_cover_fire: bool = False) -> int:
        """
        Calculates strength considering face-up/down status and relevant ongoing effects
        like Escalation or if this card is covered by an active Cover Fire.
        """
        if is_covered_by_active_cover_fire and self.is_covered: # Check this first
            return 4  # Covered by active Cover Fire

        if self.is_face_up:
            return self.card_ref.strength
        else: # Card is face-down
            return 4 if is_escalation_active_for_owner else 2

    def __repr__(self):
        # For __repr__, we might want to show its potential strength if Escalation is active,
        # but that requires passing more state. For now, keep it simple or pass a dummy for escalation.
        # This representation does not reflect ongoing effects for simplicity here.
        base_strength_repr = self.card_ref.strength if self.is_face_up else 2
        status = "FU" if self.is_face_up else "FD"
        covered_status = "C" if self.is_covered else "UC"
        # Note: The @strength shown here won't reflect Escalation/Cover Fire in this basic repr
        return f"({self.card_ref.name}@{base_strength_repr}|{status}|{covered_status})"