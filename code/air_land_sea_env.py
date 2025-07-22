import numpy as np
import random
from constants import TheaterType, CardType, NUM_THEATERS, HAND_SIZE, TOTAL_CARDS_IN_DECK, GamePhase
from card import Card
from player import Player
from theater import Theater


# --- Core Game Engine ---
class AirLandSeaBaseEnv:
    def __init__(self):
        self._master_deck = AirLandSeaBaseEnv._initialize_master_deck()
        self.battle_deck = []

        self.theaters = [
            Theater(TheaterType.AIR),
            Theater(TheaterType.LAND),
            Theater(TheaterType.SEA)
        ]
        self.physical_theater_order = [
            TheaterType.AIR,
            TheaterType.LAND,
            TheaterType.SEA
        ]
        self.players = [Player(0), Player(1)]

        self.current_player_idx = 0
        self.battle_first_player_idx = 0
        self.battle_turn_counter = 0
        self.max_plays_per_battle = HAND_SIZE * 2

        self.num_actions = (HAND_SIZE * NUM_THEATERS * 2) + 1
        self.withdraw_action_id = self.num_actions - 1

        self.game_phase = GamePhase.NORMAL_PLAY
        self.effect_resolution_data = {}

        # States for AIR DROP
        self.air_drop_primed_for_player = {0: False, 1: False}
        self.air_drop_active_this_turn = {0: False, 1: False}

        dummy_obs, _ = self.reset()
        self.observation_shape = dummy_obs.shape

    @staticmethod
    def _initialize_master_deck() -> list[Card]:
        """Creates the 18 unique cards for the game."""
        # TODO: Temporarily switched REINFORCE and REDEPLOY cards to AMBUSH type for simplicity. Maybe implement later.
        deck = [
            # Card(0, "LAND_1_REINFORCE", 1, CardType.LAND, "REINFORCE"),
            Card(0, "LAND_1_AMBUSH", 1, CardType.LAND, "AMBUSH"),
            Card(1, "LAND_2_AMBUSH", 2, CardType.LAND, "AMBUSH"),
            Card(2, "LAND_3_MANEUVER_L", 3, CardType.LAND, "MANEUVER"),
            Card(3, "LAND_5_DISRUPT", 5, CardType.LAND, "DISRUPT"),
            Card(4, "LAND_6_NO_ABILITY", 6, CardType.LAND, "[No ability]"),
            Card(5, "AIR_1_SUPPORT", 1, CardType.AIR, "SUPPORT"),
            Card(6, "AIR_2_AIR_DROP", 2, CardType.AIR, "AIR_DROP"),
            Card(7, "AIR_3_MANEUVER_A", 3, CardType.AIR, "MANEUVER"),
            Card(8, "AIR_4_AERODROME", 4, CardType.AIR, "AERODROME"),
            Card(9, "AIR_5_CONTAINMENT", 5, CardType.AIR, "CONTAINMENT"),
            Card(10, "AIR_6_NO_ABILITY", 6, CardType.AIR, "[No ability]"),
            Card(11, "SEA_1_TRANSPORT", 1, CardType.SEA, "TRANSPORT"),
            Card(12, "SEA_2_ESCALATION", 2, CardType.SEA, "ESCALATION"),
            Card(13, "SEA_3_MANEUVER_S", 3, CardType.SEA, "MANEUVER"),
            # Card(14, "SEA_4_REDEPLOY", 4, CardType.SEA, "REDEPLOY"),
            Card(14, "SEA_4_AMBUSH", 4, CardType.SEA, "AMBUSH"),
            Card(15, "SEA_5_BLOCKADE", 5, CardType.SEA, "BLOCKADE"),
            Card(16, "SEA_4_COVER_FIRE", 4, CardType.SEA, "COVER_FIRE"),
            Card(17, "SEA_6_NO_ABILITY", 6, CardType.SEA, "[No ability]"),
        ]
        assert len(deck) == TOTAL_CARDS_IN_DECK, "Master deck initialization error"
        return deck

    # --- Helper methods to check for active ONGOING effects ---
    def _is_card_effect_active_for_player(self, player_id: int, card_id_to_check: int) -> bool:
        """Checks if a specific card_id providing an ongoing effect is active for the player."""
        for theater in self.theaters:
            uncovered_card = theater.get_uncovered_card(player_id)
            if uncovered_card and uncovered_card.is_face_up and uncovered_card.card_ref.id == card_id_to_check:
                return True
        return False

    def _is_aerodrome_active_for_player(self, player_id: int) -> bool:
        return self._is_card_effect_active_for_player(player_id, 8)  # AERODROME ID = 8

    def _is_escalation_active_for_player(self, player_id: int) -> bool:
        return self._is_card_effect_active_for_player(player_id, 12)  # ESCALATION ID = 12

    def _is_containment_active_globally(self) -> bool:
        """Checks if CONTAINMENT (ID 9) is active by EITHER player."""
        for player_id in [0, 1]:
            if self._is_card_effect_active_for_player(player_id, 9):  # CONTAINMENT ID = 9
                return True
        return False

    def _get_active_blockades_affecting_theater(self, target_theater_idx: int, playing_player_id: int) -> list[Card]:
        """
        Finds active BLOCKADE cards of the opponent that affect plays into target_theater_idx.
        BLOCKADE (ID 15) affects plays into an *adjacent* theater.
        """
        active_blockades = []
        opponent_id = 1 - playing_player_id
        adjacent_to_target = self._get_adjacent_theater_indices(target_theater_idx)

        for blockade_theater_idx in range(NUM_THEATERS):
            # Check if opponent has BLOCKADE in this blockade_theater_idx
            blockade_card = self.theaters[blockade_theater_idx].get_uncovered_card(opponent_id)
            if blockade_card and blockade_card.is_face_up and blockade_card.card_ref.id == 15: # BLOCKADE ID = 15
                # Check if where BLOCKADE is (blockade_theater_idx) is adjacent to where card is being played (target_theater_idx)
                if target_theater_idx in self._get_adjacent_theater_indices(blockade_theater_idx):
                    active_blockades.append(blockade_card.card_ref)
        return active_blockades

    def _get_total_support_bonus_for_theater(self, player_id: int, current_theater_idx: int) -> int:
        """Calculates total +3 bonuses from SUPPORT cards in adjacent theaters."""
        bonus = 0
        adjacent_indices = self._get_adjacent_theater_indices(current_theater_idx)
        for adj_idx in adjacent_indices:
            support_card = self.theaters[adj_idx].get_uncovered_card(player_id)
            if support_card and support_card.is_face_up and support_card.card_ref.id == 5: # SUPPORT ID = 5
                bonus += 3
        return bonus

    def _setup_new_battle(self):
        self.battle_deck = self._master_deck[:]
        random.shuffle(self.battle_deck)
        for player in self.players:
            player.clear_hand_and_battle_outcome()
        for _ in range(HAND_SIZE):
            if self.battle_deck:
                self.players[0].draw_to_hand(self.battle_deck.pop())
            if self.battle_deck:
                self.players[1].draw_to_hand(self.battle_deck.pop())
        self.physical_theater_order = [TheaterType.AIR, TheaterType.LAND, TheaterType.SEA]
        random.shuffle(self.physical_theater_order)
        for i in range(NUM_THEATERS):
            self.theaters[i].current_type = self.physical_theater_order[i]
            self.theaters[i].clear_cards()
        self.battle_first_player_idx = 0
        self.current_player_idx = self.battle_first_player_idx
        self.battle_turn_counter = 0
        self.game_phase = GamePhase.NORMAL_PLAY
        self.effect_resolution_data = {}

        # Reset AIR DROP states
        self.air_drop_primed_for_player = {0: False, 1: False}
        self.air_drop_active_this_turn = {0: False, 1: False}

    def reset(self) -> tuple[np.ndarray, dict]:
        self._setup_new_battle()

        # --- Activate AIR DROP if primed for the starting player (P0) ---
        # This needs to happen *before* _get_observation and _get_info for the reset state
        if self.air_drop_primed_for_player[self.current_player_idx]:
            self.air_drop_active_this_turn[self.current_player_idx] = True
            self.air_drop_primed_for_player[self.current_player_idx] = False
        else: # Ensure it's off if not primed
            self.air_drop_active_this_turn[self.current_player_idx] = False

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _decode_action(self, action_id: int) -> tuple[int | None, int | None, bool | None, bool]:
        if action_id == self.withdraw_action_id:
            return None, None, None, True
        if not (0 <= action_id < self.withdraw_action_id):
            return None, None, None, False
        actions_per_card_slot = NUM_THEATERS * 2
        card_slot_idx = action_id // actions_per_card_slot
        remainder = action_id % actions_per_card_slot
        theater_idx = remainder // 2
        is_face_up = (remainder % 2 == 0)
        return card_slot_idx, theater_idx, is_face_up, False

    def _is_play_action_legal(self, player: Player, card_slot_idx: int, card_to_play: Card,
                              target_theater_idx: int, is_face_up: bool) -> bool:
        if is_face_up:
            target_type = self.theaters[target_theater_idx].current_type
            card_matches_theater = (card_to_play.card_type.name == target_type.name)
            if not card_matches_theater:
                # Check for AERODROME effect
                if self._is_aerodrome_active_for_player(player.id) and card_to_play.strength <= 3:
                    return True
                # NEW: Check for AIR DROP active this turn for this player
                elif self.air_drop_active_this_turn[player.id]:
                    return True # AIR DROP allows any non-matching play FU
                return False
        return True

    def _get_adjacent_theater_indices(self, theater_idx: int) -> list[int]:
        adj = []
        if theater_idx == 0:
            adj = [1]
        elif theater_idx == 1:
            adj = [0, 2]
        elif theater_idx == 2:
            adj = [1]
        return [i for i in adj if 0 <= i < NUM_THEATERS]

    def _flip_card_in_theater(self, target_player_id: int, target_theater_idx: int):
        # The flip itself. Ongoing effects become active/inactive due to their is_face_up status changing,
        # which is then picked up by query methods like get_player_strength_in_theater_with_effects.
        target_theater = self.theaters[target_theater_idx]
        top_card_to_flip = target_theater.get_uncovered_card(target_player_id)
        if top_card_to_flip:
            top_card_to_flip.is_face_up = not top_card_to_flip.is_face_up
            # TODO: No chained instant effects for now. Ongoing effects apply passively.

    def _get_flippable_targets_for_player(self, player_id_to_target: int) -> list[tuple[int, int, int, str]]:
        """Helper to get all top uncovered cards for a specific player."""
        targets = []
        map_id_counter = 0
        for t_idx in range(NUM_THEATERS):
            card = self.theaters[t_idx].get_uncovered_card(player_id_to_target)
            if card and map_id_counter < (self.num_actions -1): # Ensure map_id is in play range
                targets.append((map_id_counter, player_id_to_target, t_idx, card.card_ref.name))
                map_id_counter += 1
        return targets

    def get_readable_legal_moves(self) -> list[tuple[int, str]]:
        action_mask = self._get_action_mask()
        legal_action_ids = np.where(action_mask)[0]
        readable_moves = []
        current_player = self.players[self.current_player_idx]

        if self.game_phase == GamePhase.NORMAL_PLAY:
            for action_id in legal_action_ids:
                description = ""
                if action_id == self.withdraw_action_id:
                    description = "Withdraw"
                else:
                    card_slot_idx, theater_idx, is_face_up, _ = self._decode_action(action_id)
                    if (card_slot_idx is not None
                        and 0 <= card_slot_idx < len(current_player.hand)
                        and theater_idx is not None
                        and is_face_up is not None
                    ):
                        card_name = current_player.hand[card_slot_idx].name
                        theater_name = self.theaters[theater_idx].current_type.name
                        orientation = "Face-Up" if is_face_up else "Face-Down"
                        description = f"Play '{card_name}' (from slot {card_slot_idx}) to {theater_name} (Pos {theater_idx}) {orientation}"

                        # Optionally indicate if play is due to Air Drop
                        if (is_face_up
                            and card_name != "[No ability]"
                            and self.air_drop_active_this_turn[current_player.id]
                            and current_player.hand[card_slot_idx].card_type.name != self.theaters[theater_idx].current_type.name
                        ):
                            description += " (AIR DROP)"
                    else:
                        description = f"Internal Error: Play Action (ID: {action_id}) with invalid decode"
                readable_moves.append((int(action_id), description))

        elif self.game_phase == GamePhase.RESOLVING_MANEUVER or self.game_phase == GamePhase.RESOLVING_AMBUSH:
            # ... (same as before) ...
            effect_name = "MANEUVER" if self.game_phase == GamePhase.RESOLVING_MANEUVER else "AMBUSH"
            if 'possible_targets' in self.effect_resolution_data:
                for mapped_action_id, target_player_id, target_theater_idx, card_name in self.effect_resolution_data['possible_targets']:
                    if 0 <= mapped_action_id < len(action_mask) and action_mask[mapped_action_id]:
                        target_player_str = f"Player {target_player_id}'s"
                        theater_name = self.theaters[target_theater_idx].current_type.name
                        description = f"{effect_name}: Flip {target_player_str} top card ('{card_name}') in {theater_name} (Pos {target_theater_idx})"
                        readable_moves.append((mapped_action_id, description))
            if not readable_moves and self.effect_resolution_data.get('possible_targets', []) == []:
                readable_moves.append((0, f"{effect_name}: No valid targets, effect fizzles (auto-continue)"))

        elif self.game_phase == GamePhase.RESOLVING_DISRUPT_OPPONENT_CHOICE:
            if 'possible_opponent_targets' in self.effect_resolution_data:
                for mapped_action_id, target_player_id, target_theater_idx, card_name in self.effect_resolution_data['possible_opponent_targets']:
                    if 0 <= mapped_action_id < len(action_mask) and action_mask[mapped_action_id]:
                        theater_name = self.theaters[target_theater_idx].current_type.name
                        description = f"DISRUPT (Opponent P{target_player_id} chooses): Flip your ('{card_name}') in {theater_name} (Pos {target_theater_idx})"
                        readable_moves.append((mapped_action_id, description))
            if not readable_moves and self.effect_resolution_data.get('possible_opponent_targets', []) == []:
                readable_moves.append((0, "DISRUPT: No valid targets for opponent to flip (auto-continue)"))

        elif self.game_phase == GamePhase.RESOLVING_DISRUPT_SELF_CHOICE:
            if 'possible_self_targets' in self.effect_resolution_data:
                for mapped_action_id, target_player_id, target_theater_idx, card_name in self.effect_resolution_data['possible_self_targets']:
                    if 0 <= mapped_action_id < len(action_mask) and action_mask[mapped_action_id]:
                        theater_name = self.theaters[target_theater_idx].current_type.name
                        description = f"DISRUPT (Self P{target_player_id} chooses): Flip your ('{card_name}') in {theater_name} (Pos {target_theater_idx})"
                        readable_moves.append((mapped_action_id, description))
            if not readable_moves and self.effect_resolution_data.get('possible_self_targets', []) == []:
                readable_moves.append((0, "DISRUPT: No valid targets for self to flip (auto-continue)"))

        elif self.game_phase == GamePhase.RESOLVING_TRANSPORT_SELECT_CARD:
            if 'possible_transport_sources' in self.effect_resolution_data:
                for mapped_action_id, src_p_id, src_t_idx, src_stack_idx, card_name, _card_obj_id in self.effect_resolution_data['possible_transport_sources']:
                    if 0 <= mapped_action_id < len(action_mask) and action_mask[mapped_action_id]:
                        theater_name = self.theaters[src_t_idx].current_type.name
                        description = f"TRANSPORT: Select your card '{card_name}' (idx {src_stack_idx}) in {theater_name} (Pos {src_t_idx}) to move"
                        readable_moves.append((mapped_action_id, description))
            if not readable_moves and self.effect_resolution_data.get('possible_transport_sources', []) == []:
                readable_moves.append((0, "TRANSPORT: No cards to move (auto-continue)"))

        elif self.game_phase == GamePhase.RESOLVING_TRANSPORT_SELECT_DEST:
            card_to_move_name = self.effect_resolution_data.get('card_to_transport_details', {}).get('card_name', 'Unknown Card')
            if 'possible_transport_destinations' in self.effect_resolution_data:
                for mapped_action_id, dest_t_idx in self.effect_resolution_data['possible_transport_destinations']:
                    if 0 <= mapped_action_id < len(action_mask) and action_mask[mapped_action_id]:
                        theater_name = self.theaters[dest_t_idx].current_type.name
                        description = f"TRANSPORT '{card_to_move_name}': Move to {theater_name} (Pos {dest_t_idx})"
                        readable_moves.append((mapped_action_id, description))
            if not readable_moves and self.effect_resolution_data.get('possible_transport_destinations', []) == []:
                readable_moves.append((0, f"TRANSPORT '{card_to_move_name}': No valid destinations (auto-continue)"))

        return readable_moves

    def _get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=bool)
        current_player_obj = self.players[self.current_player_idx]  # Player whose turn it is to make a choice

        if self.game_phase == GamePhase.NORMAL_PLAY:
            mask[self.withdraw_action_id] = True

            if not current_player_obj.hand:
                return mask

            actions_per_card_slot = NUM_THEATERS * 2
            for hand_slot_idx in range(HAND_SIZE):
                if hand_slot_idx < len(current_player_obj.hand):
                    card_to_play = current_player_obj.hand[hand_slot_idx]
                    for theater_idx in range(NUM_THEATERS):
                        for face_up_bool_val in [True, False]:
                            orientation_offset = 0 if face_up_bool_val else 1
                            theater_orientation_part = theater_idx * 2 + orientation_offset
                            action_id = hand_slot_idx * actions_per_card_slot + theater_orientation_part
                            if action_id < self.withdraw_action_id:
                                if self._is_play_action_legal(current_player_obj, hand_slot_idx, card_to_play,
                                                              theater_idx, face_up_bool_val):
                                    mask[action_id] = True

        elif self.game_phase == GamePhase.RESOLVING_MANEUVER or self.game_phase == GamePhase.RESOLVING_AMBUSH:
            # ... (same as before) ...
            if 'possible_targets' in self.effect_resolution_data:
                for mapped_action_id, _, _, _ in self.effect_resolution_data['possible_targets']:
                    if 0 <= mapped_action_id < self.num_actions:
                        mask[mapped_action_id] = True

        elif self.game_phase == GamePhase.RESOLVING_DISRUPT_OPPONENT_CHOICE:
            # Mask for opponent's choices (current_player_idx is the opponent)
            if 'possible_opponent_targets' in self.effect_resolution_data:
                for mapped_action_id, _, _, _ in self.effect_resolution_data['possible_opponent_targets']:
                    if 0 <= mapped_action_id < self.num_actions:
                        mask[mapped_action_id] = True

        elif self.game_phase == GamePhase.RESOLVING_DISRUPT_SELF_CHOICE:
            # Mask for disrupt player's own choices (current_player_idx is the original disrupt player)
            if 'possible_self_targets' in self.effect_resolution_data:
                for mapped_action_id, _, _, _ in self.effect_resolution_data['possible_self_targets']:
                    if 0 <= mapped_action_id < self.num_actions:
                        mask[mapped_action_id] = True

        elif self.game_phase == GamePhase.RESOLVING_TRANSPORT_SELECT_CARD:
            if 'possible_transport_sources' in self.effect_resolution_data:
                for mapped_action_id, _, _, _, _, _ in self.effect_resolution_data['possible_transport_sources']:
                    if 0 <= mapped_action_id < self.num_actions:  # Max mapped_action_id is num cards on board
                        mask[mapped_action_id] = True

        elif self.game_phase == GamePhase.RESOLVING_TRANSPORT_SELECT_DEST:
            if 'possible_transport_destinations' in self.effect_resolution_data:
                for mapped_action_id, _ in self.effect_resolution_data['possible_transport_destinations']:
                    # These mapped_action_ids will be 0 and 1 for the two destination choices
                    if 0 <= mapped_action_id < self.num_actions:
                        mask[mapped_action_id] = True

        return mask

    def _get_player_strength_in_theater_with_effects(self, player_id: int, theater_idx: int) -> int:
        """Calculates player's strength in a theater, considering ongoing effects."""
        theater_obj = self.theaters[theater_idx]
        base_strength = 0

        # Check for player's own ESCALATION effect (ID 12)
        is_escalation_active = self._is_escalation_active_for_player(player_id)

        # Check for player's own active COVER FIRE (ID 16) in *this* theater
        # Note: Cover Fire only affects cards *it* covers.
        top_card_in_this_theater = theater_obj.get_uncovered_card(player_id)
        is_cover_fire_active_here_by_player = bool(
            top_card_in_this_theater and
            top_card_in_this_theater.is_face_up and
            top_card_in_this_theater.card_ref.id == 16  # COVER FIRE ID
        )

        for i, played_card in enumerate(theater_obj.player_zones[player_id]):
            card_strength = 0
            if played_card.is_face_up:
                card_strength = played_card.card_ref.strength
            else:  # Card is face-down
                card_strength = 4 if is_escalation_active else 2

            # Apply COVER FIRE if this card is covered by an active COVER FIRE
            if played_card.is_covered and is_cover_fire_active_here_by_player:
                card_strength = 4  # Or max(4, card_strength) if it could be a FU high strength card covered

            base_strength += card_strength

        # Add SUPPORT bonus (ID 5) from adjacent theaters
        support_bonus = self._get_total_support_bonus_for_theater(player_id, theater_idx)
        base_strength += support_bonus

        return base_strength

    def _handle_withdraw_action(self, current_player_obj: Player, current_action_player_idx: int,
                                step_specific_info: dict) -> tuple[float, bool]:
        """Handles the withdraw action."""
        winner_idx = 1 - current_action_player_idx
        step_specific_info["battle_winner_idx"] = winner_idx

        num_cards_left_in_hand = len(current_player_obj.hand)
        points_conceded_to_opponent = 0
        reward_for_withdrawer = 0.0

        if num_cards_left_in_hand >= 5:
            reward_for_withdrawer = -1.0
            points_conceded_to_opponent = 1
        elif num_cards_left_in_hand >= 3:
            reward_for_withdrawer = -3.0
            points_conceded_to_opponent = 3
        else:
            reward_for_withdrawer = -6.0
            points_conceded_to_opponent = 6

        self.players[winner_idx].war_vps += points_conceded_to_opponent
        self.players[current_action_player_idx].battle_reward_outcome = reward_for_withdrawer
        self.players[winner_idx].battle_reward_outcome = float(points_conceded_to_opponent)
        step_specific_info["terminated_by_action"] = True
        return reward_for_withdrawer, True  # reward, terminated

    def _apply_reactive_ongoing_effects(self, played_card_object: Card, is_face_up: bool,
                                        target_theater_idx: int, current_action_player_idx: int,
                                        cards_in_target_theater_before_play: int,
                                        step_specific_info: dict) -> bool:
        """
        Checks and applies reactive ongoing effects like CONTAINMENT and BLOCKADE.
        Returns True if the card was discarded by one of these effects, False otherwise.
        """
        # 1. Check CONTAINMENT (ID 9)
        if not is_face_up and self._is_containment_active_globally():
            step_specific_info["containment_triggered"] = f"P{current_action_player_idx}'s FD card '{played_card_object.name}' discarded by CONTAINMENT." #
            # Card is discarded. It was already removed from hand by the caller.
            # Add it to the bottom of the battle_deck.
            self.battle_deck.insert(0, played_card_object) # Insert at the beginning (index 0) to represent bottom of the deck.
            # self.battle_turn_counter += 1 # REMOVED - This will be handled by _handle_normal_play_action
            return True  # Card was discarded

        # 2. Check BLOCKADE (ID 15) - only if not discarded by CONTAINMENT
        active_blockades = self._get_active_blockades_affecting_theater(target_theater_idx, current_action_player_idx) #
        if active_blockades:
            # NEW CALCULATION: Sum of cards from BOTH players in the target theater
            total_cards_from_both_players_in_target_theater = (
                len(self.theaters[target_theater_idx].player_zones[0]) +
                len(self.theaters[target_theater_idx].player_zones[1])
            ) #

            if total_cards_from_both_players_in_target_theater >= 3: #
                blockading_card_names = ", ".join([card.name for card in active_blockades])
                step_specific_info["blockade_triggered"] = (f"P{current_action_player_idx}'s card '{played_card_object.name}' "
                                                            f"to T{target_theater_idx} (total cards: {total_cards_from_both_players_in_target_theater}) "
                                                            f"discarded by opponent's Blockade(s): {blockading_card_names}.")
                self.battle_deck.insert(0, played_card_object) #
                return True  # Card was discarded by BLOCKADE

        return False  # Card was not discarded by reactive ongoing effects

    def _trigger_instant_effects(self, played_card_object: Card, is_face_up: bool,
                                 target_theater_idx: int, current_action_player_idx: int,
                                 step_specific_info: dict) -> bool:
        """
        Checks for and initiates instant effects (MANEUVER, AMBUSH, DISRUPT).
        Sets game_phase and effect_resolution_data if an effect triggers and requires sub-steps.
        Returns True if an effect triggered and requires further sub-steps, False otherwise.
        """
        if not is_face_up:
            return False  # Instant effects typically trigger on face-up play

        effect_requires_substep = False
        original_player_of_card = current_action_player_idx  # For DISRUPT context

        if played_card_object.id == 11: # TRANSPORT (ID 11)
            self.game_phase = GamePhase.RESOLVING_TRANSPORT_SELECT_CARD
            self.effect_resolution_data = {
                'card_played_id': played_card_object.id,
                'transport_player_idx': current_action_player_idx,
                'possible_transport_sources': []
            }
            possible_sources = []
            map_id_counter = 0
            for t_idx_src in range(NUM_THEATERS):
                for stack_idx, card_in_zone in enumerate(self.theaters[t_idx_src].player_zones[current_action_player_idx]):
                    if map_id_counter < (self.num_actions - 1):  # Safety for action mapping
                        possible_sources.append((
                            map_id_counter,
                            current_action_player_idx,
                            t_idx_src, stack_idx,
                            card_in_zone.card_ref.name,
                            card_in_zone.card_ref.id
                        ))
                        map_id_counter += 1
            self.effect_resolution_data['possible_transport_sources'] = possible_sources
            if not possible_sources:  # No cards on board to move
                self.game_phase = GamePhase.NORMAL_PLAY
                self.effect_resolution_data = {}
            else:
                effect_requires_substep = True

        elif played_card_object.id == 6: # AIR DROP (ID 6)
            self.air_drop_primed_for_player[current_action_player_idx] = True
            step_specific_info["air_drop_primed"] = f"P{current_action_player_idx} will have Air Drop next turn."
            # Air Drop itself doesn't require an immediate sub-step for choice

        elif played_card_object.id == 3:  # DISRUPT
            self.game_phase = GamePhase.RESOLVING_DISRUPT_OPPONENT_CHOICE
            self.effect_resolution_data = {
                'card_played_id': played_card_object.id,
                'disrupt_player_idx': original_player_of_card,
                'opponent_idx_for_choice': 1 - original_player_of_card
            }
            opponent_targets = self._get_flippable_targets_for_player(
                self.effect_resolution_data['opponent_idx_for_choice']
            )
            self.effect_resolution_data['possible_opponent_targets'] = opponent_targets

            if not opponent_targets:  # Opponent's part fizzles
                step_specific_info["disrupt_opponent_fizzle"] = True
                self.game_phase = GamePhase.RESOLVING_DISRUPT_SELF_CHOICE  # Move to self choice
                # current_player_idx remains original_disrupt_player_idx
                self_targets = self._get_flippable_targets_for_player(original_player_of_card)
                self.effect_resolution_data['possible_self_targets'] = self_targets
                if not self_targets:  # Self part also fizzles
                    step_specific_info["disrupt_self_fizzle"] = True
                    self.game_phase = GamePhase.NORMAL_PLAY  # Fully fizzled
                    self.effect_resolution_data = {}
                    # battle_turn_counter will be incremented by caller if no sub-step
                else:
                    effect_requires_substep = True  # Still need self choice
            else:  # Opponent has targets, switch current_player_idx for their choice
                self.current_player_idx = self.effect_resolution_data['opponent_idx_for_choice']
                effect_requires_substep = True

        elif "MANEUVER" in played_card_object.ability_text:
            self.game_phase = GamePhase.RESOLVING_MANEUVER
            self.effect_resolution_data = {
                'card_played_id': played_card_object.id,
                'played_in_theater_idx': target_theater_idx
            }

            adj_indices = self._get_adjacent_theater_indices(target_theater_idx)
            possible_targets = []
            map_id_counter = 0
            for adj_idx in adj_indices:
                for p_target in [0, 1]:
                    top_card = self.theaters[adj_idx].get_uncovered_card(p_target)
                    if top_card and map_id_counter < (self.num_actions - 1):
                        possible_targets.append((map_id_counter, p_target, adj_idx, top_card.card_ref.name))
                        map_id_counter += 1
            self.effect_resolution_data['possible_targets'] = possible_targets
            if not possible_targets:  # Fizzle
                self.game_phase = GamePhase.NORMAL_PLAY
                self.effect_resolution_data = {}
            else:
                effect_requires_substep = True

        elif "AMBUSH" in played_card_object.ability_text:
            self.game_phase = GamePhase.RESOLVING_AMBUSH
            self.effect_resolution_data = {
                'card_played_id': played_card_object.id,
                'played_in_theater_idx': target_theater_idx
            }

            possible_targets = []
            map_id_counter = 0
            for t_idx_target in range(NUM_THEATERS):
                for p_target in [0, 1]:
                    top_card = self.theaters[t_idx_target].get_uncovered_card(p_target)
                    if top_card and map_id_counter < (self.num_actions - 1):
                        possible_targets.append((map_id_counter, p_target, t_idx_target, top_card.card_ref.name))
                        map_id_counter += 1
            self.effect_resolution_data['possible_targets'] = possible_targets
            if not possible_targets:  # Fizzle
                self.game_phase = GamePhase.NORMAL_PLAY
                self.effect_resolution_data = {}
            else:
                effect_requires_substep = True

        # TODO: Add other instant effects like REINFORCE, TRANSPORT, REDEPLOY here

        return effect_requires_substep

    def _handle_normal_play_action(
        self, action_id: int, current_action_player_idx: int, step_specific_info: dict
    ) -> tuple[float, bool, bool]:
        """Handles actions taken during NORMAL_PLAY phase."""
        reward = 0.0
        terminated_by_action = False
        effect_triggered_requires_substep = False
        current_player_obj = self.players[current_action_player_idx]

        # --- Activate AIR DROP if primed for this player's turn ---
        if self.air_drop_primed_for_player[current_action_player_idx]:
            self.air_drop_active_this_turn[current_action_player_idx] = True
            self.air_drop_primed_for_player[current_action_player_idx] = False
            step_specific_info["air_drop_activated_for_turn"] = f"P{current_action_player_idx}"
        # else: self.air_drop_active_this_turn[current_action_player_idx] remains False from reset or end of last turn

        card_slot_idx, target_theater_idx, is_face_up, is_withdraw = self._decode_action(action_id)

        if is_withdraw:
            reward, terminated_by_action = self._handle_withdraw_action(
                current_player_obj, current_action_player_idx, step_specific_info
            )
        else:  # Play card action
            if card_slot_idx is None or target_theater_idx is None or is_face_up is None:
                reward = -10.0
                terminated_by_action = True
                step_specific_info["error"] = "Invalid action ID for play."
            elif not (0 <= card_slot_idx < len(current_player_obj.hand)):
                reward = -10.0
                terminated_by_action = True
                step_specific_info["error"] = "Illegal action: Invalid card slot."
            else:
                card_to_play_ref = current_player_obj.hand[card_slot_idx]
                cards_in_target_th_before_play = len(
                    self.theaters[target_theater_idx].player_zones[current_action_player_idx])

                if self._is_play_action_legal(current_player_obj, card_slot_idx, card_to_play_ref, target_theater_idx,
                                              is_face_up):
                    played_card_object = current_player_obj.play_card_from_hand(card_to_play_ref.id)
                    if played_card_object:
                        # Apply reactive effects BEFORE placement if they prevent placement
                        card_discarded = self._apply_reactive_ongoing_effects(
                            played_card_object, is_face_up, target_theater_idx,
                            current_action_player_idx, cards_in_target_th_before_play,
                            step_specific_info
                        )
                        if not card_discarded:
                            self.theaters[target_theater_idx].add_card_to_zone(current_action_player_idx,
                                                                               played_card_object, is_face_up)
                            effect_triggered_requires_substep = self._trigger_instant_effects(
                                played_card_object, is_face_up, target_theater_idx,
                                current_action_player_idx, step_specific_info
                            )

                        if not effect_triggered_requires_substep:  # If no sub-step, or card was discarded
                            self.battle_turn_counter += 1
                    else:
                        reward = -10.0
                        terminated_by_action = True
                        step_specific_info["error"] = "Card consistency error."
                else:
                    reward = -10.0
                    terminated_by_action = True
                    step_specific_info["error"] = "Illegal action: Placement violation."

            if terminated_by_action and "error" in step_specific_info:  # if error caused termination
                self.players[current_action_player_idx].battle_reward_outcome = reward
                self.players[1 - current_action_player_idx].battle_reward_outcome = 0.0

        return reward, terminated_by_action, effect_triggered_requires_substep

    def _handle_effect_resolution_action(self, action_id: int, current_action_player_idx: int,
                                         step_specific_info: dict) -> tuple[float, bool, bool]:
        """Handles actions taken during an effect resolution phase."""
        reward = 0.0  # Typically no direct reward for effect choices themselves
        terminated_by_action = False  # Effect choices usually don't end the battle directly
        effect_requires_substep = False  # Assume effect will resolve fully in this step unless DISRUPT transitions

        original_disrupt_player_idx = self.effect_resolution_data.get('disrupt_player_idx')

        if self.game_phase == GamePhase.RESOLVING_TRANSPORT_SELECT_CARD:
            chosen_source_details = None
            if 'possible_transport_sources' in self.effect_resolution_data:
                for mapped_id, src_p_id, src_t_idx, src_stack_idx, card_name, card_obj_id in self.effect_resolution_data['possible_transport_sources']:
                    if action_id == mapped_id and src_p_id == current_action_player_idx:
                        chosen_source_details = {
                            "player_id": src_p_id, "theater_idx": src_t_idx,
                            "stack_idx": src_stack_idx, "card_name": card_name, "card_id": card_obj_id
                        }
                        break
            if chosen_source_details:
                self.effect_resolution_data['card_to_transport_details'] = chosen_source_details
                self.game_phase = GamePhase.RESOLVING_TRANSPORT_SELECT_DEST

                possible_destinations = []
                map_id_counter = 0
                for dest_t_idx in range(NUM_THEATERS):
                    if dest_t_idx != chosen_source_details["theater_idx"]:  # Cannot move to same theater
                        possible_destinations.append((map_id_counter, dest_t_idx))
                        map_id_counter += 1
                self.effect_resolution_data['possible_transport_destinations'] = possible_destinations

                if not possible_destinations:  # Should not happen with NUM_THEATERS = 3
                    step_specific_info["transport_effect"] = "No valid destinations, fizzled."
                    self.game_phase = GamePhase.NORMAL_PLAY
                    self.effect_resolution_data = {}
                    self.battle_turn_counter += 1
                else:
                    effect_requires_substep = True
            else:
                step_specific_info["transport_effect"] = "Invalid source card choice or no source selected, fizzled."
                self.game_phase = GamePhase.NORMAL_PLAY
                self.effect_resolution_data = {}
                self.battle_turn_counter += 1

        elif self.game_phase == GamePhase.RESOLVING_TRANSPORT_SELECT_DEST:
            chosen_dest_theater_idx = -1
            if 'possible_transport_destinations' in self.effect_resolution_data:
                for mapped_id, dest_t_idx in self.effect_resolution_data['possible_transport_destinations']:
                    if action_id == mapped_id:
                        chosen_dest_theater_idx = dest_t_idx
                        break

            if chosen_dest_theater_idx != -1 and 'card_to_transport_details' in self.effect_resolution_data:
                details = self.effect_resolution_data['card_to_transport_details']
                moved_played_card_instance = self.theaters[details['theater_idx']].remove_card_from_zone_at_index(
                    details['player_id'], details['stack_idx']
                )
                if moved_played_card_instance:
                    # Add to new theater. add_played_card_instance_to_zone handles covering.
                    self.theaters[chosen_dest_theater_idx].add_played_card_instance_to_zone(
                        details['player_id'], moved_played_card_instance
                    )
                    step_specific_info["transport_moved"] = f"'{details['card_name']}' moved from T{details['theater_idx']} to T{chosen_dest_theater_idx}"
                else:
                    step_specific_info["transport_error"] = "Failed to remove card from source for transport."
            else:
                step_specific_info["transport_effect"] = "Invalid destination choice or card details missing, fizzled."

            self.game_phase = GamePhase.NORMAL_PLAY
            self.effect_resolution_data = {}
            self.battle_turn_counter += 1
            effect_requires_substep = False

        elif self.game_phase == GamePhase.RESOLVING_MANEUVER or self.game_phase == GamePhase.RESOLVING_AMBUSH:
            effect_name = self.game_phase.name.split('_')[1].lower()
            chosen_target_details = None
            if 'possible_targets' in self.effect_resolution_data:
                for map_id, p_id, t_idx, _ in self.effect_resolution_data.get('possible_targets', []):
                    if action_id == map_id:
                        chosen_target_details = (p_id, t_idx)
                        break
            if chosen_target_details:
                self._flip_card_in_theater(chosen_target_details[0], chosen_target_details[1])
                step_specific_info[
                    f"{effect_name}_target"] = f"P{chosen_target_details[0]}'s card in T{chosen_target_details[1]}"
            else:
                step_specific_info[f"{effect_name}_effect"] = "Fizzled or invalid choice."

            self.game_phase = GamePhase.NORMAL_PLAY
            self.effect_resolution_data = {}
            self.battle_turn_counter += 1  # Count the original card play as a turn
            # effect_requires_substep remains False, as this phase concludes the effect

        elif self.game_phase == GamePhase.RESOLVING_DISRUPT_OPPONENT_CHOICE:
            # current_action_player_idx is the opponent
            chosen_target = None
            if 'possible_opponent_targets' in self.effect_resolution_data:
                for map_id, p_id, t_idx, _ in self.effect_resolution_data['possible_opponent_targets']:
                    if action_id == map_id and p_id == current_action_player_idx:  # Opponent chooses their own
                        chosen_target = (p_id, t_idx)
                        break
            if chosen_target:
                self._flip_card_in_theater(chosen_target[0], chosen_target[1])
                step_specific_info["disrupt_opponent_flip"] = f"P{chosen_target[0]}'s card in T{chosen_target[1]}"
            else:
                step_specific_info["disrupt_opponent_fizzle_choice"] = True

            # Transition to self choice
            self.current_player_idx = original_disrupt_player_idx  # Switch back to disrupt player
            self.game_phase = GamePhase.RESOLVING_DISRUPT_SELF_CHOICE
            self_targets = self._get_flippable_targets_for_player(original_disrupt_player_idx)
            self.effect_resolution_data['possible_self_targets'] = self_targets  # Store self targets
            # Update effect_resolution_data, clear opponent targets if no longer needed or keep for context
            self.effect_resolution_data.pop('possible_opponent_targets', None)

            if not self_targets:  # Self part fizzles
                step_specific_info["disrupt_self_fizzle"] = True
                self.game_phase = GamePhase.NORMAL_PLAY
                self.effect_resolution_data = {}  # Clear all effect data
                self.battle_turn_counter += 1  # Count original DISRUPT play
                effect_requires_substep = False  # Explicitly False as effect fully ended
            else:
                effect_requires_substep = True  # CORRECTED: Use the initialized variable name

        elif self.game_phase == GamePhase.RESOLVING_DISRUPT_SELF_CHOICE:
            # current_action_player_idx is the original disrupt player
            chosen_target = None
            if 'possible_self_targets' in self.effect_resolution_data:
                for map_id, p_id, t_idx, _ in self.effect_resolution_data['possible_self_targets']:
                    if action_id == map_id and p_id == current_action_player_idx:  # Player chooses their own
                        chosen_target = (p_id, t_idx)
                        break
            if chosen_target:
                self._flip_card_in_theater(chosen_target[0], chosen_target[1])
                step_specific_info["disrupt_self_flip"] = f"P{chosen_target[0]}'s card in T{chosen_target[1]}"
            else:
                step_specific_info["disrupt_self_fizzle_choice"] = True

            self.game_phase = GamePhase.NORMAL_PLAY
            self.effect_resolution_data = {}
            self.battle_turn_counter += 1  # Count original DISRUPT play
            effect_requires_substep = False  # Explicitly False as effect fully ended

        # It's good practice to have a fallback for unexpected game phases,
        # though with enums, this path should ideally not be hit.
        else:
            step_specific_info["error"] = f"Unknown or unhandled effect resolution phase: {self.game_phase}"
            terminated_by_action = True  # Or False, depending on desired error handling
            reward = -100.0  # Penalize for unexpected state
            effect_requires_substep = False  # Default for error state

        return reward, terminated_by_action, effect_requires_substep

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        current_action_player_idx = self.current_player_idx  # Player whose turn it *is* for this action/choice
        reward = 0.0
        terminated = False  # Overall battle termination
        truncated = False
        step_specific_info = {}  # Collects info specific to this action/sub-step

        # This flag indicates if the action taken has led to a new effect resolution phase
        # for the *same* player, meaning the turn does not end yet.
        effect_sub_step_pending = False

        # --- Activate AIR DROP if primed for the current player's turn, only in NORMAL_PLAY ---
        if self.game_phase == GamePhase.NORMAL_PLAY:
            if self.air_drop_primed_for_player[current_action_player_idx]:
                self.air_drop_active_this_turn[current_action_player_idx] = True
                self.air_drop_primed_for_player[current_action_player_idx] = False
                step_specific_info["air_drop_activated_for_turn"] = f"P{current_action_player_idx}"
            # Ensure it's off if it wasn't just primed (it should have been cleared end of last turn)
            # else: self.air_drop_active_this_turn[current_action_player_idx] = False
            # Better to reset it explicitly at end of turn where it was active.

        if self.game_phase == GamePhase.NORMAL_PLAY:
            reward, terminated, effect_sub_step_pending = self._handle_normal_play_action(
                action_id, current_action_player_idx, step_specific_info
            )
        elif self.game_phase in [
            GamePhase.RESOLVING_MANEUVER,
            GamePhase.RESOLVING_AMBUSH,
            GamePhase.RESOLVING_DISRUPT_OPPONENT_CHOICE,
            GamePhase.RESOLVING_DISRUPT_SELF_CHOICE,
            GamePhase.RESOLVING_TRANSPORT_SELECT_CARD,
            GamePhase.RESOLVING_TRANSPORT_SELECT_DEST
        ]:
            reward, terminated, effect_sub_step_pending = self._handle_effect_resolution_action(
                action_id, current_action_player_idx, step_specific_info
            )
        else:
            step_specific_info["error"] = f"Unknown game phase: {self.game_phase}"
            terminated = True
            reward = -100.0  # Severe penalty

        # 3. Post-Action Processing: Deactivate one-turn effects, check game end, switch player

        # If an effect sub-step is NOT pending, it means current_action_player_idx's
        # conceptual "turn" or "action sequence" is complete.
        if not effect_sub_step_pending:
            # Deactivate AIR DROP for the player who just completed their action sequence
            self.air_drop_active_this_turn[current_action_player_idx] = False

            # Now, check if this completed action sequence also ended the battle
            if not terminated:  # Only check if not already terminated by the action/effect itself
                if len(self.players[0].hand) == 0 and len(self.players[1].hand) == 0:
                    terminated = True
                    winner_idx = self._determine_battle_winner()
                    step_specific_info["battle_winner_idx"] = winner_idx
                    if winner_idx == current_action_player_idx:
                        reward = 6.0
                    elif winner_idx is not None:
                        reward = -6.0
                    # (Update War VPs and battle_reward_outcomes for normal end)
                    if winner_idx is not None:
                        self.players[winner_idx].war_vps += 6
                        self.players[winner_idx].battle_reward_outcome = 6.0
                        self.players[1 - winner_idx].battle_reward_outcome = -6.0
                    else:  # Draw
                        self.players[0].battle_reward_outcome = 0.0
                        self.players[1].battle_reward_outcome = 0.0

            # Switch player for the NEXT turn if game hasn't ended and we are in normal play phase
            # (DISRUPT handles its own internal player switches for choices)
            if not terminated and self.game_phase == GamePhase.NORMAL_PLAY:
                self.current_player_idx = 1 - current_action_player_idx

        # Ensure battle outcomes are set if terminated by an error during normal play
        if terminated and "error" in step_specific_info and not ("battle_winner_idx" in step_specific_info):
            # This error happened on current_action_player_idx's turn
            self.players[current_action_player_idx].battle_reward_outcome = reward  # e.g., -10.0
            self.players[1 - current_action_player_idx].battle_reward_outcome = 0.0

        obs = self._get_observation()  # Observation for the NEW self.current_player_idx and self.game_phase
        final_info = self._get_info()  # Info for the NEW self.current_player_idx and self.game_phase
        final_info.update(step_specific_info)  # Merge outcomes of the action just taken

        return obs, reward, terminated, truncated, final_info

    def _get_theater_control_status(self, theater_idx: int) -> int | None:
        """Determines which player controls a theater, considering all effects."""
        s0 = self._get_player_strength_in_theater_with_effects(0, theater_idx)  # Use new method
        s1 = self._get_player_strength_in_theater_with_effects(1, theater_idx)  # Use new method

        if s0 > s1: return 0
        if s1 > s0: return 1
        if s0 == s1 and s0 > 0: return self.battle_first_player_idx
        return None

    def _determine_battle_winner(self) -> int | None:
        p0_controlled_theaters = 0
        p1_controlled_theaters = 0
        for i in range(NUM_THEATERS):
            controller = self._get_theater_control_status(i)
            if controller == 0:
                p0_controlled_theaters += 1
            elif controller == 1:
                p1_controlled_theaters += 1
        if p0_controlled_theaters > p1_controlled_theaters: return 0
        if p1_controlled_theaters > p0_controlled_theaters: return 1
        return None

    def _get_observation(self) -> np.ndarray:
        obs_parts = []
        acting_player_obj = self.players[self.current_player_idx]
        opponent_player_idx = 1 - self.current_player_idx
        opponent_player_obj = self.players[opponent_player_idx]

        for i in range(HAND_SIZE):
            if i < len(acting_player_obj.hand):
                card = acting_player_obj.hand[i]
                obs_parts.extend([card.id, card.strength, card.card_type.value])
            else:
                obs_parts.extend([-1, -1, -1])

        obs_parts.append(len(opponent_player_obj.hand))

        for i in range(NUM_THEATERS):
            theater = self.theaters[i]
            obs_parts.append(theater.current_type.value)
            obs_parts.append(self._get_player_strength_in_theater_with_effects(self.current_player_idx, i))
            agent_top_card = theater.get_uncovered_card(self.current_player_idx)
            if agent_top_card:
                obs_parts.extend([1, agent_top_card.card_ref.id, agent_top_card.card_ref.strength,
                                  agent_top_card.get_current_strength(self._is_escalation_active_for_player(self.current_player_idx)),  # Corrected
                                  1 if agent_top_card.is_face_up else 0,
                                  agent_top_card.card_ref.card_type.value])
            else:
                obs_parts.extend([0, -1, -1, -1, -1, -1])
            agent_covered_count = len(theater.player_zones[self.current_player_idx]) - (1 if agent_top_card else 0)
            obs_parts.append(agent_covered_count)
            obs_parts.append(self._get_player_strength_in_theater_with_effects(opponent_player_idx, i))
            opp_top_card = theater.get_uncovered_card(opponent_player_idx)
            if opp_top_card:
                obs_parts.extend(
                    [1, opp_top_card.get_current_strength(self._is_escalation_active_for_player(opponent_player_idx)),
                     1 if opp_top_card.is_face_up else 0,
                     opp_top_card.card_ref.card_type.value])
            else:
                obs_parts.extend([0, -1, -1, -1])
            opponent_covered_count = len(theater.player_zones[opponent_player_idx]) - (1 if opp_top_card else 0)
            obs_parts.append(opponent_covered_count)
        obs_parts.append(self.battle_turn_counter)
        obs_parts.append(len(self.battle_deck))
        obs_parts.append(self.battle_first_player_idx)
        obs_parts.append(self.game_phase.value)
        # --- Add AIR DROP active status for current player to observation ---
        obs_parts.append(1 if self.air_drop_active_this_turn[self.current_player_idx] else 0)
        # New total features: 68 (old) + 1 (air_drop_active) = 69
        return np.array(obs_parts, dtype=np.float32)

    def _get_info(self) -> dict:
        p0_battle_outcome = getattr(self.players[0], 'battle_reward_outcome', 0.0)
        p1_battle_outcome = getattr(self.players[1], 'battle_reward_outcome', 0.0)
        info_dict = {
            "current_player_idx": self.current_player_idx,
            "battle_first_player_idx": self.battle_first_player_idx,
            "battle_turn_counter": self.battle_turn_counter,
            "p0_hand_size": len(self.players[0].hand),
            "p1_hand_size": len(self.players[1].hand),
            "p0_war_vps": self.players[0].war_vps,
            "p1_war_vps": self.players[1].war_vps,
            "p0_battle_reward_outcome": p0_battle_outcome,
            "p1_battle_reward_outcome": p1_battle_outcome,
            "battle_deck_size": len(self.battle_deck),
            "game_phase": self.game_phase.name,  # For readability
            "action_mask": self._get_action_mask(),
            "air_drop_primed_p0": self.air_drop_primed_for_player[0],  # For debugging
            "air_drop_primed_p1": self.air_drop_primed_for_player[1],  # For debugging
            "air_drop_active_p0": self.air_drop_active_this_turn[0],  # For debugging
            "air_drop_active_p1": self.air_drop_active_this_turn[1]  # For debugging
        }
        # Add effect-specific data if in an effect phase for debugging
        if self.game_phase != GamePhase.NORMAL_PLAY and self.effect_resolution_data:
            if self.game_phase == GamePhase.RESOLVING_DISRUPT_OPPONENT_CHOICE:
                info_dict["effect_data_opponent_targets"] = self.effect_resolution_data.get('possible_opponent_targets')
            elif self.game_phase == GamePhase.RESOLVING_DISRUPT_SELF_CHOICE:
                info_dict["effect_data_self_targets"] = self.effect_resolution_data.get('possible_self_targets')
            elif self.game_phase == GamePhase.RESOLVING_TRANSPORT_SELECT_CARD:
                info_dict["effect_data_transport_sources"] = self.effect_resolution_data.get('possible_transport_sources')
            elif self.game_phase == GamePhase.RESOLVING_TRANSPORT_SELECT_DEST:
                info_dict["effect_data_transport_destinations"] = self.effect_resolution_data.get('possible_transport_destinations')
                info_dict["effect_data_card_to_transport"] = self.effect_resolution_data.get('card_to_transport_details')
            elif 'possible_targets' in self.effect_resolution_data:  # For MANEUVER/AMBUSH
                info_dict["effect_data_targets"] = self.effect_resolution_data.get('possible_targets')
        return info_dict

    def render(self, mode='human'):
        if mode == 'human':
            print("\n" + "=" * 30)
            current_info = self._get_info()
            print(f"Current Player: P{current_info['current_player_idx']} "
                  f"(Battle First Player: P{current_info['battle_first_player_idx']}) "
                  f"Phase: {current_info['game_phase']}")
            print(f"Battle Turn: {current_info['battle_turn_counter']}")
            print(f"War VPs: P0:{current_info['p0_war_vps']}, P1:{current_info['p1_war_vps']}")
            print(f"Last Battle Outcome: "
                  f"P0:{current_info['p0_battle_reward_outcome']}, "
                  f"P1:{current_info['p1_battle_reward_outcome']}")
            print(f"Battle Deck: {current_info['battle_deck_size']} cards")
            print(f"AirDrop Primed: "
                  f"P0({current_info['air_drop_primed_p0']}), "
                  f"P1({current_info['air_drop_primed_p1']}) | "
                  f"AirDrop Active: "
                  f"P0({current_info['air_drop_active_p0']}), "
                  f"P1({current_info['air_drop_active_p1']})") # Debug AirDrop
            print("-" * 15)
            print(self.players[0])
            print(self.players[1])
            print("-" * 15)
            print("Theaters (Physical Order: Left, Center, Right):")
            for i in range(NUM_THEATERS):
                theater = self.theaters[i]
                s0 = self._get_player_strength_in_theater_with_effects(0, i)
                s1 = self._get_player_strength_in_theater_with_effects(1, i)
                controller = self._get_theater_control_status(i)
                control_str = f"Ctrl by P{controller}" if controller is not None else "Ctrl by None"
                p0_cov = len(theater.player_zones[0]) - (1 if theater.get_uncovered_card(0) else 0)
                p1_cov = len(theater.player_zones[1]) - (1 if theater.get_uncovered_card(1) else 0)
                print(f"  Pos {i} ({theater.current_type.name}): Str P0:{s0}, P1:{s1} - {control_str}")
                print(f"    P0 Zone ({p0_cov} cov): {theater.player_zones[0]} (Top: {theater.get_uncovered_card(0)})")
                print(f"    P1 Zone ({p1_cov} cov): {theater.player_zones[1]} (Top: {theater.get_uncovered_card(1)})")

            # Show effect data in render if pending
            if current_info['game_phase'] == GamePhase.RESOLVING_DISRUPT_OPPONENT_CHOICE.name:
                print(f"--- DISRUPT: P{current_info['current_player_idx']} "
                      f"(Opponent) to choose one of their cards to flip ---")
                print(f"    Targets: {current_info.get('effect_data_opponent_targets')}")
            elif current_info['game_phase'] == GamePhase.RESOLVING_DISRUPT_SELF_CHOICE.name:
                print(f"--- DISRUPT: P{current_info['current_player_idx']} "
                      f"(Self) to choose one of their cards to flip ---")
                print(f"    Targets: {current_info.get('effect_data_self_targets')}")
            elif current_info['game_phase'] == GamePhase.RESOLVING_TRANSPORT_SELECT_CARD.name:
                print(f"--- TRANSPORT Select Card: {current_info['effect_data_transport_sources']}")
            elif current_info['game_phase'] == GamePhase.RESOLVING_TRANSPORT_SELECT_DEST.name:
                print(f"--- TRANSPORT Select Dest for "
                      f"'{current_info.get('effect_data_card_to_transport', {}).get('card_name')}': "
                      f"{current_info['effect_data_transport_destinations']}")
            elif current_info['game_phase'] in [GamePhase.RESOLVING_MANEUVER.name, GamePhase.RESOLVING_AMBUSH.name]:
                print(f"--- {current_info['game_phase']}: P{current_info['current_player_idx']} to choose target ---")
                print(f"    Targets: {current_info.get('effect_data_targets')}")

            print("=" * 30 + "\n")
    @property
    def action_space(self):
        env_instance = self

        class ActionSpace:
            def __init__(self, n_actions, env_ref):
                self.n = n_actions
                self.env_ref = env_ref

            def sample(self):
                action_mask = self.env_ref._get_action_mask()
                legal_actions = np.where(action_mask)[0]
                if len(legal_actions) == 0:
                    print(f"Warning: No legal actions in mask! Phase: {self.env_ref.game_phase.name}")
                    if self.env_ref.withdraw_action_id < self.n and action_mask[self.env_ref.withdraw_action_id]:
                        return self.env_ref.withdraw_action_id
                    if self.n > 0:
                        return random.randint(0, self.n - 1)
                    raise ValueError("No legal actions & action space zero.")
                return random.choice(legal_actions)

        return ActionSpace(self.num_actions, env_instance)

    @property
    def observation_space(self):
        class ObservationSpace:
            def __init__(self, shape_dims): self.shape = shape_dims

        return ObservationSpace(self.observation_shape)

    def close(self):
        pass
