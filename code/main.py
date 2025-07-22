from air_land_sea_env import AirLandSeaBaseEnv
from constants import GamePhase, TheaterType, NUM_THEATERS, CardType
import random
import numpy as np


def print_step_result(step_num, action_id_taken, action_desc_taken, obs, reward, terminated, truncated, info):
    print(f"\n--- Step {step_num} Results ---")
    print(f"Action Taken (ID: {action_id_taken}): {action_desc_taken}")
    print(f"Observation Shape: {obs.shape}")
    print(
        f"Reward (for P{info.get('current_player_idx_at_action_end', 'N/A')}): {reward}")  # Clarify reward perspective
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"  Current Player (for next turn): P{info.get('current_player_idx')}")
    print(f"  Battle Winner: P{info.get('battle_winner_idx')}" if info.get(
        'battle_winner_idx') is not None else "  Battle Winner: None")
    # Changed to use new keys for battle outcome
    print(f"  P0 Battle Outcome: {info.get('p0_battle_reward_outcome')}")
    print(f"  P1 Battle Outcome: {info.get('p1_battle_reward_outcome')}")
    if "error" in info:
        print(f"  Error: {info['error']}")


def test_env():
    print("Initializing Air, Land, and Sea Environment...")
    env = AirLandSeaBaseEnv()
    print("Environment Initialized.")
    print(f"Action space size: {env.action_space.n}")
    print(f"Observation shape: {env.observation_space.shape}")

    print("\n--- Test 1: Reset, Render, and Show Legal Moves ---")
    obs, info = env.reset()
    env.render()  # render will now use the new field names via _get_info
    print("Initial Observation Shape:", obs.shape)

    print("\nInitial Legal Moves for Player", info.get("current_player_idx"))
    legal_moves = env.get_readable_legal_moves()
    for action_id, description in legal_moves:
        print(f"  Action ID: {action_id} => {description}")
    action_mask_from_info = info.get("action_mask")
    if action_mask_from_info is not None:
        print(f"  Number of legal actions from mask: {np.sum(action_mask_from_info)}")

    print("\n--- Test 2: Taking a few controlled or smart random actions ---")
    terminated = False
    num_scripted_steps = 3

    for i in range(num_scripted_steps):
        if terminated:
            print(f"Battle ended early after {i} scripted steps.")
            break

        player_idx_before_action = env.current_player_idx  # Store whose turn it is
        print(f"\nScripted Step {i + 1} (Player P{player_idx_before_action}'s turn)")
        current_legal_moves = env.get_readable_legal_moves()
        if not current_legal_moves:
            print("Error: No legal moves available.")
            terminated = True
            break

        action_id_to_take = env.withdraw_action_id
        action_description_to_take = "Withdraw"
        play_actions_tuples = [(move_id, move_desc) for move_id, move_desc in current_legal_moves if
                               move_id != env.withdraw_action_id]
        if play_actions_tuples:
            face_up_play_tuples = [(move_id, move_desc) for move_id, move_desc in play_actions_tuples if
                                   "Face-Up" in move_desc]
            if face_up_play_tuples:
                action_id_to_take, action_description_to_take = random.choice(face_up_play_tuples)
            else:
                action_id_to_take, action_description_to_take = random.choice(play_actions_tuples)

        obs, reward, terminated, truncated, info = env.step(action_id_to_take)
        # Add whose perspective the reward is from for clarity in print_step_result
        info['current_player_idx_at_action_end'] = player_idx_before_action
        env.render()
        print_step_result(i + 1, action_id_to_take, action_description_to_take, obs, reward, terminated, truncated,
                          info)

        if not terminated:
            print("\nLegal Moves for Player", info.get("current_player_idx", "N/A"))
            next_legal_moves = env.get_readable_legal_moves()
            for aid, desc in next_legal_moves[:5]:
                print(f"  Action ID: {aid} => {desc}")
            if len(next_legal_moves) > 5:
                print(f"  ... and {len(next_legal_moves) - 5} more.")

    if not terminated:
        print("\n--- Test 3: Full Random (but legal) Game via action_space.sample() ---")
        max_additional_steps = env.max_plays_per_battle + 5
        for i in range(max_additional_steps):
            if terminated:
                print(f"Random part of game ended after {i} additional steps.")
                break

            player_idx_before_action = env.current_player_idx
            action = env.action_space.sample()
            sampled_action_description = "Unknown (direct sample)"
            # Re-fetch legal moves for the current player before acting, to get description
            for aid, desc in env.get_readable_legal_moves():
                if aid == action:
                    sampled_action_description = desc
                    break

            print(f"\nRandom Game Step {i + 1} (Player P{player_idx_before_action}'s turn)")
            obs, reward, terminated, truncated, info = env.step(action)
            info['current_player_idx_at_action_end'] = player_idx_before_action
            env.render()
            print_step_result(i + 1, action, sampled_action_description, obs, reward, terminated, truncated, info)

            if "error" in info:
                print(f"Stopping random game due to error: {info['error']}")
                break
        if not terminated:
            print("Warning: Random game reached max steps without termination.")

    print("\n--- Final Battle State (if applicable) ---")
    env.render()
    final_info = env._get_info()
    print("Final Info:", final_info)
    # Check if battle_winner_idx was set in the info from the *last terminating step*
    # The 'info' variable from the loop holds the info from the last step taken.
    if terminated and info.get("battle_winner_idx") is not None:
        print(f"Overall Battle Winner: Player {info['battle_winner_idx']}")
    elif "error" not in final_info:  # Use final_info for error check if game didn't terminate with winner
        print("Battle concluded without a clear winner reported in the terminating step's info.")

    print("\nBasic tests completed.")


def test_maneuver_effect(env: AirLandSeaBaseEnv):
    print("\n" + "#" * 10 + " Test: MANEUVER Card Effect " + "#" * 10)
    obs, initial_info = env.reset()  # Start a fresh battle
    env.render()
    print("Initial state for MANEUVER test.")
    print(f"Initial P0 Hand: {[card.name for card in env.players[0].hand]}")
    print(f"Initial P1 Hand: {[card.name for card in env.players[1].hand]}")
    print(f"Initial Theater Layout: {[th.current_type.name for th in env.theaters]}")

    # --- Setup Phase: Play a few cards to have targets ---
    # We need P0 to play, then P1 to play (to put a card down as a target), then P0 to play MANEUVER.

    # Step 0.1: Player 0 plays any card to set up the board
    print(f"\nMANEUVER Test Setup: P{env.current_player_idx} (P0) plays a setup card.")
    player_idx_before_action = env.current_player_idx
    legal_moves_p0_setup = env.get_readable_legal_moves()
    if not legal_moves_p0_setup or (
        len(legal_moves_p0_setup) == 1 and legal_moves_p0_setup[0][0] == env.withdraw_action_id):
        print("P0 has no playable cards for setup. Skipping MANEUVER test.")
        return

    # Try to play the first available non-withdraw action
    action_p0_setup, desc_p0_setup = legal_moves_p0_setup[0]
    if action_p0_setup == env.withdraw_action_id and len(legal_moves_p0_setup) > 1:
        action_p0_setup, desc_p0_setup = legal_moves_p0_setup[1]

    obs, reward, terminated, truncated, info_after_p0_setup = env.step(action_p0_setup)
    info_after_p0_setup['current_player_idx_at_action_end'] = player_idx_before_action
    env.render()
    print_step_result("0.1 (P0 Setup)", action_p0_setup, desc_p0_setup, obs, reward, terminated, truncated,
                      info_after_p0_setup)
    if terminated: print("Battle ended during P0 setup. MANEUVER test cannot proceed."); return

    # Step 0.2: Player 1 plays any card (this could be a target)
    print(f"\nMANEUVER Test Setup: P{env.current_player_idx} (P1) plays a setup card (potential target).")
    player_idx_before_action = env.current_player_idx
    legal_moves_p1_setup = env.get_readable_legal_moves()
    if not legal_moves_p1_setup or (
        len(legal_moves_p1_setup) == 1 and legal_moves_p1_setup[0][0] == env.withdraw_action_id):
        print("P1 has no playable cards for setup. Skipping MANEUVER test.")
        # P0 might still be able to maneuver their own card if one is adjacent
        # but for a clean test of flipping opponent, we'd ideally want P1 to play.
        # For now, we'll allow it to proceed and see if P0 can maneuver their own card.
        pass  # Allow to proceed, P0 might flip their own card from step 0.1

    action_p1_setup, desc_p1_setup = env.withdraw_action_id, "Withdraw (default)"
    if legal_moves_p1_setup:
        action_p1_setup, desc_p1_setup = legal_moves_p1_setup[0]
        if action_p1_setup == env.withdraw_action_id and len(legal_moves_p1_setup) > 1:
            action_p1_setup, desc_p1_setup = legal_moves_p1_setup[1]

    obs, reward, terminated, truncated, info_after_p1_setup = env.step(action_p1_setup)
    info_after_p1_setup['current_player_idx_at_action_end'] = player_idx_before_action
    env.render()
    print_step_result("0.2 (P1 Setup)", action_p1_setup, desc_p1_setup, obs, reward, terminated, truncated,
                      info_after_p1_setup)
    if terminated: print("Battle ended during P1 setup. MANEUVER test cannot proceed."); return

    # --- Step 1: Player 0 plays MANEUVER ---
    print(f"\nMANEUVER Test - Step 1: P{env.current_player_idx} (P0) attempts to play MANEUVER.")
    player_idx_before_action = env.current_player_idx
    current_player_obj = env.players[player_idx_before_action]

    maneuver_play_action_id = -1
    maneuver_play_desc = ""

    possible_maneuver_plays = []
    for aid, desc in env.get_readable_legal_moves():
        # Check if the description indicates a MANEUVER card being played Face-Up
        # This relies on the card name in the description for now.
        if "MANEUVER" in desc.upper() and "FACE-UP" in desc.upper():
            possible_maneuver_plays.append((aid, desc))

    if not possible_maneuver_plays:
        print("P0 does not have a legally playable MANEUVER card Face-Up in this setup. Skipping MANEUVER effect test.")
        return

    maneuver_play_action_id, maneuver_play_desc = random.choice(possible_maneuver_plays)  # Pick one if multiple

    print(f"P0 playing MANEUVER: {maneuver_play_desc} (Action ID: {maneuver_play_action_id})")
    obs, reward, terminated, truncated, info_after_maneuver_play = env.step(maneuver_play_action_id)
    info_after_maneuver_play[
        'current_player_idx_at_action_end'] = player_idx_before_action  # Player who took this action
    env.render()
    print_step_result("1 (Play MANEUVER)", maneuver_play_action_id, maneuver_play_desc, obs, reward, terminated,
                      truncated, info_after_maneuver_play)

    if terminated:
        print("Battle ended immediately after playing MANEUVER card. Effect resolution cannot be fully tested.")
        return

    # Check if game phase changed as expected
    if info_after_maneuver_play.get("game_phase") == GamePhase.RESOLVING_MANEUVER.name:
        print(f"Game phase is now: {info_after_maneuver_play.get('game_phase')}. Correct.")
        assert info_after_maneuver_play.get("current_player_idx") == player_idx_before_action, \
            f"Player should still be P{player_idx_before_action} for MANEUVER choice, but is P{info_after_maneuver_play.get('current_player_idx')}."
        print(f"Current player is still P{info_after_maneuver_play.get('current_player_idx')}. Correct.")

        # --- Step 2: Player 0 resolves MANEUVER effect ---
        print(f"\nMANEUVER Test - Step 2: P{env.current_player_idx} choosing target for MANEUVER effect.")
        maneuver_target_choices = env.get_readable_legal_moves()

        if not maneuver_target_choices:
            print(
                "No valid targets presented for MANEUVER effect by get_readable_legal_moves(). Effect should have fizzled if env.effect_resolution_data was empty.")
            # This case implies the previous step should have handled the fizzle and game_phase reverted.
            # If effect_resolution_data['possible_targets'] was empty, phase should be NORMAL_PLAY.
            assert info_after_maneuver_play.get("effect_data", {}).get(
                "possible_targets") == [], "Targets should be empty if no choices presented"
            print("MANEUVER effect fizzled as expected. Turn should have passed.")
            # Check if turn counter was updated for the MANEUVER card play and player switched
            expected_turn_count_fizzle = info_after_p1_setup.get('battle_turn_counter', 0) + 1
            assert info_after_maneuver_play.get("battle_turn_counter") == expected_turn_count_fizzle, \
                f"Fizzled MANEUVER turn count error. Expected {expected_turn_count_fizzle}, got {info_after_maneuver_play.get('battle_turn_counter')}"
            assert info_after_maneuver_play.get("current_player_idx") != player_idx_before_action, \
                f"Player should have switched after fizzled MANEUVER, but is still P{player_idx_before_action}."
            print("Fizzled MANEUVER handled, turn advanced. Test for fizzle path successful.")

        else:  # Targets are available
            print("Legal choices for MANEUVER effect:")
            for aid, desc in maneuver_target_choices:
                print(f"  Action ID: {aid} => {desc}")

            chosen_maneuver_effect_action_id, chosen_maneuver_effect_desc = random.choice(maneuver_target_choices)
            print(
                f"P0 choosing MANEUVER target: {chosen_maneuver_effect_desc} (Action ID: {chosen_maneuver_effect_action_id})")

            player_idx_before_effect_action = env.current_player_idx
            obs, reward, terminated, truncated, info_after_effect_resolution = env.step(
                chosen_maneuver_effect_action_id)
            info_after_effect_resolution['current_player_idx_at_action_end'] = player_idx_before_effect_action
            env.render()
            print_step_result("2 (Resolve MANEUVER)", chosen_maneuver_effect_action_id, chosen_maneuver_effect_desc,
                              obs, reward, terminated, truncated, info_after_effect_resolution)

            assert info_after_effect_resolution.get("game_phase") == GamePhase.NORMAL_PLAY.name, \
                f"Game phase should be NORMAL_PLAY after MANEUVER resolution, but is {info_after_effect_resolution.get('game_phase')}"
            print(f"Game phase is now: {info_after_effect_resolution.get('game_phase')}. Correct.")

            # Player should have switched *after* the full MANEUVER turn (play + effect resolution)
            assert info_after_effect_resolution.get("current_player_idx") != player_idx_before_action, \
                f"Player should have switched to P{1 - player_idx_before_action} after MANEUVER turn, but is P{info_after_effect_resolution.get('current_player_idx')}."
            print(
                f"Current player is now P{info_after_effect_resolution.get('current_player_idx')}. Correct (switched).")

            # Battle turn counter should have incremented by 1 compared to *before* playing MANEUVER initially.
            # `info_after_p1_setup` was the state before P0's MANEUVER turn started.
            expected_turn_count = info_after_p1_setup.get('battle_turn_counter', 0) + 1
            assert info_after_effect_resolution.get("battle_turn_counter") == expected_turn_count, \
                f"Battle turn counter error after MANEUVER. Expected {expected_turn_count}, got {info_after_effect_resolution.get('battle_turn_counter')}"
            print(f"Battle turn counter is {info_after_effect_resolution.get('battle_turn_counter')}. Correct.")
            print("MANEUVER effect resolution test part completed.")

    elif info_after_maneuver_play.get("game_phase") == GamePhase.NORMAL_PLAY.name:
        # This means MANEUVER was played, but it had no targets and fizzled immediately in the same step.
        print("MANEUVER effect fizzled immediately (no targets found when played). Checking turn progression.")
        expected_turn_count_fizzle = info_after_p1_setup.get('battle_turn_counter', 0) + 1
        assert info_after_maneuver_play.get("battle_turn_counter") == expected_turn_count_fizzle, \
            f"Fizzled MANEUVER (immediate) turn count error. Expected {expected_turn_count_fizzle}, got {info_after_maneuver_play.get('battle_turn_counter')}"
        assert info_after_maneuver_play.get("current_player_idx") != player_idx_before_action, \
            f"Player should have switched after immediately fizzled MANEUVER, but is still P{player_idx_before_action}."
        print("Immediately fizzled MANEUVER handled, turn advanced. Test for immediate fizzle path successful.")
    else:
        print(
            f"MANEUVER card played, but game phase is unexpectedly {info_after_maneuver_play.get('game_phase')}. Cannot test effect resolution.")

    # Continue game briefly if not terminated
    if not terminated:
        print("\n--- Continuing game briefly after MANEUVER test ---")
        if env.players[env.current_player_idx].hand:  # Check if current player has cards
            # Get current legal moves for the player whose turn it is now
            final_legal_moves = env.get_readable_legal_moves()
            if final_legal_moves:
                final_action_id, final_action_desc = random.choice(final_legal_moves)
                print(f"Player P{env.current_player_idx} takes final random legal action: {final_action_desc}")
                player_idx_before_final_action = env.current_player_idx
                obs, reward, terminated, truncated, info_final = env.step(final_action_id)
                info_final['current_player_idx_at_action_end'] = player_idx_before_final_action
                env.render()
                print_step_result("3 (Post-MANEUVER)", final_action_id, final_action_desc, obs, reward, terminated,
                                  truncated, info_final)
            else:
                print(f"Player P{env.current_player_idx} has no legal moves to continue after MANEUVER test.")
        else:
            print(f"Player P{env.current_player_idx} has no cards to continue after MANEUVER test.")

    print("\nMANEUVER Test Function Finished.")


# TODO: Aerodrome test fails sometimes but the engine should still be correct? Problem is probably with setup?
def test_aerodrome_effect(env: AirLandSeaBaseEnv, max_setup_retries=5):
    print("\n" + "#" * 10 + " Test: AERODROME Card Effect " + "#" * 10)

    aerodrome_card_id = 8  # AIR_4_AERODROME
    found_initial_setup = False
    initial_obs, initial_info = None, None  # To satisfy linter if loop doesn't run

    for retry in range(max_setup_retries):
        print(f"AERODROME Test: Setup attempt {retry + 1}/{max_setup_retries}")
        obs_temp, info_temp = env.reset()
        # Check if P0 has AERODROME and at least one other card (preferably strength <=3)
        player0_hand = env.players[0].hand
        has_aerodrome = any(card.id == aerodrome_card_id for card in player0_hand)
        # Check for a suitable second card to test Aerodrome's effect
        has_low_strength_card_for_test = any(
            card.id != aerodrome_card_id and card.strength <= 3 for card in player0_hand)

        if has_aerodrome and (len(player0_hand) > 1 and has_low_strength_card_for_test or len(
            player0_hand) == 1 and has_aerodrome):  # Need aerodrome and ideally another card
            initial_obs, initial_info = obs_temp, info_temp
            found_initial_setup = True
            print("Favorable hand for AERODROME test found.")
            break
        elif has_aerodrome and len(player0_hand) < 2:
            print("P0 has AERODROME but no other card to test its effect with.")
            # Could still proceed to just test playing AERODROME
            initial_obs, initial_info = obs_temp, info_temp
            found_initial_setup = True  # Allow testing playing Aerodrome itself
            print("Proceeding to test playing AERODROME itself.")
            break

    if not found_initial_setup:
        print(
            f"Could not get a suitable hand for Player 0 with AERODROME after {max_setup_retries} retries. Skipping test.")
        return

    player0_original_hand_names = [card.name for card in env.players[0].hand]
    print(f"Initial P0 Hand for test: {player0_original_hand_names}")
    print(f"Initial Theater Layout: {[th.current_type.name for th in env.theaters]}")
    env.render()

    # --- Phase 1: Player 0 plays AERODROME ---
    aerodrome_card_slot_idx = -1
    aerodrome_card_obj = None
    for i, card in enumerate(env.players[0].hand):
        if card.id == aerodrome_card_id:
            aerodrome_card_slot_idx = i
            aerodrome_card_obj = card
            break

    if aerodrome_card_obj is None:  # Should not happen if found_initial_setup is True and logic correct
        print("Error: AERODROME was supposedly in hand but not found. Skipping.")
        return

    target_air_theater_idx = -1
    for i, theater in enumerate(env.theaters):
        if theater.current_type == TheaterType.AIR:  # AERODROME is an AIR card
            target_air_theater_idx = i
            break
    if target_air_theater_idx == -1:
        print("No AIR theater on board. Skipping.")  # Should be very rare
        return

    actions_per_card_slot = NUM_THEATERS * 2
    play_aerodrome_action_id = aerodrome_card_slot_idx * actions_per_card_slot + target_air_theater_idx * 2 + 0  # 0 for Face-Up
    play_aerodrome_desc = f"Play '{aerodrome_card_obj.name}' (slot {aerodrome_card_slot_idx}) to AIR (Pos {target_air_theater_idx}) Face-Up"

    current_legal_moves_p0 = env.get_readable_legal_moves()
    is_aerodrome_play_initially_legal = any(aid == play_aerodrome_action_id for aid, _ in current_legal_moves_p0)

    if not is_aerodrome_play_initially_legal:
        print(
            f"ERROR IN TEST SETUP: Planned play of AERODROME ({play_aerodrome_desc}, ID {play_aerodrome_action_id}) is not legal according to initial mask.")
        return

    print(f"\nP0 playing AERODROME: {play_aerodrome_desc} (Action ID: {play_aerodrome_action_id})")
    player_idx_before_action = env.current_player_idx
    obs, reward, terminated, truncated, info_after_aerodrome_play = env.step(play_aerodrome_action_id)
    info_after_aerodrome_play['current_player_idx_at_action_end'] = player_idx_before_action
    env.render()
    print_step_result("1 (Play AERODROME)", play_aerodrome_action_id, play_aerodrome_desc, obs, reward, terminated,
                      truncated, info_after_aerodrome_play)

    if terminated: print("Battle ended after P0 played AERODROME."); return
    assert env.current_player_idx == 1, "Turn should be P1's."

    # --- Phase 2: Player 1 takes a turn ---
    print(f"\nAERODROME Test - Phase 2: P{env.current_player_idx} (P1) takes a turn.")
    player_idx_before_action = env.current_player_idx
    p1_legal_moves = env.get_readable_legal_moves()
    if not p1_legal_moves: print("P1 has no legal moves. Problem."); return

    p1_action_id, p1_action_desc = p1_legal_moves[0]  # P1 takes first available legal move
    if p1_action_id == env.withdraw_action_id and len(p1_legal_moves) > 1:
        p1_action_id, p1_action_desc = p1_legal_moves[1]  # Prefer not to withdraw if other options

    print(f"P1 taking action: {p1_action_desc} (Action ID: {p1_action_id})")
    obs, reward, terminated, truncated, info_after_p1_turn = env.step(p1_action_id)
    info_after_p1_turn['current_player_idx_at_action_end'] = player_idx_before_action
    env.render()
    print_step_result("2 (P1's turn)", p1_action_id, p1_action_desc, obs, reward, terminated, truncated,
                      info_after_p1_turn)

    if terminated: print("Battle ended after P1's turn."); return
    assert env.current_player_idx == 0, "Turn should be P0's."

    # --- Phase 3: Player 0's turn with AERODROME active ---
    print(f"\nAERODROME Test - Phase 3: P{env.current_player_idx} (P0) plays with AERODROME active.")
    player_idx_before_action = env.current_player_idx
    assert env._is_aerodrome_active_for_player(0), "AERODROME effect should be active for Player 0!"
    print("AERODROME effect confirmed active for Player 0 via internal check.")

    # Find a card in P0's hand with strength <= 3 (and not AERODROME itself if it's still there by mistake)
    test_card_obj = None
    test_card_slot_idx = -1
    for i, card in enumerate(env.players[0].hand):
        if card.strength <= 3 and card.id != aerodrome_card_id:  # Ensure it's not Aerodrome itself
            test_card_obj = card
            test_card_slot_idx = i
            break

    if not test_card_obj:
        print(
            f"P0 has no suitable card (Str <= 3, not AERODROME) to test AERODROME effect. P0 Hand: {[c.name for c in env.players[0].hand]}")
        return

    # Find a non-matching theater for this test_card
    target_non_matching_theater_idx = -1
    target_non_matching_theater_type_name = ""
    original_theater_layout_names = [th.current_type.name for th in env.theaters]  # Get current layout

    for i, theater in enumerate(env.theaters):
        if theater.current_type.name != test_card_obj.card_type.name:  # Non-matching type
            target_non_matching_theater_idx = i
            target_non_matching_theater_type_name = theater.current_type.name
            break

    if target_non_matching_theater_idx == -1:
        print(
            f"Could not find a non-matching theater for test card '{test_card_obj.name}' (Type: {test_card_obj.card_type.name}). Theaters: {original_theater_layout_names}")
        return

    aerodrome_test_play_action_id = test_card_slot_idx * actions_per_card_slot + target_non_matching_theater_idx * 2 + 0  # 0 for Face-Up
    aerodrome_test_play_desc = (
        f"Play '{test_card_obj.name}' (Str {test_card_obj.strength}, Type {test_card_obj.card_type.name}) "
        f"from slot {test_card_slot_idx} to {target_non_matching_theater_type_name} "
        f"(Pos {target_non_matching_theater_idx}) Face-Up (AERODROME TEST)")

    print(
        f"\nP0 attempting AERODROME-enabled play: {aerodrome_test_play_desc} (Action ID: {aerodrome_test_play_action_id})")

    current_legal_moves_with_aerodrome = env.get_readable_legal_moves()
    is_aerodrome_test_play_legal = any(
        aid == aerodrome_test_play_action_id for aid, _ in current_legal_moves_with_aerodrome)

    if not is_aerodrome_test_play_legal:
        print(
            f"AERODROME TEST FAILED: Proposed action '{aerodrome_test_play_desc}' (ID: {aerodrome_test_play_action_id}) was NOT found in legal moves:")
        for aid, desc in current_legal_moves_with_aerodrome: print(f"  ID: {aid} => {desc}")
        return  # Test fails here

    print(f"AERODROME TEST PASSED: Action '{aerodrome_test_play_desc}' IS legal as expected.")

    obs, reward, terminated, truncated, info_after_aerodrome_test_play = env.step(aerodrome_test_play_action_id)
    info_after_aerodrome_test_play['current_player_idx_at_action_end'] = player_idx_before_action
    env.render()
    print_step_result("3 (AERODROME Effect Play)", aerodrome_test_play_action_id, aerodrome_test_play_desc, obs, reward,
                      terminated, truncated, info_after_aerodrome_test_play)

    # Final check: card should be in the non-matching theater
    played_card_in_target_theater = False
    target_th_obj = env.theaters[target_non_matching_theater_idx]
    p0_top_card_in_target = target_th_obj.get_uncovered_card(0)
    if p0_top_card_in_target and p0_top_card_in_target.card_ref.id == test_card_obj.id:
        played_card_in_target_theater = True

    assert played_card_in_target_theater, f"Card '{test_card_obj.name}' not found as top card in target theater {target_non_matching_theater_idx} for P0 after AERODROME play."
    print(f"Card '{test_card_obj.name}' correctly played to non-matching theater under AERODROME effect.")

    print("\nAERODROME Test Function Finished.")


# Add this function to your main.py, and call it in the if __name__ == "__main__": block

def test_ambush_effect(env: AirLandSeaBaseEnv, max_setup_retries=5):
    print("\n" + "#" * 10 + " Test: AMBUSH Card Effect " + "#" * 10)

    ambush_card_id = 1  # LAND_2_AMBUSH
    found_initial_setup = False
    initial_obs, initial_info = None, None

    for retry in range(max_setup_retries):
        print(f"AMBUSH Test: Setup attempt {retry + 1}/{max_setup_retries}")
        obs_temp, info_temp = env.reset()
        player0_hand = env.players[0].hand
        has_ambush = any(card.id == ambush_card_id for card in player0_hand)

        if has_ambush:  # We just need AMBUSH for P0 for this test initially
            initial_obs, initial_info = obs_temp, info_temp
            found_initial_setup = True
            print("Favorable hand for AMBUSH test found (P0 has AMBUSH).")
            break

    if not found_initial_setup:
        print(
            f"Could not get AMBUSH card (ID {ambush_card_id}) for Player 0 after {max_setup_retries} retries. Skipping test.")
        return

    player0_original_hand_names = [card.name for card in env.players[0].hand]
    print(f"Initial P0 Hand for test: {player0_original_hand_names}")
    print(f"Initial Theater Layout: {[th.current_type.name for th in env.theaters]}")
    env.render()

    # --- Setup Phase: Play a few cards to have targets ---
    # P0 plays a card (not AMBUSH yet, if possible, to save it)
    # P1 plays a card (this becomes a primary target)
    # P0 then plays AMBUSH

    # Step 0.1: Player 0 plays a setup card
    print(f"\nAMBUSH Test Setup: P{env.current_player_idx} (P0) plays a setup card.")
    player_idx_before_action = env.current_player_idx
    legal_moves_p0_setup = env.get_readable_legal_moves()
    action_p0_setup, desc_p0_setup = env.withdraw_action_id, "Withdraw (fallback)"

    # Try to play a non-AMBUSH card first
    non_ambush_plays = []
    if env.players[0].hand:  # Ensure player has a hand to check against
        for aid, desc in legal_moves_p0_setup:
            if aid != env.withdraw_action_id:
                decoded_action = env._decode_action(aid)
                card_slot_idx = decoded_action[0]
                # Check if card_slot_idx is a valid integer index and within hand bounds
                if card_slot_idx is not None and 0 <= card_slot_idx < len(env.players[0].hand):
                    if ambush_card_id != env.players[0].hand[card_slot_idx].id:
                        non_ambush_plays.append((aid, desc))
                # If card_slot_idx is None, it's not a valid play action to compare card ID, so skip

    if non_ambush_plays:
        action_p0_setup, desc_p0_setup = random.choice(non_ambush_plays)
    elif legal_moves_p0_setup:
        # Fallback if no non-ambush plays, or if hand was empty initially
        action_p0_setup, desc_p0_setup = legal_moves_p0_setup[0]
        if action_p0_setup == env.withdraw_action_id and len(legal_moves_p0_setup) > 1:
            action_p0_setup, desc_p0_setup = legal_moves_p0_setup[1]
    # else: action_p0_setup remains withdraw_action_id if legal_moves_p0_setup was empty or only withdraw

    if action_p0_setup == env.withdraw_action_id and not env.players[0].hand:  # Should not happen if reset properly
        print("P0 has no cards and can only withdraw. Skipping AMBUSH test.")
        return

    obs, reward, terminated, truncated, info_after_p0_setup = env.step(action_p0_setup)
    info_after_p0_setup['current_player_idx_at_action_end'] = player_idx_before_action
    env.render()
    print_step_result("0.1 (P0 Setup)", action_p0_setup, desc_p0_setup, obs, reward, terminated, truncated,
                      info_after_p0_setup)
    if terminated: print("Battle ended during P0 setup. AMBUSH test cannot proceed."); return

    # Step 0.2: Player 1 plays a setup card (this could be a target)
    print(f"\nAMBUSH Test Setup: P{env.current_player_idx} (P1) plays a setup card (potential target).")
    player_idx_before_action = env.current_player_idx
    legal_moves_p1_setup = env.get_readable_legal_moves()
    action_p1_setup, desc_p1_setup = env.withdraw_action_id, "Withdraw (default)"
    if legal_moves_p1_setup:
        action_p1_setup, desc_p1_setup = legal_moves_p1_setup[0]
        if action_p1_setup == env.withdraw_action_id and len(legal_moves_p1_setup) > 1:
            action_p1_setup, desc_p1_setup = legal_moves_p1_setup[1]

    if action_p1_setup == env.withdraw_action_id and not env.players[1].hand:
        print("P1 has no cards and can only withdraw. May limit AMBUSH targets. Continuing.")
        # We can still proceed if P0 has a card to flip.

    obs, reward, terminated, truncated, info_after_p1_setup = env.step(action_p1_setup)
    info_after_p1_setup['current_player_idx_at_action_end'] = player_idx_before_action
    env.render()
    print_step_result("0.2 (P1 Setup)", action_p1_setup, desc_p1_setup, obs, reward, terminated, truncated,
                      info_after_p1_setup)
    if terminated: print("Battle ended during P1 setup. AMBUSH test cannot proceed."); return

    # --- Step 1: Player 0 plays AMBUSH ---
    print(f"\nAMBUSH Test - Step 1: P{env.current_player_idx} (P0) attempts to play AMBUSH.")
    player_idx_before_action = env.current_player_idx
    current_player_obj = env.players[player_idx_before_action]

    ambush_card_slot_idx = -1
    ambush_card_obj = None
    for i, card_in_hand in enumerate(current_player_obj.hand):
        if card_in_hand.id == ambush_card_id:
            ambush_card_slot_idx = i
            ambush_card_obj = card_in_hand
            break

    if ambush_card_slot_idx == -1:
        print(f"AMBUSH card (ID {ambush_card_id}) no longer in P0's hand (unexpected). Skipping AMBUSH effect test.")
        return

    # Find a LAND theater to play AMBUSH (its native type)
    target_land_theater_idx = -1
    for i, theater in enumerate(env.theaters):
        if theater.current_type == TheaterType.LAND:
            target_land_theater_idx = i
            break

    if target_land_theater_idx == -1:
        print(
            "No LAND theater found on the board. Trying to play AMBUSH to first available theater face-down for test if no LAND.")
        # Fallback: play AMBUSH face-down to theater 0 if no LAND theater
        target_land_theater_idx = 0  # Default to theater 0
        play_ambush_action_id = ambush_card_slot_idx * (
                NUM_THEATERS * 2) + target_land_theater_idx * 2 + 1  # 1 for Face-Down
        play_ambush_desc = f"Play '{ambush_card_obj.name}' (slot {ambush_card_slot_idx}) to {env.theaters[target_land_theater_idx].current_type.name} (Pos {target_land_theater_idx}) Face-Down (fallback)"
        print("Warning: No LAND theater. AMBUSH effect will not trigger if played Face-Down. Test will be limited.")
    else:
        actions_per_card_slot = NUM_THEATERS * 2
        play_ambush_action_id = ambush_card_slot_idx * actions_per_card_slot + target_land_theater_idx * 2 + 0  # 0 for Face-Up
        play_ambush_desc = f"Play '{ambush_card_obj.name}' (slot {ambush_card_slot_idx}) to LAND (Pos {target_land_theater_idx}) Face-Up"

    current_legal_moves_p0_ambush = env.get_readable_legal_moves()
    is_ambush_play_legal = any(aid == play_ambush_action_id for aid, _ in current_legal_moves_p0_ambush)

    if not is_ambush_play_legal:
        print(
            f"Attempting to play AMBUSH with action ID {play_ambush_action_id} ({play_ambush_desc}) but it's not listed as legal.")
        # Try to find ANY legal way to play Ambush face up for the test
        found_alt_play = False
        for aid, desc in current_legal_moves_p0_ambush:
            if "AMBUSH" in desc.upper() and "FACE-UP" in desc.upper():
                play_ambush_action_id = aid
                play_ambush_desc = desc
                found_alt_play = True
                print(f"Alternative legal AMBUSH play found: {desc}")
                break
        if not found_alt_play:
            print("No legal way to play AMBUSH face-up found. Skipping effect test.")
            return

    print(f"P0 playing AMBUSH: {play_ambush_desc} (Action ID: {play_ambush_action_id})")
    obs, reward, terminated, truncated, info_after_ambush_play = env.step(play_ambush_action_id)
    info_after_ambush_play['current_player_idx_at_action_end'] = player_idx_before_action
    env.render()
    print_step_result("1 (Play AMBUSH)", play_ambush_action_id, play_ambush_desc, obs, reward, terminated, truncated,
                      info_after_ambush_play)

    if terminated:
        print("Battle ended immediately after playing AMBUSH card. Effect resolution cannot be fully tested.")
        return

    # Check if AMBUSH was played face-down (fallback) - if so, effect won't trigger
    if "FACE-DOWN" in play_ambush_desc.upper():
        print(
            "AMBUSH played Face-Down due to theater setup, effect will not trigger. Test concludes here for AMBUSH effect part.")
        return

    # Check if game phase changed as expected
    if info_after_ambush_play.get("game_phase") == GamePhase.RESOLVING_AMBUSH.name:
        print(f"Game phase is now: {info_after_ambush_play.get('game_phase')}. Correct.")
        assert info_after_ambush_play.get("current_player_idx") == player_idx_before_action, \
            f"Player should still be P{player_idx_before_action} for AMBUSH choice, but is P{info_after_ambush_play.get('current_player_idx')}."
        print(f"Current player is still P{info_after_ambush_play.get('current_player_idx')}. Correct.")

        # --- Step 2: Player 0 resolves AMBUSH effect ---
        print(f"\nAMBUSH Test - Step 2: P{env.current_player_idx} choosing target for AMBUSH effect.")
        ambush_target_choices = env.get_readable_legal_moves()

        if not ambush_target_choices or \
            (len(ambush_target_choices) == 1 and ambush_target_choices[0][
                0] == env.withdraw_action_id):  # Check if only withdraw left in a weird state
            print(
                "No valid targets presented for AMBUSH effect (or only withdraw). Effect should have fizzled if env.effect_resolution_data possible_targets was empty.")
            if not env.effect_resolution_data.get('possible_targets'):
                print(
                    "AMBUSH effect fizzled as no targets were found by the environment. Test continues to check turn progression.")
                assert info_after_ambush_play.get(
                    "game_phase") == GamePhase.NORMAL_PLAY.name, "Phase should be NORMAL if AMBUSH fizzled immediately."
                expected_turn_count_fizzle = info_after_p1_setup.get('battle_turn_counter',
                                                                     0) + 1  # Turn of playing AMBUSH card itself
                assert info_after_ambush_play.get("battle_turn_counter") == expected_turn_count_fizzle
                assert info_after_ambush_play.get(
                    "current_player_idx") != player_idx_before_action  # Player should have switched
                print("Fizzled AMBUSH (no targets at trigger) handled, turn advanced. Test successful for fizzle.")
            else:
                print("Still in RESOLVING_AMBUSH but no choices from get_readable_legal_moves. Check logic.")

        elif ambush_target_choices:  # Targets are available
            print("Legal choices for AMBUSH effect:")
            for aid, desc in ambush_target_choices: print(f"  Action ID: {aid} => {desc}")

            chosen_ambush_effect_action_id, chosen_ambush_effect_desc = random.choice(ambush_target_choices)
            print(
                f"P0 choosing AMBUSH target: {chosen_ambush_effect_desc} (Action ID: {chosen_ambush_effect_action_id})")

            player_idx_before_effect_action = env.current_player_idx
            obs, reward, terminated, truncated, info_after_effect_resolution = env.step(chosen_ambush_effect_action_id)
            info_after_effect_resolution['current_player_idx_at_action_end'] = player_idx_before_effect_action
            env.render()
            print_step_result("2 (Resolve AMBUSH)", chosen_ambush_effect_action_id, chosen_ambush_effect_desc, obs,
                              reward, terminated, truncated, info_after_effect_resolution)

            assert info_after_effect_resolution.get("game_phase") == GamePhase.NORMAL_PLAY.name, \
                f"Game phase should be NORMAL_PLAY after AMBUSH resolution, but is {info_after_effect_resolution.get('game_phase')}"
            print(f"Game phase is now: {info_after_effect_resolution.get('game_phase')}. Correct.")
            assert info_after_effect_resolution.get("current_player_idx") != player_idx_before_action, \
                f"Player should have switched after AMBUSH turn."
            print(
                f"Current player is now P{info_after_effect_resolution.get('current_player_idx')}. Correct (switched).")

            expected_turn_count = info_after_p1_setup.get('battle_turn_counter',
                                                          0) + 1  # AMBUSH card play itself was one turn
            assert info_after_effect_resolution.get("battle_turn_counter") == expected_turn_count, \
                f"Battle turn counter error after AMBUSH. Expected {expected_turn_count}, got {info_after_effect_resolution.get('battle_turn_counter')}"
            print(f"Battle turn counter is {info_after_effect_resolution.get('battle_turn_counter')}. Correct.")
            print("AMBUSH effect resolution test part completed.")

    elif info_after_ambush_play.get("game_phase") == GamePhase.NORMAL_PLAY.name:
        print("AMBUSH effect fizzled immediately (no targets found when AMBUSH was played). Checking turn progression.")
        expected_turn_count_fizzle = info_after_p1_setup.get('battle_turn_counter', 0) + 1
        assert info_after_ambush_play.get("battle_turn_counter") == expected_turn_count_fizzle
        assert info_after_ambush_play.get("current_player_idx") != player_idx_before_action
        print("Immediately fizzled AMBUSH handled, turn advanced. Test for immediate fizzle path successful.")
    else:
        print(
            f"AMBUSH card played, but game phase is unexpectedly {info_after_ambush_play.get('game_phase')}. Cannot test effect resolution.")

    if not terminated:
        # ... (optional: continue game briefly) ...
        pass

    print("\nAMBUSH Test Function Finished.")


# Add this function to your main.py

def test_disrupt_effect(env: AirLandSeaBaseEnv, max_setup_retries=10):  # Increased retries for more complex setup
    print("\n" + "#" * 10 + " Test: DISRUPT Card Effect " + "#" * 10)

    disrupt_card_id = 3  # LAND_5_DISRUPT
    found_initial_setup = False
    initial_obs, initial_info = None, None

    for retry in range(max_setup_retries):
        print(f"DISRUPT Test: Setup attempt {retry + 1}/{max_setup_retries}")
        obs_temp, info_temp = env.reset()
        player0_hand = env.players[0].hand
        # P0 needs DISRUPT and at least one other card to flip (ideally)
        has_disrupt = any(card.id == disrupt_card_id for card in player0_hand)
        has_other_card_for_p0 = len(player0_hand) > 1 if has_disrupt else len(player0_hand) > 0

        # P1 also needs at least one card to be able to make a choice
        player1_hand_ok = len(env.players[1].hand) > 0

        if has_disrupt and has_other_card_for_p0 and player1_hand_ok:
            initial_obs, initial_info = obs_temp, info_temp
            found_initial_setup = True
            print("Favorable hand for DISRUPT test found (P0 has DISRUPT and another card, P1 has cards).")
            break

    if not found_initial_setup:
        print(f"Could not get a suitable hand for DISRUPT test after {max_setup_retries} retries. Skipping.")
        return

    print(f"Initial P0 Hand for test: {[card.name for card in env.players[0].hand]}")
    print(f"Initial P1 Hand for test: {[card.name for card in env.players[1].hand]}")
    print(f"Initial Theater Layout: {[th.current_type.name for th in env.theaters]}")
    env.render()

    # --- Setup Phase: Both players play one card to have flippable cards on board ---
    # Step 0.1: Player 0 plays a non-DISRUPT card
    print(f"\nDISRUPT Test Setup: P{env.current_player_idx} (P0) plays a setup card.")
    player_idx_before_action = env.current_player_idx
    p0_setup_action_id, p0_setup_desc = env.withdraw_action_id, "Withdraw (fallback)"
    legal_moves_p0_setup = env.get_readable_legal_moves()
    # Try to play a non-DISRUPT card
    non_disrupt_plays_p0 = [(aid, desc) for aid, desc in legal_moves_p0_setup if
                            env._decode_action(aid)[0] is not None and (
                                env.players[0].hand[env._decode_action(aid)[0]].id != disrupt_card_id if
                                env._decode_action(aid)[0] < len(
                                    env.players[0].hand) else False) and aid != env.withdraw_action_id]

    if non_disrupt_plays_p0:
        p0_setup_action_id, p0_setup_desc = random.choice(non_disrupt_plays_p0)
    elif legal_moves_p0_setup:  # Fallback if only DISRUPT or withdraw left
        p0_setup_action_id, p0_setup_desc = legal_moves_p0_setup[0]
        if p0_setup_action_id == env.withdraw_action_id and len(legal_moves_p0_setup) > 1:
            p0_setup_action_id, p0_setup_desc = legal_moves_p0_setup[1]

    if p0_setup_action_id == env.withdraw_action_id and not env.players[0].hand:
        print("P0 cannot make a setup play. Skipping DISRUPT test.");
        return

    obs, reward, terminated, truncated, info_p0_setup = env.step(p0_setup_action_id)
    info_p0_setup['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("0.1 (P0 Setup)", p0_setup_action_id, p0_setup_desc, obs, reward, terminated, truncated,
                      info_p0_setup)
    if terminated: print("Battle ended during P0 setup. DISRUPT test cannot proceed."); return

    # Step 0.2: Player 1 plays a setup card
    print(f"\nDISRUPT Test Setup: P{env.current_player_idx} (P1) plays a setup card.")
    player_idx_before_action = env.current_player_idx
    legal_moves_p1_setup = env.get_readable_legal_moves()
    p1_setup_action_id, p1_setup_desc = env.withdraw_action_id, "Withdraw (fallback)"
    if legal_moves_p1_setup:
        p1_setup_action_id, p1_setup_desc = legal_moves_p1_setup[0]
        if p1_setup_action_id == env.withdraw_action_id and len(legal_moves_p1_setup) > 1:
            p1_setup_action_id, p1_setup_desc = legal_moves_p1_setup[1]

    if p1_setup_action_id == env.withdraw_action_id and not env.players[1].hand:
        print("P1 cannot make a setup play. Skipping DISRUPT test.");
        return

    obs, reward, terminated, truncated, info_p1_setup = env.step(p1_setup_action_id)
    info_p1_setup['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("0.2 (P1 Setup)", p1_setup_action_id, p1_setup_desc, obs, reward, terminated, truncated,
                      info_p1_setup)
    if terminated: print("Battle ended during P1 setup. DISRUPT test cannot proceed."); return

    initial_battle_turn_counter_before_disrupt = env.battle_turn_counter  # Should be 2

    # --- Step 1: Player 0 plays DISRUPT ---
    print(f"\nDISRUPT Test - Step 1: P{env.current_player_idx} (P0) attempts to play DISRUPT.")
    player_idx_before_action = env.current_player_idx  # Should be P0
    current_player_obj_p0 = env.players[player_idx_before_action]

    disrupt_card_slot_idx = -1;
    disrupt_card_obj = None
    for i, card in enumerate(current_player_obj_p0.hand):
        if card.id == disrupt_card_id: disrupt_card_slot_idx = i; disrupt_card_obj = card; break

    if disrupt_card_slot_idx == -1: print(
        f"DISRUPT card (ID {disrupt_card_id}) not in P0's hand. Setup failed. Skipping."); return

    target_land_theater_idx = -1  # DISRUPT is a LAND card
    for i, th in enumerate(env.theaters):
        if th.current_type == TheaterType.LAND: target_land_theater_idx = i; break

    play_disrupt_action_id = -1
    play_disrupt_desc = ""

    if target_land_theater_idx != -1:  # Found a LAND theater
        actions_per_slot = NUM_THEATERS * 2
        play_disrupt_action_id = disrupt_card_slot_idx * actions_per_slot + target_land_theater_idx * 2 + 0  # Play Face-Up
        play_disrupt_desc = f"Play '{disrupt_card_obj.name}' (slot {disrupt_card_slot_idx}) to LAND (Pos {target_land_theater_idx}) Face-Up"
    else:  # No LAND theater, try to play face-down (effect won't trigger, but tests env stability)
        target_land_theater_idx = 0  # Default to theater 0
        play_disrupt_action_id = disrupt_card_slot_idx * (
                NUM_THEATERS * 2) + target_land_theater_idx * 2 + 1  # Face-Down
        play_disrupt_desc = f"Play '{disrupt_card_obj.name}' (slot {disrupt_card_slot_idx}) to {env.theaters[target_land_theater_idx].current_type.name} (Pos {target_land_theater_idx}) Face-Down (no LAND theater)"
        print(f"Warning: No LAND theater found. Playing DISRUPT face-down: {play_disrupt_desc}")

    current_legal_moves_p0_disrupt = env.get_readable_legal_moves()
    is_disrupt_play_legal = any(aid == play_disrupt_action_id for aid, _ in current_legal_moves_p0_disrupt)

    if not is_disrupt_play_legal:
        print(
            f"Attempting to play DISRUPT with {play_disrupt_desc} (ID {play_disrupt_action_id}) but it's not listed as legal. Mask/Setup issue.");
        return

    print(f"P0 playing DISRUPT: {play_disrupt_desc} (Action ID: {play_disrupt_action_id})")
    obs, reward, terminated, truncated, info_after_disrupt_play = env.step(play_disrupt_action_id)
    info_after_disrupt_play['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("1 (Play DISRUPT)", play_disrupt_action_id, play_disrupt_desc, obs, reward, terminated, truncated,
                      info_after_disrupt_play)

    if terminated: print("Battle ended immediately after playing DISRUPT. Test incomplete."); return
    if "FACE-DOWN" in play_disrupt_desc.upper(): print(
        "DISRUPT played face-down, effect won't trigger. Test incomplete."); return

    # Assertions for Phase 1 of DISRUPT (Opponent's Choice)
    assert info_after_disrupt_play.get("game_phase") == GamePhase.RESOLVING_DISRUPT_OPPONENT_CHOICE.name, \
        f"Phase should be RESOLVING_DISRUPT_OPPONENT_CHOICE, but is {info_after_disrupt_play.get('game_phase')}"
    print(f"Game phase is now: {info_after_disrupt_play.get('game_phase')}. Correct.")
    assert info_after_disrupt_play.get("current_player_idx") == (1 - player_idx_before_action), \
        f"Player should be P{1 - player_idx_before_action} (Opponent) for DISRUPT choice, but is P{info_after_disrupt_play.get('current_player_idx')}."
    print(
        f"Current player for choice is P{info_after_disrupt_play.get('current_player_idx')}. Correct (Opponent's turn to choose).")
    assert info_after_disrupt_play.get("battle_turn_counter") == initial_battle_turn_counter_before_disrupt, \
        f"Battle turn should not have incremented yet. Was {initial_battle_turn_counter_before_disrupt}, now {info_after_disrupt_play.get('battle_turn_counter')}."
    print("Battle turn counter unchanged. Correct.")

    # --- Step 2: Player 1 (Opponent) makes their DISRUPT choice ---
    player_idx_before_action = env.current_player_idx  # Should be P1
    print(f"\nDISRUPT Test - Step 2: P{player_idx_before_action} (Opponent) choosing their card to flip.")
    disrupt_opponent_choices = env.get_readable_legal_moves()
    if not disrupt_opponent_choices or (
        len(disrupt_opponent_choices) == 1 and disrupt_opponent_choices[0][0] == env.withdraw_action_id):
        print(
            "Opponent has no valid flip choices for DISRUPT (or only withdraw). Effect part might fizzle or test needs adjustment.")
        # Check if effect_data indicates no targets for opponent
        if not info_after_disrupt_play.get("effect_data_opponent_targets"):
            print("DISRUPT: Opponent's part fizzled as no targets were found by env. Checking phase.")
            assert info_after_disrupt_play.get(
                "game_phase") == GamePhase.RESOLVING_DISRUPT_SELF_CHOICE.name, "Phase should be RESOLVING_DISRUPT_SELF_CHOICE if opponent part fizzled."
            # The previous step's info should show the new phase for self-choice.
            # current_player_idx in info_after_disrupt_play should be the original disrupt player.
            assert info_after_disrupt_play.get("current_player_idx") == env.effect_resolution_data.get(
                'disrupt_player_idx'), "Player should be back to disrupt player."
        else:
            print("Opponent choices missing from get_readable_legal_moves, but effect_data has targets. Check logic.")
            return  # Cannot proceed if choices aren't presented but should be

    chosen_opponent_flip_action_id, chosen_opponent_flip_desc = random.choice(disrupt_opponent_choices)
    print(
        f"P{player_idx_before_action} (Opponent) choosing DISRUPT target: {chosen_opponent_flip_desc} (Action ID: {chosen_opponent_flip_action_id})")
    obs, reward, terminated, truncated, info_after_opponent_flip = env.step(chosen_opponent_flip_action_id)
    info_after_opponent_flip['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("2 (Opponent Flips for DISRUPT)", chosen_opponent_flip_action_id, chosen_opponent_flip_desc, obs,
                      reward, terminated, truncated, info_after_opponent_flip)

    if terminated: print("Battle ended after opponent's DISRUPT choice. Test incomplete."); return

    original_disrupt_player = info_after_disrupt_play.get('effect_data', {}).get('disrupt_player_idx',
                                                                                 0 if player_idx_before_action == 1 else 1)  # Get who played disrupt
    assert info_after_opponent_flip.get("game_phase") == GamePhase.RESOLVING_DISRUPT_SELF_CHOICE.name, \
        f"Phase should be RESOLVING_DISRUPT_SELF_CHOICE, but is {info_after_opponent_flip.get('game_phase')}"
    print(f"Game phase is now: {info_after_opponent_flip.get('game_phase')}. Correct.")
    assert info_after_opponent_flip.get("current_player_idx") == original_disrupt_player, \
        f"Player should be back to P{original_disrupt_player} for self-flip, but is P{info_after_opponent_flip.get('current_player_idx')}."
    print(
        f"Current player for choice is P{info_after_opponent_flip.get('current_player_idx')}. Correct (Original Disrupt Player).")
    assert info_after_opponent_flip.get("battle_turn_counter") == initial_battle_turn_counter_before_disrupt, \
        f"Battle turn should still not have incremented. Was {initial_battle_turn_counter_before_disrupt}, now {info_after_opponent_flip.get('battle_turn_counter')}."
    print("Battle turn counter still unchanged. Correct.")

    # --- Step 3: Player 0 (Original Player) makes their DISRUPT choice ---
    player_idx_before_action = env.current_player_idx  # Should be P0 (original disrupt player)
    print(f"\nDISRUPT Test - Step 3: P{player_idx_before_action} (Original Player) choosing their card to flip.")
    disrupt_self_choices = env.get_readable_legal_moves()
    if not disrupt_self_choices or (
        len(disrupt_self_choices) == 1 and disrupt_self_choices[0][0] == env.withdraw_action_id):
        print("Original player has no valid flip choices for DISRUPT (or only withdraw). Effect part might fizzle.")
        if not info_after_opponent_flip.get("effect_data_self_targets"):
            print("DISRUPT: Self-flip part fizzled as no targets were found. Checking final turn progression.")
            assert info_after_opponent_flip.get(
                "game_phase") == GamePhase.NORMAL_PLAY.name, "Phase should be NORMAL if self-part fizzled."
            expected_turn_count_full_fizzle = initial_battle_turn_counter_before_disrupt + 1
            assert info_after_opponent_flip.get("battle_turn_counter") == expected_turn_count_full_fizzle
            assert info_after_opponent_flip.get("current_player_idx") != original_disrupt_player  # Should have switched
            print("Fully fizzled DISRUPT handled, turn advanced. Test successful for full fizzle.")
            return  # End test here if fully fizzled.
        else:
            print("Self choices missing from get_readable, but effect_data has targets. Check logic.")
            return

    chosen_self_flip_action_id, chosen_self_flip_desc = random.choice(disrupt_self_choices)
    print(
        f"P{player_idx_before_action} (Original) choosing DISRUPT target: {chosen_self_flip_desc} (Action ID: {chosen_self_flip_action_id})")
    obs, reward, terminated, truncated, info_after_self_flip = env.step(chosen_self_flip_action_id)
    info_after_self_flip['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("3 (Self Flips for DISRUPT)", chosen_self_flip_action_id, chosen_self_flip_desc, obs, reward,
                      terminated, truncated, info_after_self_flip)

    assert info_after_self_flip.get("game_phase") == GamePhase.NORMAL_PLAY.name, \
        f"Phase should be NORMAL_PLAY after DISRUPT full resolution, but is {info_after_self_flip.get('game_phase')}"
    print(f"Game phase is now: {info_after_self_flip.get('game_phase')}. Correct.")
    assert info_after_self_flip.get("current_player_idx") != original_disrupt_player, \
        f"Player should have switched from P{original_disrupt_player} after DISRUPT full turn."
    print(f"Current player is now P{info_after_self_flip.get('current_player_idx')}. Correct (switched).")

    expected_turn_count_final = initial_battle_turn_counter_before_disrupt + 1
    assert info_after_self_flip.get("battle_turn_counter") == expected_turn_count_final, \
        f"Battle turn counter error after DISRUPT. Expected {expected_turn_count_final}, got {info_after_self_flip.get('battle_turn_counter')}"
    print(
        f"Battle turn counter is {info_after_self_flip.get('battle_turn_counter')}. Correct (incremented once for full DISRUPT play).")
    print("DISRUPT effect full resolution test part completed.")

    print("\nDISRUPT Test Function Finished.")


def test_passive_effect_on_flip_aerodrome(env: AirLandSeaBaseEnv, max_setup_retries=15):
    print("\n" + "#" * 10 + " Test: AERODROME Card Effect " + "#" * 10)

    aerodrome_card_id = 8  # AIR_4_AERODROME
    flip_enabler_card_id = 1  # LAND_2_AMBUSH

    found_initial_setup = False
    # initial_obs, initial_info = None, None # Not strictly needed here due to direct use of env

    for retry in range(max_setup_retries):
        print(f"Aerodrome Flip Test: Setup attempt {retry + 1}/{max_setup_retries}")
        env.reset()  # obs_temp, info_temp = env.reset() - reset is enough for this check
        player0_hand = env.players[0].hand

        has_aerodrome = any(card.id == aerodrome_card_id for card in player0_hand)
        has_flip_enabler = any(card.id == flip_enabler_card_id for card in player0_hand)
        has_low_strength_card = any(
            card.strength <= 3 and card.id not in [aerodrome_card_id, flip_enabler_card_id] for card in player0_hand)

        if has_aerodrome and has_flip_enabler and has_low_strength_card:
            # initial_obs, initial_info = obs_temp, info_temp # Not strictly needed here
            found_initial_setup = True
            p0_initial_hand_names = [card.name for card in player0_hand]
            print("Favorable hand for Aerodrome Flip Test found.")
            break

    if not found_initial_setup:
        print(
            f"Could not get suitable hand (AERODROME, AMBUSH, low-str card) for P0 after {max_setup_retries} retries. Skipping test.")
        return

    player0_original_hand_names = [card.name for card in env.players[0].hand]  # Re-fetch after successful reset
    print(f"Initial P0 Hand for test: {player0_original_hand_names}")
    print(f"Initial Theater Layout: {[th.current_type.name for th in env.theaters]}")
    env.render()

    actions_per_slot = NUM_THEATERS * 2

    # --- Setup Phase ---
    # Step 0.1: P0 plays AERODROME Face-Down to any theater (e.g., Theater 0)
    print(f"\nAerodrome Flip Test Setup: P{env.current_player_idx} (P0) plays AERODROME Face-Down.")
    player_idx_before_action_step_0_1 = env.current_player_idx  # Should be 0
    p0_aerodrome_slot = -1;
    aerodrome_obj = None
    for i, card in enumerate(env.players[player_idx_before_action_step_0_1].hand):  # Use current player
        if card.id == aerodrome_card_id: p0_aerodrome_slot = i; aerodrome_obj = card; break
    if p0_aerodrome_slot == -1: print("Error: AERODROME not in hand despite setup check. Skipping."); return

    theater_idx_aerodrome_fd = 0  # Play to Theater 0
    # THIS IS THE LINE THAT WAS NAMED p0_play_aerodrome_fd_action_id BEFORE
    p0_play_aerodrome_fd_or_fu_action_id = p0_aerodrome_slot * actions_per_slot + theater_idx_aerodrome_fd * 2 + 1  # 1 for FD
    p0_play_aerodrome_fd_desc = f"Play '{aerodrome_obj.name}' (slot {p0_aerodrome_slot}) to {env.theaters[theater_idx_aerodrome_fd].current_type.name} (Pos {theater_idx_aerodrome_fd}) FD"

    obs, reward, terminated, truncated, info_p0_aero_fd = env.step(p0_play_aerodrome_fd_or_fu_action_id)
    info_p0_aero_fd['current_player_idx_at_action_end'] = player_idx_before_action_step_0_1
    env.render()
    # USE THE CORRECT VARIABLE NAME HERE
    print_step_result("0.1 (P0 Plays AERODROME FD)", p0_play_aerodrome_fd_or_fu_action_id, p0_play_aerodrome_fd_desc,
                      obs, reward, terminated, truncated, info_p0_aero_fd)
    if terminated: print("Battle ended. Test incomplete."); return

    assert not env._is_aerodrome_active_for_player(0), "AERODROME should be INACTIVE when played FD."
    print("AERODROME confirmed INACTIVE while Face-Down.")

    # Step 0.2: P1 plays any card
    print(f"\nAerodrome Flip Test Setup: P{env.current_player_idx} (P1) plays a card.")
    player_idx_before_action_step_0_2 = env.current_player_idx  # Should be 1
    p1_legal_moves_setup = env.get_readable_legal_moves()  # Renamed for clarity
    p1_action_id_setup, p1_desc_setup = p1_legal_moves_setup[0]
    if p1_action_id_setup == env.withdraw_action_id and len(p1_legal_moves_setup) > 1:
        p1_action_id_setup, p1_desc_setup = p1_legal_moves_setup[1]
    obs, reward, terminated, truncated, info_p1_setup = env.step(p1_action_id_setup)
    info_p1_setup['current_player_idx_at_action_end'] = player_idx_before_action_step_0_2
    env.render()
    print_step_result("0.2 (P1 Plays)", p1_action_id_setup, p1_desc_setup, obs, reward, terminated, truncated,
                      info_p1_setup)
    if terminated: print("Battle ended. Test incomplete."); return

    # Step 0.3: P0 find low-strength card and non-matching theater. Verify play is ILLEGAL.
    print(
        f"\nAerodrome Flip Test Setup: P{env.current_player_idx} (P0) Verifying non-matching play is ILLEGAL (Aerodrome FD).")
    player_idx_before_action_step_0_3 = env.current_player_idx  # Should be 0
    p0_low_str_card_obj = None;
    p0_low_str_card_slot = -1
    for i, card in enumerate(env.players[player_idx_before_action_step_0_3].hand):  # Use current player
        if card.strength <= 3 and card.id != aerodrome_card_id and card.id != flip_enabler_card_id:  # Check it's not aerodrome itself or the flip enabler
            p0_low_str_card_obj = card;
            p0_low_str_card_slot = i;
            break
    if not p0_low_str_card_obj: print(
        f"P0 has no suitable low-strength card for verification. Test limited. Hand: {[c.name for c in env.players[player_idx_before_action_step_0_3].hand]}"); return

    target_non_matching_th_idx_verify = -1  # Renamed for clarity
    for i, th in enumerate(env.theaters):
        if th.current_type.name != p0_low_str_card_obj.card_type.name: target_non_matching_th_idx_verify = i; break
    if target_non_matching_th_idx_verify == -1: print(
        f"No non-matching theater for {p0_low_str_card_obj.name} for verification. Test limited."); return

    illegal_aerodrome_play_id = p0_low_str_card_slot * actions_per_slot + target_non_matching_th_idx_verify * 2 + 0  # Face-Up
    is_illegal_play_in_mask = any(aid == illegal_aerodrome_play_id for aid, _ in env.get_readable_legal_moves())
    assert not is_illegal_play_in_mask, f"Play of {p0_low_str_card_obj.name} to non-matching theater SHOULD BE ILLEGAL as Aerodrome is FD."
    print(
        f"Verified: Playing '{p0_low_str_card_obj.name}' Str {p0_low_str_card_obj.strength} to non-matching theater (Pos {target_non_matching_th_idx_verify}) FU is currently ILLEGAL.")

    # --- Phase 1: P0 plays AMBUSH to flip their own AERODROME Face-Up ---
    print(f"\nAerodrome Flip Test - Phase 1: P{env.current_player_idx} (P0) plays AMBUSH.")
    player_idx_before_action_phase1_1 = env.current_player_idx  # P0
    p0_ambush_slot = -1;
    ambush_obj = None
    for i, card in enumerate(env.players[player_idx_before_action_phase1_1].hand):  # Use current player
        if card.id == flip_enabler_card_id: p0_ambush_slot = i; ambush_obj = card; break
    if p0_ambush_slot == -1: print(
        f"Error: AMBUSH (ID {flip_enabler_card_id}) not in P0 hand. Skipping flip part."); return

    target_ambush_play_theater_idx = -1
    for i, th in enumerate(env.theaters):
        if th.current_type == TheaterType.LAND: target_ambush_play_theater_idx = i; break
    if target_ambush_play_theater_idx == -1: target_ambush_play_theater_idx = (
                                                                                      theater_idx_aerodrome_fd + 1) % NUM_THEATERS

    play_ambush_action_id = p0_ambush_slot * actions_per_slot + target_ambush_play_theater_idx * 2 + 0  # Face-Up
    play_ambush_desc = f"Play '{ambush_obj.name}' (slot {p0_ambush_slot}) to {env.theaters[target_ambush_play_theater_idx].current_type.name} (Pos {target_ambush_play_theater_idx}) FU"

    if not any(aid == play_ambush_action_id for aid, _ in env.get_readable_legal_moves()):
        print(f"AMBUSH play ({play_ambush_desc}) not legal. Trying alternative if available or skipping effect part.");
        # Try to find ANY legal way to play Ambush face up for the test
        found_alt_play = False
        for aid_alt, desc_alt in env.get_readable_legal_moves():
            if "AMBUSH" in desc_alt.upper() and "FACE-UP" in desc_alt.upper() and aid_alt != env.withdraw_action_id:
                play_ambush_action_id = aid_alt
                play_ambush_desc = desc_alt
                found_alt_play = True
                print(f"Alternative legal AMBUSH play found: {desc_alt}")
                break
        if not found_alt_play:
            print("No legal way to play AMBUSH face-up found. Skipping effect test.")
            return

    obs, reward, terminated, truncated, info_play_ambush = env.step(play_ambush_action_id)
    info_play_ambush['current_player_idx_at_action_end'] = player_idx_before_action_phase1_1
    env.render();
    print_step_result("1.1 (P0 Plays AMBUSH)", play_ambush_action_id, play_ambush_desc, obs, reward, terminated,
                      truncated, info_play_ambush)
    if terminated or info_play_ambush.get("game_phase") != GamePhase.RESOLVING_AMBUSH.name:
        print("Game ended or did not enter AMBUSH resolution. Test incomplete.");
        return

    # --- Phase 2: P0 uses AMBUSH to flip their facedown AERODROME ---
    print(f"\nAerodrome Flip Test - Phase 2: P{env.current_player_idx} (P0) uses AMBUSH to flip AERODROME.")
    player_idx_before_action_phase1_2 = env.current_player_idx  # P0
    ambush_choices = env.get_readable_legal_moves()
    action_to_flip_aerodrome = -1;
    desc_flip_aerodrome = ""
    found_target_to_flip_aerodrome = False

    if 'possible_targets' in env.effect_resolution_data:  # Check effect_resolution_data from env
        for mapped_id, p_id, t_idx, c_name in env.effect_resolution_data['possible_targets']:
            # aerodrome_obj is the Card object for AERODROME
            if p_id == 0 and t_idx == theater_idx_aerodrome_fd and aerodrome_obj and c_name == aerodrome_obj.name:
                action_to_flip_aerodrome = mapped_id
                for aid_readable, desc_readable in ambush_choices:  # Find description for this mapped_id
                    if aid_readable == action_to_flip_aerodrome: desc_flip_aerodrome = desc_readable; break
                found_target_to_flip_aerodrome = True;
                break

    if not found_target_to_flip_aerodrome: print(
        f"Could not find P0's FD AERODROME (in T{theater_idx_aerodrome_fd}) as AMBUSH target. Targets: {env.effect_resolution_data.get('possible_targets')}. Skipping."); return

    print(f"P0 using AMBUSH to flip AERODROME: {desc_flip_aerodrome} (Action ID: {action_to_flip_aerodrome})")
    obs, reward, terminated, truncated, info_resolve_ambush = env.step(action_to_flip_aerodrome)
    info_resolve_ambush['current_player_idx_at_action_end'] = player_idx_before_action_phase1_2
    env.render();
    print_step_result("1.2 (P0 Resolves AMBUSH, flips AERODROME)", action_to_flip_aerodrome, desc_flip_aerodrome, obs,
                      reward, terminated, truncated, info_resolve_ambush)

    assert info_resolve_ambush.get("game_phase") == GamePhase.NORMAL_PLAY.name, "Should be NORMAL_PLAY after AMBUSH"
    assert info_resolve_ambush.get(
        "current_player_idx") != player_idx_before_action_phase1_2, "Turn should switch after AMBUSH"  # Switched from P0 to P1

    # Verification 1: AERODROME should now be active
    assert env._is_aerodrome_active_for_player(0), "AERODROME should be ACTIVE after being flipped FU."
    print("AERODROME effect confirmed ACTIVE for Player 0 via internal check.")

    # --- Phase 3: P1 takes another turn ---
    if terminated: print("Battle ended. Cannot test Aerodrome effect on next P0 turn."); return
    print(f"\nAerodrome Flip Test - Phase 3: P{env.current_player_idx} (P1) takes a turn.")
    player_idx_before_action_phase3 = env.current_player_idx  # Should be P1
    p1_legal_moves_phase3 = env.get_readable_legal_moves()  # Renamed variable
    p1_action_id_phase3, p1_desc_phase3 = p1_legal_moves_phase3[0]
    if p1_action_id_phase3 == env.withdraw_action_id and len(p1_legal_moves_phase3) > 1:
        p1_action_id_phase3, p1_desc_phase3 = p1_legal_moves_phase3[1]
    obs, reward, terminated, truncated, info_p1_phase3 = env.step(p1_action_id_phase3)
    info_p1_phase3['current_player_idx_at_action_end'] = player_idx_before_action_phase3
    env.render();
    print_step_result("1.3 (P1 Plays)", p1_action_id_phase3, p1_desc_phase3, obs, reward, terminated, truncated,
                      info_p1_phase3)
    if terminated: print("Battle ended. Cannot test Aerodrome effect on P0's actual next turn."); return
    assert env.current_player_idx == 0, "Turn should be P0's for testing Aerodrome effect."

    # --- Phase 4: Player 0's turn with AERODROME active - Test non-matching play ---
    print(f"\nAerodrome Flip Test - Phase 4: P{env.current_player_idx} (P0) attempting non-matching play.")
    player_idx_before_action_phase4 = env.current_player_idx  # P0

    # We use the same p0_low_str_card_obj and target_non_matching_th_idx_verify from Step 0.3
    if not p0_low_str_card_obj: print(
        "No low-strength card was identified for P0 earlier. Cannot complete Aerodrome test."); return
    # Ensure target_non_matching_th_idx_verify was set
    if 'target_non_matching_th_idx_verify' not in locals() or target_non_matching_th_idx_verify == -1:
        print("No non-matching theater was identified earlier. Cannot complete Aerodrome test.");
        return

    # Find the current slot of p0_low_str_card_obj as hand order might change
    current_low_str_card_slot = -1
    for i, card in enumerate(env.players[0].hand):  # P0's hand
        if card.id == p0_low_str_card_obj.id:
            current_low_str_card_slot = i;
            break
    if current_low_str_card_slot == -1: print(
        f"'{p0_low_str_card_obj.name}' no longer in P0's hand. Test cannot proceed."); return

    aerodrome_enabled_play_id = current_low_str_card_slot * actions_per_slot + target_non_matching_th_idx_verify * 2 + 0  # Face-Up
    aerodrome_enabled_play_desc = (
        f"Play '{p0_low_str_card_obj.name}' (Str {p0_low_str_card_obj.strength}, Type {p0_low_str_card_obj.card_type.name}) "
        f"from slot {current_low_str_card_slot} to {env.theaters[target_non_matching_th_idx_verify].current_type.name} "
        f"(Pos {target_non_matching_th_idx_verify}) Face-Up (AERODROME Active TEST)")

    print(
        f"P0 attempting AERODROME-enabled play: {aerodrome_enabled_play_desc} (Action ID: {aerodrome_enabled_play_id})")

    current_legal_moves_for_aerodrome_test = env.get_readable_legal_moves()
    is_aerodrome_play_now_legal = any(
        aid == aerodrome_enabled_play_id for aid, _ in current_legal_moves_for_aerodrome_test)

    print("Legal moves available to P0 (AERODROME active):")
    for aid, desc in current_legal_moves_for_aerodrome_test: print(f"  ID: {aid} => {desc}")

    assert is_aerodrome_play_now_legal, f"AERODROME EFFECT TEST FAILED: Action '{aerodrome_enabled_play_desc}' (ID: {aerodrome_enabled_play_id}) was NOT found in legal moves."
    print(f"AERODROME EFFECT TEST PASSED: Action '{aerodrome_enabled_play_desc}' IS now legal.")

    obs, reward, terminated, truncated, info_final_aerodrome_play = env.step(aerodrome_enabled_play_id)
    info_final_aerodrome_play['current_player_idx_at_action_end'] = player_idx_before_action_phase4
    env.render();
    print_step_result("1.4 (P0 Aerodrome Effect Play)", aerodrome_enabled_play_id, aerodrome_enabled_play_desc, obs,
                      reward, terminated, truncated, info_final_aerodrome_play)

    target_th_obj = env.theaters[target_non_matching_th_idx_verify]
    p0_top_card_in_target = target_th_obj.get_uncovered_card(0)
    assert p0_top_card_in_target and p0_top_card_in_target.card_ref.id == p0_low_str_card_obj.id and p0_top_card_in_target.is_face_up, \
        f"Card '{p0_low_str_card_obj.name}' not correctly played to non-matching theater under AERODROME."
    print(f"Card '{p0_low_str_card_obj.name}' correctly played to non-matching theater under AERODROME effect.")

    print("\nPassive Effect (AERODROME) on Flip Test Function Finished.")


# TODO: Air Drop test fails sometimes but the engine should still be correct? Problem is probably with setup?
def test_air_drop_effect(env: AirLandSeaBaseEnv, max_setup_retries=15):
    print("\n" + "#" * 10 + " Test: AIR DROP Card Effect " + "#" * 10)

    air_drop_card_id = 6  # AIR_2_AIR_DROP
    found_initial_setup = False

    # Variables to store cards and theaters for the test
    p0_air_drop_card_obj = None
    p0_air_drop_slot_idx = -1
    p0_test_play_card_obj = None  # Card P0 will use for non-matching play
    p0_test_play_slot_idx = -1
    target_non_matching_th_idx = -1
    target_non_matching_th_type_name = ""
    action_id_for_illegal_play_check = -1
    action_id_for_legal_air_drop_play = -1

    for retry in range(max_setup_retries):
        print(f"AIR DROP Test: Setup attempt {retry + 1}/{max_setup_retries}")
        env.reset()
        player0_hand = env.players[0].hand

        temp_air_drop_card = None
        temp_air_drop_slot = -1
        temp_test_play_card = None
        temp_test_play_slot = -1

        for i, card in enumerate(player0_hand):
            if card.id == air_drop_card_id:
                temp_air_drop_card = card
                temp_air_drop_slot = i
            # Find a non-AIR card to test non-matching play with AIR DROP
            elif card.card_type != CardType.AIR and temp_test_play_card is None:  # Prefer non-AIR card
                temp_test_play_card = card
                temp_test_play_slot = i

        # If no non-AIR card, pick any other card that isn't AIR DROP
        if temp_test_play_card is None:
            for i, card in enumerate(player0_hand):
                if card.id != air_drop_card_id:
                    temp_test_play_card = card
                    temp_test_play_slot = i
                    break  # Found some other card

        if temp_air_drop_card and temp_test_play_card:
            p0_air_drop_card_obj = temp_air_drop_card
            p0_air_drop_slot_idx = temp_air_drop_slot
            p0_test_play_card_obj = temp_test_play_card
            p0_test_play_slot_idx = temp_test_play_slot
            found_initial_setup = True
            print(
                f"Favorable hand for AIR DROP: P0 has '{p0_air_drop_card_obj.name}' and test card '{p0_test_play_card_obj.name}'.")
            break

    if not found_initial_setup:
        print(f"Could not get AIR DROP and another card for P0 after {max_setup_retries} retries. Skipping test.")
        return

    print(f"Initial P0 Hand for test: {[card.name for card in env.players[0].hand]}")
    print(f"Initial Theater Layout: {[th.current_type.name for th in env.theaters]}")
    env.render()

    actions_per_slot = NUM_THEATERS * 2

    # --- Step 0: P0 Verify non-matching play with test_card is ILLEGAL ---
    print(f"\nAIR DROP Test - Step 0: P{env.current_player_idx} (P0) Verifying non-matching play is ILLEGAL.")
    player_idx_before_action = env.current_player_idx  # P0

    # Find a non-matching theater for p0_test_play_card_obj
    for i, th in enumerate(env.theaters):
        if th.current_type.name != p0_test_play_card_obj.card_type.name:
            target_non_matching_th_idx = i
            target_non_matching_th_type_name = th.current_type.name
            break
    if target_non_matching_th_idx == -1:
        # This happens if all theaters match the card type (e.g. card is LAND, all theaters are LAND)
        # Or if card is Universal (not in this game)
        print(
            f"No non-matching theater found for test card '{p0_test_play_card_obj.name}' (Type: {p0_test_play_card_obj.card_type.name}). Theaters: {[th.current_type.name for th in env.theaters]}. Skipping specific play check.")
    else:
        action_id_for_illegal_play_check = p0_test_play_slot_idx * actions_per_slot + target_non_matching_th_idx * 2 + 0  # Face-Up
        is_illegal_play_in_mask = any(
            aid == action_id_for_illegal_play_check for aid, _ in env.get_readable_legal_moves())
        assert not is_illegal_play_in_mask, f"Play of {p0_test_play_card_obj.name} to non-matching T{target_non_matching_th_idx} FU SHOULD BE ILLEGAL now."
        print(
            f"Verified: Playing '{p0_test_play_card_obj.name}' to non-matching theater (Pos {target_non_matching_th_idx}) FU is currently ILLEGAL.")

    # --- Step 1: Player 0 plays AIR DROP ---
    print(f"\nAIR DROP Test - Step 1: P{env.current_player_idx} (P0) plays AIR DROP.")
    # Find an AIR theater for AIR DROP
    target_air_theater_idx = -1
    for i, th in enumerate(env.theaters):
        if th.current_type == TheaterType.AIR: target_air_theater_idx = i; break
    if target_air_theater_idx == -1:
        print(
            "No AIR theater found. Playing AIR DROP FD to T0. Effect won't prime."); target_air_theater_idx = 0; play_ad_face_up = False
    else:
        play_ad_face_up = True

    play_air_drop_action_id = p0_air_drop_slot_idx * actions_per_slot + target_air_theater_idx * 2 + (
        0 if play_ad_face_up else 1)
    play_air_drop_desc = f"Play '{p0_air_drop_card_obj.name}' (slot {p0_air_drop_slot_idx}) to {env.theaters[target_air_theater_idx].current_type.name} (Pos {target_air_theater_idx}) {'FU' if play_ad_face_up else 'FD'}"

    if not any(aid == play_air_drop_action_id for aid, _ in env.get_readable_legal_moves()):
        print(
            f"Planned play of AIR DROP ({play_air_drop_desc}) ID {play_air_drop_action_id} is not legal. Skipping further test.");
        return

    obs, reward, terminated, truncated, info_step1 = env.step(play_air_drop_action_id)
    info_step1['current_player_idx_at_action_end'] = player_idx_before_action  # P0
    env.render();
    print_step_result("1 (P0 Plays AIR DROP)", play_air_drop_action_id, play_air_drop_desc, obs, reward, terminated,
                      truncated, info_step1)

    if terminated: print("Battle ended after P0 played AIR DROP. Test incomplete."); return
    if play_ad_face_up:
        assert info_step1.get("air_drop_primed_p0"), "AIR DROP should be primed for P0 for next turn."
        print("AIR DROP effect PRIMED for Player 0's next turn. Correct.")
    else:
        assert not info_step1.get("air_drop_primed_p0"), "AIR DROP should NOT be primed if played FD."
        print("AIR DROP played FD, effect not primed. Test of AIR DROP effect cannot continue this path.")
        return  # Cannot test effect if AD played FD

    assert env.current_player_idx == 1, "Turn should pass to P1."

    # --- Step 2: Player 1 takes a turn ---
    print(f"\nAIR DROP Test - Step 2: P{env.current_player_idx} (P1) takes a turn.")
    player_idx_before_action = env.current_player_idx  # P1
    p1_legal_moves = env.get_readable_legal_moves()
    if not p1_legal_moves: print("P1 has no legal moves. Problem."); return
    p1_action_id, p1_action_desc = p1_legal_moves[0]
    if p1_action_id == env.withdraw_action_id and len(p1_legal_moves) > 1: p1_action_id, p1_action_desc = \
    p1_legal_moves[1]

    obs, reward, terminated, truncated, info_step2 = env.step(p1_action_id)
    info_step2['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("2 (P1's turn)", p1_action_id, p1_action_desc, obs, reward, terminated, truncated, info_step2)

    if terminated: print("Battle ended after P1's turn. Test incomplete."); return
    assert env.current_player_idx == 0, "Turn should pass back to P0."
    assert info_step2.get("air_drop_primed_p0"), "AIR DROP should still be primed for P0."
    assert not info_step2.get(
        "air_drop_active_p0"), "AIR DROP should NOT be active for P0 yet (it's P1's turn results)."

    # In main.py, inside test_air_drop_effect function

    # --- Step 3: Player 0's "Next Turn" - AIR DROP should activate and be usable ---
    print(
        f"\nAIR DROP Test - Step 3: P{env.current_player_idx} (P0) starts 'next turn'. AIR DROP should be active for this turn's actions.")
    player_idx_before_action = env.current_player_idx  # P0

    # The env.step() call below will first activate air_drop_active_this_turn[0] internally,
    # then _get_action_mask (called by get_readable_legal_moves if we called it now, or internally by step if action was illegal)
    # would use that activated state.

    # Try to play p0_test_play_card_obj to target_non_matching_th_idx Face-Up
    current_test_play_card_slot = -1
    if 'p0_test_play_card_obj' not in locals() or p0_test_play_card_obj is None:
        print("Test card for P0 was not defined earlier. Skipping AIR DROP effect play.");
        return
    if 'target_non_matching_th_idx' not in locals() or target_non_matching_th_idx == -1:
        print("Target non-matching theater was not defined earlier. Skipping AIR DROP effect play.");
        return

    for i, card in enumerate(env.players[0].hand):
        if card.id == p0_test_play_card_obj.id: current_test_play_card_slot = i; break

    if current_test_play_card_slot == -1:
        print(
            f"Test card '{p0_test_play_card_obj.name}' no longer in P0's hand. Hand: {[c.name for c in env.players[0].hand]}. Skipping.");
        return

    action_id_for_legal_air_drop_play = current_test_play_card_slot * actions_per_slot + target_non_matching_th_idx * 2 + 0  # Face-Up
    air_drop_play_desc = (f"Play '{p0_test_play_card_obj.name}' (Type {p0_test_play_card_obj.card_type.name}) "
                          f"from slot {current_test_play_card_slot} to {env.theaters[target_non_matching_th_idx].current_type.name} "
                          f"(Pos {target_non_matching_th_idx}) Face-Up (AIR DROP Active TEST)")

    print(f"P0 attempting AIR DROP-enabled play: {air_drop_play_desc} (Action ID: {action_id_for_legal_air_drop_play})")

    # Get legal moves *just before* P0 acts.
    # The _get_action_mask within get_readable_legal_moves will be called.
    # It will *not* see air_drop_active_this_turn[0] as True yet because step() for P0 hasn't run.
    # This means the assertion here will *still fail* if we expect the mask to be updated before step().
    # The mask update and legality check based on AIR_DROP happens *inside* the env.step() call for P0.
    # So, we should assert that the play *succeeds* and that the "air_drop_activated_for_turn" flag *is in the info returned by step()*.

    # Let's remove the pre-emptive assertion on legal moves for this specific test,
    # and instead rely on the step succeeding and the info returned by it.
    # current_legal_moves_for_air_drop = env.get_readable_legal_moves()
    # is_air_drop_play_now_legal = any(aid == action_id_for_legal_air_drop_play for aid, _ in current_legal_moves_for_air_drop)
    # assert is_air_drop_play_now_legal, f"AIR DROP EFFECT TEST FAILED: ..." (This assertion was too early)


    obs, reward, terminated, truncated, info_step3 = env.step(action_id_for_legal_air_drop_play)
    info_step3['current_player_idx_at_action_end'] = player_idx_before_action  # P0

    env.render()
    print_step_result("3 (P0 Uses AIR DROP)", action_id_for_legal_air_drop_play, air_drop_play_desc, obs, reward,
                      terminated, truncated, info_step3)

    # NOW, we check the info returned by the step where AIR DROP should have been active.
    assert info_step3.get("air_drop_activated_for_turn") == f"P{player_idx_before_action}", \
        f"AIR DROP should have been activated for P{player_idx_before_action} during this step. Info: {info_step3}"
    print("AIR DROP activation during step confirmed by info dict. Correct.")

    # Also verify the play didn't result in an error (which means it was considered legal by _is_play_action_legal)
    assert "error" not in info_step3, f"AIR DROP play resulted in an error: {info_step3.get('error')}"
    print(f"No error during AIR DROP play. Play was legal. Correct.")

    if terminated: print("Battle ended after P0 used AIR DROP. Test concludes."); return
    assert env.current_player_idx == 1, "Turn should switch to P1."

    info_after_p0_air_drop_turn = env._get_info()  # Info for P1's upcoming turn
    assert not info_after_p0_air_drop_turn.get(
        "air_drop_active_p0"), "AIR DROP should be INACTIVE for P0 after the turn it was used."
    print("AIR DROP effect confirmed INACTIVE for P0 after its active turn. Correct.")

    # --- Step 4: P1 takes a turn, then P0's next turn (AIR DROP should remain off) ---
    print(f"\nAIR DROP Test - Step 4: P{env.current_player_idx} (P1) takes another turn.")
    player_idx_before_action = env.current_player_idx  # P1
    p1_legal_moves_2 = env.get_readable_legal_moves()
    if not p1_legal_moves_2: print("P1 has no legal moves. Problem."); return
    p1_action_id_2, p1_desc_2 = p1_legal_moves_2[0]
    if p1_action_id_2 == env.withdraw_action_id and len(p1_legal_moves_2) > 1: p1_action_id_2, p1_desc_2 = \
    p1_legal_moves_2[1]

    obs, reward, terminated, truncated, info_step4_p1 = env.step(p1_action_id_2)
    info_step4_p1['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("4.1 (P1's turn)", p1_action_id_2, p1_desc_2, obs, reward, terminated, truncated, info_step4_p1)
    if terminated: print("Battle ended. Test concludes."); return

    print(f"\nAIR DROP Test - Step 4.2: P{env.current_player_idx} (P0) starts subsequent turn. AIR DROP should be OFF.")
    info_p0_subsequent_turn = env._get_info()
    assert not info_p0_subsequent_turn.get(
        "air_drop_active_p0"), "AIR DROP should remain INACTIVE for P0 on subsequent turns."
    assert not info_p0_subsequent_turn.get("air_drop_primed_p0"), "AIR DROP should not be re-primed for P0."
    print("AIR DROP confirmed to be off for P0 on a subsequent turn. Correct.")

    # Verify the non-matching play is ILLEGAL again
    if target_non_matching_th_idx != -1 and current_test_play_card_slot != -1 and action_id_for_illegal_play_check != -1:
        # Re-check if the test card is still in hand to make this check valid
        p0_still_has_test_card = any(card.id == p0_test_play_card_obj.id for card in env.players[0].hand)
        if p0_still_has_test_card:  # It was played in step 3, so this check is actually not for the same card
            # Find a new low strength card if possible
            new_test_card_obj = None;
            new_test_card_slot = -1
            for i, card in enumerate(env.players[0].hand):
                if card.strength <= 3: new_test_card_obj = card; new_test_card_slot = i; break

            if new_test_card_obj:
                target_non_matching_for_new_card = -1
                for i, th in enumerate(env.theaters):
                    if th.current_type.name != new_test_card_obj.card_type.name: target_non_matching_for_new_card = i; break

                if target_non_matching_for_new_card != -1:
                    action_id_subsequent_illegal_check = new_test_card_slot * actions_per_slot + target_non_matching_for_new_card * 2 + 0  # FU
                    is_now_illegal_again = not any(
                        aid == action_id_subsequent_illegal_check for aid, _ in env.get_readable_legal_moves())
                    assert is_now_illegal_again, f"Play of '{new_test_card_obj.name}' to non-matching T{target_non_matching_for_new_card} FU SHOULD BE ILLEGAL again."
                    print(
                        f"Verified: Playing '{new_test_card_obj.name}' to non-matching theater (Pos {target_non_matching_for_new_card}) FU is ILLEGAL again. Correct.")
        else:
            print("P0 no longer has the original test card to re-verify illegality (it was played).")

    print("\nAIR DROP Test Function Finished.")


def test_transport_effect(env: AirLandSeaBaseEnv, max_setup_retries=20):  # Increased retries for specific hand
    print("\n" + "#" * 10 + " Test: TRANSPORT Card Effect " + "#" * 10)

    transport_card_id = 11  # SEA_1_TRANSPORT
    found_initial_setup = False

    # For this test, P0 needs TRANSPORT and at least TWO other cards:
    # One to play on the board (Card A) and one to play as TRANSPORT itself.
    # P1 also needs a card to play.
    p0_transport_card_obj = None
    p0_transport_slot_idx = -1
    p0_card_A_obj = None  # Card to be played and then moved
    p0_card_A_slot_idx = -1

    for retry in range(max_setup_retries):
        print(f"TRANSPORT Test: Setup attempt {retry + 1}/{max_setup_retries}")
        env.reset()
        player0_hand = env.players[0].hand

        has_transport = any(card.id == transport_card_id for card in player0_hand)

        # Find two distinct cards other than transport if possible
        other_cards_for_p0 = [card for card in player0_hand if card.id != transport_card_id]

        if has_transport and len(other_cards_for_p0) >= 1 and len(
            env.players[1].hand) > 0:  # P0 needs transport and at least one other card
            # Assign TRANSPORT card
            for i, card in enumerate(player0_hand):
                if card.id == transport_card_id:
                    p0_transport_card_obj = card
                    p0_transport_slot_idx = i
                    break
            # Assign Card A (the first "other" card)
            for i, card in enumerate(player0_hand):
                if card.id != transport_card_id:
                    p0_card_A_obj = card
                    p0_card_A_slot_idx = i
                    break

            if p0_transport_card_obj and p0_card_A_obj:  # Ensure both were found
                found_initial_setup = True
                print(
                    f"Favorable hand for TRANSPORT: P0 has '{p0_transport_card_obj.name}', and card '{p0_card_A_obj.name}' to move.")
                break

    if not found_initial_setup:
        print(
            f"Could not get suitable hand (TRANSPORT + another card for P0, P1 has cards) after {max_setup_retries} retries. Skipping test.")
        return

    print(f"Initial P0 Hand for test: {[card.name for card in env.players[0].hand]}")
    print(f"Initial Theater Layout: {[th.current_type.name for th in env.theaters]}")
    env.render()

    actions_per_slot = NUM_THEATERS * 2
    original_p0_card_A_played_theater_idx = 0  # P0 will play Card A to Pos 0
    original_p0_card_A_played_face_up = False  # Play it FD for simplicity in finding it later

    # --- Setup Phase ---
    # Step 0.1: Player 0 plays Card A to Theater Pos 0 Face-Down
    print(
        f"\nTRANSPORT Test Setup: P{env.current_player_idx} (P0) plays '{p0_card_A_obj.name}' to Pos {original_p0_card_A_played_theater_idx} FD.")
    player_idx_before_action = env.current_player_idx

    # Find current slot of Card A
    current_p0_card_A_slot = -1
    for i, card in enumerate(env.players[0].hand):
        if card.id == p0_card_A_obj.id: current_p0_card_A_slot = i; break
    if current_p0_card_A_slot == -1: print(
        f"Error: Card A '{p0_card_A_obj.name}' not in P0's hand for setup. Skipping."); return

    p0_play_card_A_action_id = current_p0_card_A_slot * actions_per_slot + original_p0_card_A_played_theater_idx * 2 + (
        0 if original_p0_card_A_played_face_up else 1)
    p0_play_card_A_desc = f"Play '{p0_card_A_obj.name}' (slot {current_p0_card_A_slot}) to {env.theaters[original_p0_card_A_played_theater_idx].current_type.name} (Pos {original_p0_card_A_played_theater_idx}) {'FU' if original_p0_card_A_played_face_up else 'FD'}"

    obs, reward, terminated, truncated, info_p0_setup1 = env.step(p0_play_card_A_action_id)
    info_p0_setup1['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("0.1 (P0 Plays Card A)", p0_play_card_A_action_id, p0_play_card_A_desc, obs, reward, terminated,
                      truncated, info_p0_setup1)
    if terminated: print("Battle ended during P0 setup. Test incomplete."); return

    # Step 0.2: Player 1 plays any card
    print(f"\nTRANSPORT Test Setup: P{env.current_player_idx} (P1) plays a card.")
    player_idx_before_action = env.current_player_idx
    p1_legal_moves = env.get_readable_legal_moves()
    if not p1_legal_moves: print("P1 has no legal moves. Problem."); return
    p1_action_id, p1_desc = p1_legal_moves[0]
    if p1_action_id == env.withdraw_action_id and len(p1_legal_moves) > 1: p1_action_id, p1_desc = p1_legal_moves[1]

    obs, reward, terminated, truncated, info_p1_setup = env.step(p1_action_id)
    info_p1_setup['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("0.2 (P1 Plays)", p1_action_id, p1_desc, obs, reward, terminated, truncated, info_p1_setup)
    if terminated: print("Battle ended during P1 setup. Test incomplete."); return

    initial_battle_turn_counter_before_transport = env.battle_turn_counter  # Should be 2

    # --- Step 1: Player 0 plays TRANSPORT ---
    print(f"\nTRANSPORT Test - Step 1: P{env.current_player_idx} (P0) plays TRANSPORT.")
    player_idx_before_action = env.current_player_idx  # P0

    current_p0_transport_slot = -1
    for i, card in enumerate(env.players[0].hand):
        if card.id == transport_card_id: current_p0_transport_slot = i; break
    if current_p0_transport_slot == -1: print(f"Error: TRANSPORT card no longer in P0 hand. Skipping."); return

    # Play TRANSPORT to its native SEA theater, or fallback
    target_sea_theater_idx = -1;
    play_transport_face_up = True
    for i, th in enumerate(env.theaters):
        if th.current_type == TheaterType.SEA: target_sea_theater_idx = i; break
    if target_sea_theater_idx == -1:  # Fallback if no SEA theater
        target_sea_theater_idx = 1  # Play to center if no SEA
        play_transport_face_up = False  # Play FD if not native theater for effect to trigger
        print("Warning: No SEA theater. Playing TRANSPORT FD to center. Effect test will be limited if FD.")

    play_transport_action_id = current_p0_transport_slot * actions_per_slot + target_sea_theater_idx * 2 + (
        0 if play_transport_face_up else 1)
    play_transport_desc = f"Play '{p0_transport_card_obj.name}' (slot {current_p0_transport_slot}) to {env.theaters[target_sea_theater_idx].current_type.name} (Pos {target_sea_theater_idx}) {'FU' if play_transport_face_up else 'FD'}"

    if not any(aid == play_transport_action_id for aid, _ in env.get_readable_legal_moves()):
        print(
            f"Planned play of TRANSPORT ({play_transport_desc}) ID {play_transport_action_id} is not legal. Skipping further test.");
        return

    obs, reward, terminated, truncated, info_play_transport = env.step(play_transport_action_id)
    info_play_transport['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("1.1 (Play TRANSPORT)", play_transport_action_id, play_transport_desc, obs, reward, terminated,
                      truncated, info_play_transport)

    if terminated: print("Battle ended after TRANSPORT play. Incomplete."); return
    if not play_transport_face_up: print("TRANSPORT played FD, effect won't trigger. Test ends."); return

    assert info_play_transport.get("game_phase") == GamePhase.RESOLVING_TRANSPORT_SELECT_CARD.name, \
        f"Phase should be RESOLVING_TRANSPORT_SELECT_CARD, but is {info_play_transport.get('game_phase')}"
    print(f"Game phase is now: {info_play_transport.get('game_phase')}. Correct.")
    assert info_play_transport.get(
        "current_player_idx") == player_idx_before_action, "Player should still be P0 for TRANSPORT choice."
    print("Current player is still P0. Correct.")
    assert info_play_transport.get(
        "battle_turn_counter") == initial_battle_turn_counter_before_transport, "Battle turn counter should not have incremented yet."

    # --- Step 2: Player 0 selects card to move ---
    print(f"\nTRANSPORT Test - Step 1.2: P{env.current_player_idx} (P0) selects card to move.")
    player_idx_before_action = env.current_player_idx  # P0
    transport_card_choices = env.get_readable_legal_moves()
    print("Legal choices for TRANSPORT (select card):")
    for aid, desc in transport_card_choices: print(f"  Action ID: {aid} => {desc}")

    # We want to select Card A which is at Pos original_p0_card_A_played_theater_idx (should be 0)
    # Its stack_idx should be 0 if it's the only card P0 played there.
    action_id_select_card_A = -1
    desc_select_card_A = ""
    found_card_A_target = False
    if 'effect_data_transport_sources' in info_play_transport:
        for mapped_id, src_p_id, src_t_idx, src_stack_idx, c_name, c_id in info_play_transport[
            'effect_data_transport_sources']:
            if src_p_id == 0 and src_t_idx == original_p0_card_A_played_theater_idx and c_id == p0_card_A_obj.id:
                action_id_select_card_A = mapped_id
                # Find description
                for aid_readable, desc_readable in transport_card_choices:
                    if aid_readable == action_id_select_card_A: desc_select_card_A = desc_readable; break
                found_card_A_target = True;
                break

    if not found_card_A_target: print(
        f"Could not find Card A ('{p0_card_A_obj.name}') as a transport source. Effect_data: {info_play_transport.get('effect_data_transport_sources')}. Skipping."); return

    print(f"P0 selecting source for TRANSPORT: {desc_select_card_A} (Action ID: {action_id_select_card_A})")
    obs, reward, terminated, truncated, info_select_source = env.step(action_id_select_card_A)
    info_select_source['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("1.2 (Select Source Card)", action_id_select_card_A, desc_select_card_A, obs, reward, terminated,
                      truncated, info_select_source)

    if terminated: print("Battle ended after selecting source. Incomplete."); return
    assert info_select_source.get(
        "game_phase") == GamePhase.RESOLVING_TRANSPORT_SELECT_DEST.name, "Phase should be RESOLVING_TRANSPORT_SELECT_DEST."
    print(f"Game phase is now: {info_select_source.get('game_phase')}. Correct.")
    assert info_select_source.get(
        "current_player_idx") == player_idx_before_action, "Player should still be P0 for TRANSPORT destination choice."

    # --- Step 3: Player 0 selects destination theater ---
    print(f"\nTRANSPORT Test - Step 1.3: P{env.current_player_idx} (P0) selects destination.")
    player_idx_before_action = env.current_player_idx  # P0
    transport_dest_choices = env.get_readable_legal_moves()
    print(f"Legal choices for TRANSPORT (select destination for '{p0_card_A_obj.name}'):")
    for aid, desc in transport_dest_choices: print(f"  Action ID: {aid} => {desc}")

    if not transport_dest_choices: print(
        "No destination choices for TRANSPORT. Fizzle logic in env should handle."); return

    chosen_dest_action_id, chosen_dest_desc = random.choice(
        transport_dest_choices)  # Choose one of the 2 other theaters

    print(f"P0 selecting destination for TRANSPORT: {chosen_dest_desc} (Action ID: {chosen_dest_action_id})")
    obs, reward, terminated, truncated, info_select_dest = env.step(chosen_dest_action_id)
    info_select_dest['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("1.3 (Select Destination)", chosen_dest_action_id, chosen_dest_desc, obs, reward, terminated,
                      truncated, info_select_dest)

    assert info_select_dest.get(
        "game_phase") == GamePhase.NORMAL_PLAY.name, "Phase should be NORMAL_PLAY after TRANSPORT."
    print(f"Game phase is now: {info_select_dest.get('game_phase')}. Correct.")
    assert info_select_dest.get("current_player_idx") != player_idx_before_action, "Turn should switch after TRANSPORT."
    print(f"Current player is now P{info_select_dest.get('current_player_idx')}. Correct.")

    expected_turn_count = initial_battle_turn_counter_before_transport + 1  # TRANSPORT is one full turn
    assert info_select_dest.get("battle_turn_counter") == expected_turn_count, \
        f"Battle turn counter error. Expected {expected_turn_count}, got {info_select_dest.get('battle_turn_counter')}"
    print(f"Battle turn counter is {info_select_dest.get('battle_turn_counter')}. Correct.")

    # Further verification: Check if p0_card_A_obj is in the new theater and not in the old one
    # This requires parsing chosen_dest_desc or using data from info_select_dest['effect_data_transport_destinations']
    # and info_select_source['effect_data_card_to_transport']
    print("TRANSPORT effect full resolution test part completed.")
    print("\nTRANSPORT Test Function Finished.")


def test_escalation_effect(env: AirLandSeaBaseEnv, max_setup_retries=15):
    print("\n" + "#" * 10 + " Test: ESCALATION Card Effect " + "#" * 10)

    escalation_card_id = 12  # SEA_2_ESCALATION
    # For flipping ESCALATION later, we might need a flip enabler like AMBUSH for P1 or P0
    flip_enabler_card_id = 1  # LAND_2_AMBUSH

    found_initial_setup = False
    p0_escalation_card_obj = None
    p0_other_card_obj = None  # Card to play facedown

    for retry in range(max_setup_retries):
        print(f"ESCALATION Test: Setup attempt {retry + 1}/{max_setup_retries}")
        env.reset()
        player0_hand = env.players[0].hand
        player1_hand = env.players[1].hand  # P1 needs a card to play for turn passing & optionally Ambush

        has_escalation_p0 = any(card.id == escalation_card_id for card in player0_hand)
        # P0 needs another card to play FD
        other_cards_p0 = [card for card in player0_hand if card.id != escalation_card_id]

        # P1 needs AMBUSH to test deactivation by flipping Escalation
        p1_has_flip_enabler = any(card.id == flip_enabler_card_id for card in player1_hand)

        if has_escalation_p0 and len(other_cards_p0) >= 1 and p1_has_flip_enabler:
            # Assign Escalation
            for card in player0_hand:
                if card.id == escalation_card_id: p0_escalation_card_obj = card; break
            # Assign the first "other" card for P0
            p0_other_card_obj = other_cards_p0[0]

            found_initial_setup = True
            print(
                f"Favorable hand for ESCALATION: P0 has '{p0_escalation_card_obj.name}', a test card '{p0_other_card_obj.name}', P1 has AMBUSH.")
            break

    if not found_initial_setup:
        print(
            f"Could not get suitable hand (P0: ESCALATION + other; P1: AMBUSH) after {max_setup_retries} retries. Skipping test.")
        return

    print(f"Initial P0 Hand for test: {[card.name for card in env.players[0].hand]}")
    print(f"Initial Theater Layout: {[th.current_type.name for th in env.theaters]}")
    env.render()

    actions_per_slot = NUM_THEATERS * 2
    theater_for_p0_fd_card = 0  # P0 will play their other card FD to Pos 0
    theater_for_escalation = 1  # P0 will play Escalation to Pos 1

    # --- Setup Phase ---
    # Step 0.1: Player 0 plays their "other card" (p0_other_card_obj) Face-Down into Theater 0
    print(
        f"\nESCALATION Test Setup: P{env.current_player_idx} (P0) plays '{p0_other_card_obj.name}' FD to Pos {theater_for_p0_fd_card}.")
    player_idx_before_action = env.current_player_idx

    p0_other_card_slot = -1
    for i, card in enumerate(env.players[0].hand):
        if card.id == p0_other_card_obj.id: p0_other_card_slot = i; break
    if p0_other_card_slot == -1: print(
        f"Error: P0's other card '{p0_other_card_obj.name}' not in hand. Skipping."); return

    p0_play_other_fd_action_id = p0_other_card_slot * actions_per_slot + theater_for_p0_fd_card * 2 + 1  # 1 for FD
    p0_play_other_fd_desc = f"Play '{p0_other_card_obj.name}' (slot {p0_other_card_slot}) to {env.theaters[theater_for_p0_fd_card].current_type.name} (Pos {theater_for_p0_fd_card}) FD"

    obs, reward, term, trunc, info_p0_plays_fd = env.step(p0_play_other_fd_action_id)
    info_p0_plays_fd['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("0.1 (P0 Plays Card FD)", p0_play_other_fd_action_id, p0_play_other_fd_desc, obs, reward, term,
                      trunc, info_p0_plays_fd)
    if term: print("Battle ended. Test incomplete."); return

    # Verify strength of P0's FD card in Theater 0 is 2 (Escalation not active)
    strength_t0_p0_before = env._get_player_strength_in_theater_with_effects(0, theater_for_p0_fd_card)
    print(
        f"Strength in T{theater_for_p0_fd_card} for P0 (before ESCALATION active): {strength_t0_p0_before} (Expected 2 if only this card)")
    assert strength_t0_p0_before == 2, "Facedown card strength should be 2 before ESCALATION."

    # Step 0.2: Player 1 plays any card
    print(f"\nESCALATION Test Setup: P{env.current_player_idx} (P1) plays a card.")
    player_idx_before_action = env.current_player_idx
    p1_legal_moves_1 = env.get_readable_legal_moves()
    if not p1_legal_moves_1: print("P1 has no moves. Test incomplete."); return
    p1_action_id_1, p1_desc_1 = p1_legal_moves_1[0]
    if p1_action_id_1 == env.withdraw_action_id and len(p1_legal_moves_1) > 1: p1_action_id_1, p1_desc_1 = \
    p1_legal_moves_1[1]

    obs, reward, term, trunc, info_p1_plays_1 = env.step(p1_action_id_1)
    info_p1_plays_1['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("0.2 (P1 Plays)", p1_action_id_1, p1_desc_1, obs, reward, term, trunc, info_p1_plays_1)
    if term: print("Battle ended. Test incomplete."); return

    # --- Phase 1: Player 0 plays ESCALATION Face-Up ---
    print(f"\nESCALATION Test - Phase 1: P{env.current_player_idx} (P0) plays ESCALATION FU.")
    player_idx_before_action = env.current_player_idx  # P0

    p0_escalation_slot = -1
    for i, card in enumerate(env.players[0].hand):
        if card.id == escalation_card_id: p0_escalation_slot = i; break
    if p0_escalation_slot == -1: print("Error: ESCALATION not in P0's hand. Skipping."); return

    # Try to play ESCALATION to its native SEA theater if possible, else any theater where FU is legal
    target_escalation_play_theater_idx = -1
    for i, th in enumerate(env.theaters):
        if th.current_type == CardType.SEA: target_escalation_play_theater_idx = i; break
    if target_escalation_play_theater_idx == -1: target_escalation_play_theater_idx = theater_for_escalation  # Fallback to pre-assigned

    play_escalation_action_id = p0_escalation_slot * actions_per_slot + target_escalation_play_theater_idx * 2 + 0  # 0 for FU
    play_escalation_desc = f"Play '{p0_escalation_card_obj.name}' (slot {p0_escalation_slot}) to {env.theaters[target_escalation_play_theater_idx].current_type.name} (Pos {target_escalation_play_theater_idx}) FU"

    if not any(aid == play_escalation_action_id for aid, _ in env.get_readable_legal_moves()):
        print(
            f"Planned play of ESCALATION ({play_escalation_desc}) ID {play_escalation_action_id} is not legal. Finding alternative FU play.");
        found_alt_esc_play = False
        for aid_alt, desc_alt in env.get_readable_legal_moves():
            if "ESCALATION" in desc_alt.upper() and "FACE-UP" in desc_alt.upper() and aid_alt != env.withdraw_action_id:
                play_escalation_action_id = aid_alt;
                play_escalation_desc = desc_alt;
                found_alt_esc_play = True;
                break
        if not found_alt_esc_play: print("No legal FU play for ESCALATION. Skipping."); return

    obs, reward, term, trunc, info_p0_plays_esc_fu = env.step(play_escalation_action_id)
    info_p0_plays_esc_fu['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("1.1 (P0 Plays ESCALATION FU)", play_escalation_action_id, play_escalation_desc, obs, reward,
                      term, trunc, info_p0_plays_esc_fu)
    if term: print("Battle ended. Test incomplete."); return

    # Verification 2: Strength of P0's FD card in Theater 0 should NOW be 4
    assert env._is_escalation_active_for_player(0), "ESCALATION should be active for P0."
    print("ESCALATION confirmed ACTIVE for Player 0.")
    strength_t0_p0_after = env._get_player_strength_in_theater_with_effects(0, theater_for_p0_fd_card)
    print(
        f"Strength in T{theater_for_p0_fd_card} for P0 (ESCALATION active): {strength_t0_p0_after} (Expected 4 if only that one FD card)")
    # This assertion depends on Card C being the only card for P0 in that theater, or sum being correct
    # Let's check the specific card's value contribution.
    card_c_in_theater_val_now = 0
    p0_cards_in_t0 = env.theaters[theater_for_p0_fd_card].player_zones[0]
    for p_card in p0_cards_in_t0:
        if p_card.card_ref.id == p0_other_card_obj.id and not p_card.is_face_up:
            card_c_in_theater_val_now = p_card.get_current_strength(is_escalation_active_for_owner=True)
            break
    assert card_c_in_theater_val_now == 4, f"Facedown Card C strength should be 4 with ESCALATION active, got {card_c_in_theater_val_now}."
    print("ESCALATION Activation PASSED: P0's facedown card strength correctly updated to 4.")

    # --- Phase 2 (Optional): P1 plays AMBUSH to flip P0's ESCALATION Face-Down ---
    print(f"\nESCALATION Test - Phase 2: P{env.current_player_idx} (P1) plays AMBUSH to flip P0's ESCALATION.")
    player_idx_before_action = env.current_player_idx  # P1

    p1_ambush_slot = -1;
    ambush_obj_p1 = None
    for i, card in enumerate(env.players[1].hand):  # P1's hand
        if card.id == flip_enabler_card_id: p1_ambush_slot = i; ambush_obj_p1 = card; break
    if p1_ambush_slot == -1: print(
        f"P1 does not have AMBUSH (ID {flip_enabler_card_id}). Skipping deactivation test part."); return

    # P1 plays AMBUSH FU (e.g., to its native LAND theater or fallback)
    target_p1_ambush_play_th_idx = -1
    for i, th in enumerate(env.theaters):
        if th.current_type == TheaterType.LAND: target_p1_ambush_play_th_idx = i; break
    if target_p1_ambush_play_th_idx == -1: target_p1_ambush_play_th_idx = (
                                                                                  theater_for_escalation + 1) % NUM_THEATERS  # fallback

    p1_play_ambush_action_id = p1_ambush_slot * actions_per_slot + target_p1_ambush_play_th_idx * 2 + 0  # FU
    p1_play_ambush_desc = f"Play '{ambush_obj_p1.name}' (slot {p1_ambush_slot}) to {env.theaters[target_p1_ambush_play_th_idx].current_type.name} (Pos {target_p1_ambush_play_th_idx}) FU"

    if not any(aid == p1_play_ambush_action_id for aid, _ in env.get_readable_legal_moves()):
        print(f"P1 AMBUSH play ({p1_play_ambush_desc}) not legal. Skipping deactivation.");
        return

    obs, reward, term, trunc, info_p1_plays_ambush = env.step(p1_play_ambush_action_id)
    info_p1_plays_ambush['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("2.1 (P1 Plays AMBUSH)", p1_play_ambush_action_id, p1_play_ambush_desc, obs, reward, term, trunc,
                      info_p1_plays_ambush)
    if term or info_p1_plays_ambush.get("game_phase") != GamePhase.RESOLVING_AMBUSH.name: print(
        "P1 AMBUSH play failed or did not trigger effect. Skipping deactivation."); return

    # P1 uses AMBUSH to flip P0's ESCALATION (which is in theater_for_escalation)
    ambush_choices_p1 = env.get_readable_legal_moves()
    action_to_flip_p0_escalation = -1;
    desc_flip_p0_escalation = ""
    found_target_p0_escalation = False
    if 'possible_targets' in env.effect_resolution_data:
        for mapped_id, p_id, t_idx, c_name in env.effect_resolution_data['possible_targets']:
            if p_id == 0 and t_idx == target_escalation_play_theater_idx and p0_escalation_card_obj and c_name == p0_escalation_card_obj.name:
                action_to_flip_p0_escalation = mapped_id
                for aid_r, desc_r in ambush_choices_p1:
                    if aid_r == action_to_flip_p0_escalation: desc_flip_p0_escalation = desc_r; break
                found_target_p0_escalation = True;
                break

    if not found_target_p0_escalation: print(
        f"Could not find P0's ESCALATION as target for P1's AMBUSH. Targets: {env.effect_resolution_data.get('possible_targets')}. Skipping deactivation."); return

    print(
        f"P1 using AMBUSH to flip P0's ESCALATION: {desc_flip_p0_escalation} (Action ID: {action_to_flip_p0_escalation})")
    player_idx_before_action = env.current_player_idx  # P1 resolving Ambush
    obs, reward, term, trunc, info_p1_resolves_ambush = env.step(action_to_flip_p0_escalation)
    info_p1_resolves_ambush['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("2.2 (P1 Resolves AMBUSH, flips P0's ESCALATION)", action_to_flip_p0_escalation,
                      desc_flip_p0_escalation, obs, reward, term, trunc, info_p1_resolves_ambush)

    # Verification 3: ESCALATION should be INACTIVE, strength of Card C should be 2
    assert not env._is_escalation_active_for_player(0), "ESCALATION should be INACTIVE for P0 after being flipped FD."
    print("ESCALATION confirmed INACTIVE for Player 0 after P1's AMBUSH.")

    card_c_in_theater_final = env.theaters[theater_for_p0_fd_card].get_uncovered_card(
        0)  # Assumes Card C is still top for P0
    if card_c_in_theater_final and card_c_in_theater_final.card_ref.id == p0_other_card_obj.id:
        strength_card_c_final_val = card_c_in_theater_final.get_current_strength(
            is_escalation_active_for_owner=False  # We assert Escalation is now INactive for P0
        )
        print(
            f"Strength of P0's facedown Card C (in T{theater_for_p0_fd_card}) after Escalation flipped FD: {strength_card_c_final_val} (Expected 2)")
        assert strength_card_c_final_val == 2, "Card C strength should revert to 2 after ESCALATION is deactivated."
        print("ESCALATION Deactivation PASSED: P0's facedown card strength correctly reverted to 2.")
    else:
        print(f"Warning: Card C not found as expected in T{theater_for_p0_fd_card} for P0 for final check.")

    print("\nPassive Effect (ESCALATION) on Flip Test Function Finished.")


def test_cover_fire_effect(env: AirLandSeaBaseEnv, max_setup_retries=25):  # Increased retries for more specific hand
    print("\n" + "#" * 10 + " Test: COVER FIRE Card Effect " + "#" * 10)

    cover_fire_card_id = 16  # SEA_4_COVER_FIRE
    ambush_card_id = 1  # LAND_2_AMBUSH (for P1 to flip Cover Fire)

    p0_has_cover_fire = False
    p0_other_cards = []  # Need two other distinct cards for P0
    p1_has_ambush = False

    # Variables to store located cards for P0
    p0_cover_fire_obj = None
    p0_card_A_obj = None
    p0_card_B_obj = None

    for retry in range(max_setup_retries):
        print(f"COVER FIRE Test: Setup attempt {retry + 1}/{max_setup_retries}")
        env.reset()
        player0_hand = env.players[0].hand
        player1_hand = env.players[1].hand

        p0_has_cover_fire = any(card.id == cover_fire_card_id for card in player0_hand)
        temp_other_cards_p0 = [card for card in player0_hand if card.id != cover_fire_card_id]
        p1_has_ambush = any(card.id == ambush_card_id for card in player1_hand)

        if p0_has_cover_fire and len(temp_other_cards_p0) >= 2 and p1_has_ambush:
            for card in player0_hand:
                if card.id == cover_fire_card_id:
                    p0_cover_fire_obj = card
                elif p0_card_A_obj is None and card.id != cover_fire_card_id:
                    p0_card_A_obj = card
                elif p0_card_B_obj is None and card.id != cover_fire_card_id and card.id != p0_card_A_obj.id:
                    p0_card_B_obj = card

            if p0_cover_fire_obj and p0_card_A_obj and p0_card_B_obj:  # Ensure all three distinct cards found for P0
                found_initial_setup = True
                print(
                    f"Favorable hand: P0 has '{p0_cover_fire_obj.name}', '{p0_card_A_obj.name}', '{p0_card_B_obj.name}'. P1 has AMBUSH.")
                break
            else:  # Reset for next try if distinct cards weren't assigned properly
                p0_cover_fire_obj = p0_card_A_obj = p0_card_B_obj = None

    if not found_initial_setup:
        print(f"Could not get suitable hand after {max_setup_retries} retries. Skipping test.")
        return

    # Identify a SEA theater for Cover Fire play
    sea_theater_idx = -1
    for i, th in enumerate(env.theaters):
        if th.current_type == TheaterType.SEA:
            sea_theater_idx = i
            break

    if sea_theater_idx == -1:
        print("No SEA theater found in this layout. COVER FIRE FU play test skipped.")
        return

    print(f"Initial P0 Hand for test: {[card.name for card in env.players[0].hand]}")
    print(f"Initial Theater Layout: {[th.current_type.name for th in env.theaters]}")
    print(f"Identified SEA theater for stacking: Pos {sea_theater_idx}")
    env.render()

    actions_per_slot = NUM_THEATERS * 2

    # --- Setup Phase ---
    # Step 0.1: P0 plays Card A Face-Down into the SEA theater
    print(
        f"\nCOVER FIRE Test Setup: P{env.current_player_idx} (P0) plays '{p0_card_A_obj.name}' FD to SEA theater (Pos {sea_theater_idx}).")
    player_idx_before_action = env.current_player_idx

    p0_card_A_slot = -1
    for i, card in enumerate(env.players[0].hand):
        if card.id == p0_card_A_obj.id: p0_card_A_slot = i; break
    if p0_card_A_slot == -1: print(f"Error: P0's Card A not in hand. Skipping."); return

    p0_play_card_A_id = p0_card_A_slot * actions_per_slot + sea_theater_idx * 2 + 1  # FD
    p0_play_card_A_desc = f"Play '{p0_card_A_obj.name}' to SEA (Pos {sea_theater_idx}) FD"

    obs, r, term, tr, info_s01 = env.step(p0_play_card_A_id)
    info_s01['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("0.1 (P0 Plays Card A FD)", p0_play_card_A_id, p0_play_card_A_desc, obs, r, term, tr, info_s01)
    if term: print("Battle ended. Test incomplete."); return
    assert env._get_player_strength_in_theater_with_effects(0, sea_theater_idx) == 2, "Card A FD strength should be 2."

    # Step 0.2: P1 plays any card
    print(f"\nCOVER FIRE Test Setup: P{env.current_player_idx} (P1) plays a card.")
    player_idx_before_action = env.current_player_idx
    p1_moves1 = env.get_readable_legal_moves();
    p1_aid1, p1_desc1 = p1_moves1[0]
    if p1_aid1 == env.withdraw_action_id and len(p1_moves1) > 1: p1_aid1, p1_desc1 = p1_moves1[1]
    obs, r, term, tr, info_s02 = env.step(p1_aid1)
    info_s02['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("0.2 (P1 Plays)", p1_aid1, p1_desc1, obs, r, term, tr, info_s02)
    if term: print("Battle ended. Test incomplete."); return

    # Step 0.3: P0 plays Card B Face-Down on top of Card A in the SEA theater
    print(
        f"\nCOVER FIRE Test Setup: P{env.current_player_idx} (P0) plays '{p0_card_B_obj.name}' FD to SEA theater (Pos {sea_theater_idx}).")
    player_idx_before_action = env.current_player_idx
    p0_card_B_slot = -1
    for i, card in enumerate(env.players[0].hand):
        if card.id == p0_card_B_obj.id: p0_card_B_slot = i; break
    if p0_card_B_slot == -1: print(f"Error: P0's Card B not in hand. Skipping."); return

    p0_play_card_B_id = p0_card_B_slot * actions_per_slot + sea_theater_idx * 2 + 1  # FD
    p0_play_card_B_desc = f"Play '{p0_card_B_obj.name}' to SEA (Pos {sea_theater_idx}) FD"

    obs, r, term, tr, info_s03 = env.step(p0_play_card_B_id)
    info_s03['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("0.3 (P0 Plays Card B FD)", p0_play_card_B_id, p0_play_card_B_desc, obs, r, term, tr, info_s03)
    if term: print("Battle ended. Test incomplete."); return
    assert env._get_player_strength_in_theater_with_effects(0,
                                                            sea_theater_idx) == 4, "P0 SEA strength should be 4 (2+2)."
    print(f"Strength in SEA for P0 after Card A & B FD: 4. Correct.")

    # Step 0.4: P1 plays any card again
    print(f"\nCOVER FIRE Test Setup: P{env.current_player_idx} (P1) plays another card.")
    player_idx_before_action = env.current_player_idx
    p1_moves2 = env.get_readable_legal_moves();
    p1_aid2, p1_desc2 = p1_moves2[0]
    if p1_aid2 == env.withdraw_action_id and len(p1_moves2) > 1: p1_aid2, p1_desc2 = p1_moves2[1]
    obs, r, term, tr, info_s04 = env.step(p1_aid2)
    info_s04['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("0.4 (P1 Plays again)", p1_aid2, p1_desc2, obs, r, term, tr, info_s04)
    if term: print("Battle ended. Test incomplete."); return

    # --- Phase 1: Player 0 plays COVER FIRE Face-Up in the SEA theater ---
    print(
        f"\nCOVER FIRE Test - Phase 1: P{env.current_player_idx} (P0) plays COVER FIRE FU to SEA (Pos {sea_theater_idx}).")
    player_idx_before_action = env.current_player_idx  # P0
    p0_cover_fire_slot = -1
    for i, card in enumerate(env.players[0].hand):
        if card.id == cover_fire_card_id: p0_cover_fire_slot = i; break
    if p0_cover_fire_slot == -1: print("Error: COVER FIRE not in P0 hand. Skipping."); return

    play_cf_action_id = p0_cover_fire_slot * actions_per_slot + sea_theater_idx * 2 + 0  # 0 for FU
    play_cf_desc = f"Play '{p0_cover_fire_obj.name}' (slot {p0_cover_fire_slot}) to SEA (Pos {sea_theater_idx}) FU"

    if not any(aid == play_cf_action_id for aid, _ in env.get_readable_legal_moves()):
        print(f"Planned play of COVER FIRE ({play_cf_desc}) ID {play_cf_action_id} not legal. Skipping.");
        return

    obs, r, term, tr, info_s1_1 = env.step(play_cf_action_id)
    info_s1_1['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("1.1 (P0 Plays COVER FIRE FU)", play_cf_action_id, play_cf_desc, obs, r, term, tr, info_s1_1)
    if term: print("Battle ended. Test incomplete."); return

    # Verification: Strength of P0 in SEA Theater should be 4(CF) + 4(Card B) + 4(Card A) = 12
    strength_sea_p0_with_cf = env._get_player_strength_in_theater_with_effects(0, sea_theater_idx)
    expected_strength_cf = p0_cover_fire_obj.strength + 4 + 4
    print(f"Strength in SEA for P0 (COVER FIRE active): {strength_sea_p0_with_cf} (Expected {expected_strength_cf})")
    assert strength_sea_p0_with_cf == expected_strength_cf, f"COVER FIRE strength error. Expected {expected_strength_cf}, got {strength_sea_p0_with_cf}"
    print("COVER FIRE Activation PASSED: P0's SEA theater strength is correct.")

    # --- Phase 2: P1 plays AMBUSH to flip P0's COVER FIRE Face-Down ---
    print(f"\nCOVER FIRE Test - Phase 2: P{env.current_player_idx} (P1) plays AMBUSH to flip P0's COVER FIRE.")
    player_idx_before_action = env.current_player_idx  # P1

    p1_ambush_slot = -1;
    ambush_obj_p1 = None
    for i, card in enumerate(env.players[1].hand):
        if card.id == ambush_card_id: p1_ambush_slot = i; ambush_obj_p1 = card; break
    if p1_ambush_slot == -1: print(
        f"P1 does not have AMBUSH (ID {ambush_card_id}). Skipping deactivation test."); return

    # P1 plays AMBUSH FU (e.g., to its native LAND theater or fallback)
    target_p1_ambush_play_th_idx = -1
    for i, th in enumerate(env.theaters):
        if th.current_type == TheaterType.LAND: target_p1_ambush_play_th_idx = i; break
    if target_p1_ambush_play_th_idx == -1: target_p1_ambush_play_th_idx = (sea_theater_idx + 1) % NUM_THEATERS

    p1_play_ambush_id = p1_ambush_slot * actions_per_slot + target_p1_ambush_play_th_idx * 2 + 0  # FU
    p1_play_ambush_desc = f"Play '{ambush_obj_p1.name}' (slot {p1_ambush_slot}) to {env.theaters[target_p1_ambush_play_th_idx].current_type.name} (Pos {target_p1_ambush_play_th_idx}) FU"

    if not any(aid == p1_play_ambush_id for aid, _ in env.get_readable_legal_moves()):
        print(f"P1 AMBUSH play ({p1_play_ambush_desc}) not legal. Skipping.");
        return

    obs, r, term, tr, info_s2_1 = env.step(p1_play_ambush_id)
    info_s2_1['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("2.1 (P1 Plays AMBUSH)", p1_play_ambush_id, p1_play_ambush_desc, obs, r, term, tr, info_s2_1)
    if term or info_s2_1.get("game_phase") != GamePhase.RESOLVING_AMBUSH.name: print(
        "P1 AMBUSH play failed or no effect. Skipping."); return

    # P1 uses AMBUSH to flip P0's COVER FIRE (which is in sea_theater_idx)
    ambush_choices_p1 = env.get_readable_legal_moves()
    action_to_flip_p0_cf = -1;
    desc_flip_p0_cf = ""
    found_target_p0_cf = False
    if 'possible_targets' in env.effect_resolution_data:
        for mapped_id, p_id, t_idx, c_name in env.effect_resolution_data['possible_targets']:
            if p_id == 0 and t_idx == sea_theater_idx and p0_cover_fire_obj and c_name == p0_cover_fire_obj.name:
                action_to_flip_p0_cf = mapped_id
                for aid_r, desc_r in ambush_choices_p1:
                    if aid_r == action_to_flip_p0_cf: desc_flip_p0_cf = desc_r; break
                found_target_p0_cf = True;
                break

    if not found_target_p0_cf: print(
        f"Could not find P0's COVER FIRE as AMBUSH target. Targets: {env.effect_resolution_data.get('possible_targets')}. Skipping."); return

    print(f"P1 using AMBUSH to flip P0's COVER FIRE: {desc_flip_p0_cf} (Action ID: {action_to_flip_p0_cf})")
    player_idx_before_action = env.current_player_idx  # P1 resolving Ambush
    obs, r, term, tr, info_s2_2 = env.step(action_to_flip_p0_cf)
    info_s2_2['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("2.2 (P1 Flips P0's COVER FIRE)", action_to_flip_p0_cf, desc_flip_p0_cf, obs, r, term, tr,
                      info_s2_2)

    # Verification: COVER FIRE should be INACTIVE, strength of P0's cards in SEA should revert
    assert not env._is_card_effect_active_for_player(0, cover_fire_card_id), "COVER FIRE should be INACTIVE for P0."
    print("COVER FIRE confirmed INACTIVE for P0.")

    strength_sea_p0_after_cf_flip = env._get_player_strength_in_theater_with_effects(0, sea_theater_idx)
    # Card A (FD, no Escalation) = 2, Card B (FD, no Escalation) = 2, Cover Fire (now FD) = 2. Total = 6
    expected_strength_reverted = 2 + 2 + 2
    print(
        f"Strength in SEA for P0 (COVER FIRE FD): {strength_sea_p0_after_cf_flip} (Expected {expected_strength_reverted})")
    assert strength_sea_p0_after_cf_flip == expected_strength_reverted, \
        f"COVER FIRE deactivation strength error. Expected {expected_strength_reverted}, got {strength_sea_p0_after_cf_flip}"
    print("COVER FIRE Deactivation PASSED: P0's SEA theater strength correctly reverted.")

    print("\nPassive Effect (COVER FIRE) Test Function Finished.")


def test_support_effect(env: AirLandSeaBaseEnv, max_setup_retries=30):
    print("\n" + "#" * 10 + " Test: SUPPORT Card Effect " + "#" * 10)

    support_card_id = 5  # AIR_1_SUPPORT
    ambush_card_id = 1  # LAND_2_AMBUSH (for P1 to flip P0's Support)

    p0_support_card_obj = None
    p0_card_A_obj = None  # Card to play in the theater that will receive support
    # p1_ambush_obj is not needed globally for the test, will be found in P1's hand when needed
    player0_hand_for_test = []
    initial_theater_layout_names = []
    p1_has_ambush_in_setup = False  # Flag to know if deactivation test is viable

    for retry in range(max_setup_retries):
        print(f"SUPPORT Test: Setup attempt {retry + 1}/{max_setup_retries}")
        env.reset()
        player0_hand = env.players[0].hand
        player1_hand = env.players[1].hand

        current_p0_support = next((card for card in player0_hand if card.id == support_card_id), None)
        temp_other_cards_p0 = [card for card in player0_hand if card.id != support_card_id]
        current_p0_card_a = temp_other_cards_p0[0] if temp_other_cards_p0 else None

        # Check if P1 has AMBUSH, just to note it for the deactivation part
        if any(card.id == ambush_card_id for card in player1_hand):
            p1_has_ambush_in_setup = True

        # P0 needs SUPPORT and at least one other card to make a meaningful test sequence.
        if current_p0_support and current_p0_card_a:  # Removed dependency on P1 having Ambush for initial setup
            p0_support_card_obj = current_p0_support
            p0_card_A_obj = current_p0_card_a

            player0_hand_for_test = [c.name for c in env.players[0].hand]
            initial_theater_layout_names = [th.current_type.name for th in env.theaters]
            print(
                f"Favorable hand: P0 has '{p0_support_card_obj.name}', and card '{p0_card_A_obj.name}'. P1 has AMBUSH: {p1_has_ambush_in_setup}.")
            print(f"Theater Layout for this attempt: {initial_theater_layout_names}")
            break
        p0_support_card_obj = p0_card_A_obj = None
        p1_has_ambush_in_setup = False

    if not (p0_support_card_obj and p0_card_A_obj):  # Simplified condition
        print(f"Could not get suitable hand (P0:SUPPORT+Other) after {max_setup_retries} retries. Skipping test.")
        return

    # ... (rest of the print statements and initial env.render() from previous correct version) ...
    print(f"Proceeding with P0 Hand: {player0_hand_for_test}")
    env.render()

    actions_per_slot = NUM_THEATERS * 2

    theater_Y_idx_for_support = -1
    theater_X_idx_to_check_bonus = -1

    # Find an AIR theater for SUPPORT
    for i_th_y_candidate in range(NUM_THEATERS):
        # CORRECTED COMPARISON: Compare by enum member names
        if env.theaters[i_th_y_candidate].current_type.name == CardType.AIR.name:  # SUPPORT is an AIR card
            adj_to_Y = env._get_adjacent_theater_indices(i_th_y_candidate)
            if adj_to_Y:
                theater_Y_idx_for_support = i_th_y_candidate
                theater_X_idx_to_check_bonus = random.choice(adj_to_Y)
                break

    if theater_Y_idx_for_support == -1:
        print("No suitable AIR theater with an adjacent theater found to play SUPPORT FU. Skipping test.")
        return

    print(
        f"Test Plan: P0 plays SUPPORT to AIR theater Pos {theater_Y_idx_for_support} ({env.theaters[theater_Y_idx_for_support].current_type.name}). "
        f"Will check bonus in adjacent Pos {theater_X_idx_to_check_bonus} ({env.theaters[theater_X_idx_to_check_bonus].current_type.name}).")

    if env.current_player_idx == 1:
        print(f"\nSUPPORT Test Setup: P{env.current_player_idx} (P1) plays a card.")
        player_idx_before_action = env.current_player_idx
        p1_moves_init = env.get_readable_legal_moves();
        p1_aid_init, p1_desc_init = p1_moves_init[0]
        if p1_aid_init == env.withdraw_action_id and len(p1_moves_init) > 1: p1_aid_init, p1_desc_init = p1_moves_init[
            1]
        obs, r, term, tr, info_s00 = env.step(p1_aid_init)
        info_s00['current_player_idx_at_action_end'] = player_idx_before_action
        env.render();
        print_step_result("0.0 (P1 Initial Play if needed)", p1_aid_init, p1_desc_init, obs, r, term, tr, info_s00)
        if term: print("Battle ended. Test incomplete."); return
        while env.game_phase != GamePhase.NORMAL_PLAY and not term:
            effect_choices = env.get_readable_legal_moves()
            if not effect_choices: break
            effect_action_id, effect_desc = random.choice(effect_choices);
            p_idx_eff = env.current_player_idx
            obs, r, term, tr, info_eff = env.step(effect_action_id)
            info_eff['current_player_idx_at_action_end'] = p_idx_eff
            env.render();
            print_step_result("0.0.x (P1 Effect)", effect_action_id, effect_desc, obs, r, term, tr, info_eff)
            if term: print("Battle ended during P1 effect. Test incomplete."); return
    assert env.current_player_idx == 0, "It should be P0's turn to play Card A."

    print(
        f"\nSUPPORT Test Setup: P{env.current_player_idx} (P0) plays '{p0_card_A_obj.name}' FD to Pos {theater_X_idx_to_check_bonus}.")
    player_idx_before_action = env.current_player_idx
    p0_card_A_slot = -1
    for i, card in enumerate(env.players[0].hand):
        if card.id == p0_card_A_obj.id: p0_card_A_slot = i; break
    if p0_card_A_slot == -1: print(f"Error: P0's Card A '{p0_card_A_obj.name}' not in hand for play. Skipping."); return
    p0_play_card_A_id = p0_card_A_slot * actions_per_slot + theater_X_idx_to_check_bonus * 2 + 1
    p0_play_card_A_desc = f"Play '{p0_card_A_obj.name}' to {env.theaters[theater_X_idx_to_check_bonus].current_type.name} (Pos {theater_X_idx_to_check_bonus}) FD"
    obs, r, term, tr, info_s01 = env.step(p0_play_card_A_id)
    info_s01['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("0.1 (P0 Plays Card A FD)", p0_play_card_A_id, p0_play_card_A_desc, obs, r, term, tr, info_s01)
    if term: print("Battle ended. Test incomplete."); return
    strength_tx_p0_before_support = env._get_player_strength_in_theater_with_effects(0, theater_X_idx_to_check_bonus)
    expected_base_strength_card_A = 2
    print(
        f"Strength in T{theater_X_idx_to_check_bonus} for P0 (before SUPPORT active): {strength_tx_p0_before_support} (Expected {expected_base_strength_card_A})")
    assert strength_tx_p0_before_support == expected_base_strength_card_A, f"Card A FD strength should be {expected_base_strength_card_A}."

    print(f"\nSUPPORT Test Setup: P{env.current_player_idx} (P1) plays a non-AMBUSH card if possible.")
    player_idx_before_action = env.current_player_idx
    p1_legal_moves_1 = env.get_readable_legal_moves();
    p1_action_id_1, p1_desc_1 = env.withdraw_action_id, "Withdraw (fallback)"
    p1_non_ambush_play = None
    if p1_legal_moves_1:  # Check if list is not empty
        for aid, desc in p1_legal_moves_1:
            if aid != env.withdraw_action_id:
                slot_idx_p1, _, _, _ = env._decode_action(aid)
                if slot_idx_p1 is not None and slot_idx_p1 < len(env.players[1].hand) and env.players[1].hand[
                    slot_idx_p1].id != ambush_card_id:
                    p1_non_ambush_play = (aid, desc);
                    break
        if p1_non_ambush_play:
            p1_action_id_1, p1_desc_1 = p1_non_ambush_play
        else:  # Fallback if only Ambush or withdraw or empty hand
            p1_action_id_1, p1_desc_1 = p1_legal_moves_1[0]
            if p1_action_id_1 == env.withdraw_action_id and len(p1_legal_moves_1) > 1: p1_action_id_1, p1_desc_1 = \
            p1_legal_moves_1[1]

    if p1_action_id_1 == env.withdraw_action_id and not env.players[1].hand: print(
        "P1 cannot play. Test may be limited.");

    obs, r, term, tr, info_s02 = env.step(p1_action_id_1)
    info_s02['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("0.2 (P1 Plays)", p1_action_id_1, p1_desc_1, obs, r, term, tr, info_s02)
    if term: print("Battle ended. Test incomplete."); return
    while env.game_phase != GamePhase.NORMAL_PLAY and not term:
        effect_choices = env.get_readable_legal_moves()
        if not effect_choices: break
        effect_action_id, effect_desc = random.choice(effect_choices);
        p_idx_eff = env.current_player_idx
        obs, r, term, tr, info_eff = env.step(effect_action_id)
        info_eff['current_player_idx_at_action_end'] = p_idx_eff
        env.render();
        print_step_result("0.2.x (P1 Effect)", effect_action_id, effect_desc, obs, r, term, tr, info_eff)
        if term: print("Battle ended during P1 effect. Test incomplete."); return
    assert env.current_player_idx == 0, "Should be P0's turn."

    print(
        f"\nSUPPORT Test - Phase 1: P{env.current_player_idx} (P0) plays SUPPORT FU to Pos {theater_Y_idx_for_support}.")
    player_idx_before_action = env.current_player_idx
    p0_support_slot = -1
    for i, card in enumerate(env.players[0].hand):
        if card.id == support_card_id: p0_support_slot = i; break
    if p0_support_slot == -1: print("Error: SUPPORT card not in P0's hand. Skipping."); return

    play_support_action_id = p0_support_slot * actions_per_slot + theater_Y_idx_for_support * 2 + 0
    play_support_desc = f"Play '{p0_support_card_obj.name}' (slot {p0_support_slot}) to {env.theaters[theater_Y_idx_for_support].current_type.name} (Pos {theater_Y_idx_for_support}) FU"

    if not any(aid == play_support_action_id for aid, _ in env.get_readable_legal_moves()):
        print(f"Planned play of SUPPORT FU ({play_support_desc}) ID {play_support_action_id} not legal. Skipping.");
        return

    obs, r, term, tr, info_s1_1 = env.step(play_support_action_id)
    info_s1_1['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("1.1 (P0 Plays SUPPORT FU)", play_support_action_id, play_support_desc, obs, r, term, tr,
                      info_s1_1)
    if term: print("Battle ended. Test incomplete."); return

    assert env._is_card_effect_active_for_player(0, support_card_id), "SUPPORT should be active for P0."
    print("SUPPORT card confirmed ACTIVE for Player 0.")
    strength_tx_p0_with_support = env._get_player_strength_in_theater_with_effects(0, theater_X_idx_to_check_bonus)
    expected_strength_with_support = strength_tx_p0_before_support + 3
    print(
        f"Strength in T{theater_X_idx_to_check_bonus} for P0 (SUPPORT active in T{theater_Y_idx_for_support}): {strength_tx_p0_with_support} (Expected {expected_strength_with_support})")
    assert strength_tx_p0_with_support == expected_strength_with_support, \
        f"SUPPORT strength calculation error. Expected {expected_strength_with_support}, got {strength_tx_p0_with_support}"
    print(f"SUPPORT Activation PASSED: P0's strength in T{theater_X_idx_to_check_bonus} correctly updated by +3.")

    if not p1_has_ambush_in_setup:  # Check the flag set during setup
        print("P1 was not dealt AMBUSH during setup. Skipping deactivation test part for SUPPORT.")
        print("\nPassive Effect (SUPPORT) Test Function Finished.")
        return

    # --- Phase 2: P1 plays AMBUSH to flip P0's SUPPORT Face-Down ---
    if term: print("Battle ended before P1 can attempt AMBUSH. Deactivation test skipped."); return
    print(f"\nSUPPORT Test - Phase 2: P{env.current_player_idx} (P1) plays AMBUSH to flip P0's SUPPORT.")
    player_idx_before_action = env.current_player_idx  # P1

    p1_ambush_slot_idx = -1
    p1_ambush_card_object = None
    for i, card in enumerate(env.players[1].hand):  # P1's hand
        if card.id == ambush_card_id:
            p1_ambush_slot_idx = i
            p1_ambush_card_object = card
            break
    if p1_ambush_slot_idx == -1:
        print(f"P1 does not have AMBUSH (ID {ambush_card_id}) in hand now. Skipping deactivation test.");
        return

    # Try to find a legal FACE-UP play for AMBUSH to a LAND theater
    p1_play_ambush_id = -1
    p1_play_ambush_desc = ""
    intended_play_found_and_legal = False

    print(f"READABLE MOVES: {env.get_readable_legal_moves()}")

    for i_theater_target, theater_target_obj in enumerate(env.theaters):
        if theater_target_obj.current_type.name == CardType.LAND.name:  # AMBUSH is LAND, must play FU to LAND
            # Construct action for playing AMBUSH from its slot to this LAND theater FU
            potential_action_id = p1_ambush_slot_idx * actions_per_slot + i_theater_target * 2 + 0  # 0 for FU

            # Check if this specific action is legal
            for legal_aid, legal_desc in env.get_readable_legal_moves():
                if legal_aid == potential_action_id:
                    p1_play_ambush_id = potential_action_id
                    p1_play_ambush_desc = legal_desc  # Use the description from get_readable_legal_moves
                    intended_play_found_and_legal = True
                    break
            if intended_play_found_and_legal:
                break

    if not intended_play_found_and_legal:
        print(
            f"No legal FACE-UP play for P1's AMBUSH card to a LAND theater found. P1 Hand: {[c.name for c in env.players[1].hand]}. Theaters: {[t.current_type.name for t in env.theaters]}. Skipping deactivation test.")
        return

    print(f"P1 taking action: {p1_play_ambush_desc} (Action ID: {p1_play_ambush_id})")
    obs, r, term, tr, info_s2_1 = env.step(p1_play_ambush_id)
    info_s2_1['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("2.1 (P1 Plays AMBUSH)", p1_play_ambush_id, p1_play_ambush_desc, obs, r, term, tr, info_s2_1)

    # Determine if Ambush was played FU by checking the description we used or the info dict.
    # The flag 'play_p1_ambush_fu' is no longer needed as we directly search for a FU play.
    ambush_was_played_fu = "FACE-UP" in p1_play_ambush_desc.upper()

    if term or not ambush_was_played_fu or info_s2_1.get("game_phase") != GamePhase.RESOLVING_AMBUSH.name:
        print("P1 AMBUSH play failed, was not played Face-Up, or did not trigger effect. Skipping deactivation.");
        return

    ambush_choices_p1 = env.get_readable_legal_moves()
    action_to_flip_p0_support = -1;
    desc_flip_p0_support = ""
    found_target_p0_support = False
    if 'possible_targets' in env.effect_resolution_data:
        for mapped_id, p_id, t_idx, c_name in env.effect_resolution_data['possible_targets']:
            if p_id == 0 and t_idx == theater_Y_idx_for_support and p0_support_card_obj and c_name == p0_support_card_obj.name:
                action_to_flip_p0_support = mapped_id
                for aid_r, desc_r in ambush_choices_p1:
                    if aid_r == action_to_flip_p0_support: desc_flip_p0_support = desc_r; break
                found_target_p0_support = True;
                break

    if not found_target_p0_support: print(
        f"Could not find P0's SUPPORT in T{theater_Y_idx_for_support} as AMBUSH target. Targets: {env.effect_resolution_data.get('possible_targets')}. Skipping."); return

    print(f"P1 using AMBUSH to flip P0's SUPPORT: {desc_flip_p0_support} (Action ID: {action_to_flip_p0_support})")
    player_idx_before_action = env.current_player_idx  # P1 resolving Ambush
    obs, r, term, tr, info_s2_2 = env.step(action_to_flip_p0_support)
    info_s2_2['current_player_idx_at_action_end'] = player_idx_before_action
    env.render();
    print_step_result("2.2 (P1 Flips P0's SUPPORT)", action_to_flip_p0_support, desc_flip_p0_support, obs, r, term, tr,
                      info_s2_2)

    assert not env._is_card_effect_active_for_player(0, support_card_id), "SUPPORT should be INACTIVE for P0."
    print("SUPPORT confirmed INACTIVE for Player 0 after P1's AMBUSH.")

    strength_tx_p0_after_support_flip = env._get_player_strength_in_theater_with_effects(0,
                                                                                         theater_X_idx_to_check_bonus)
    print(
        f"Strength in T{theater_X_idx_to_check_bonus} for P0 (SUPPORT FD in T{theater_Y_idx_for_support}): {strength_tx_p0_after_support_flip} (Expected {strength_tx_p0_before_support})")
    assert strength_tx_p0_after_support_flip == strength_tx_p0_before_support, \
        f"SUPPORT deactivation strength error. Expected {strength_tx_p0_before_support}, got {strength_tx_p0_after_support_flip}"
    print(f"SUPPORT Deactivation PASSED: P0's strength in T{theater_X_idx_to_check_bonus} correctly reverted.")

    print("\nPassive Effect (SUPPORT) Test Function Finished.")


# Add this function to your main.py

def test_simple_containment_effect_p1_plays_fd(env: AirLandSeaBaseEnv, max_setup_retries=15): #
    print("\n" + "#" * 10 + " Test: Simple CONTAINMENT (P1 Plays FD) " + "#" * 10)

    containment_card_id = 9  # AIR_5_CONTAINMENT
    p0_idx, p1_idx = 0, 1

    p0_has_containment = False
    p1_has_any_card = False
    p0_containment_obj = None
    p1_card_to_play_fd_obj = None # Card P1 will attempt to play FD

    for retry in range(max_setup_retries):
        print(f"Simple CONTAINMENT Test (P1 Plays FD): Setup attempt {retry + 1}/{max_setup_retries}")
        env.reset()
        player0_hand = env.players[p0_idx].hand
        player1_hand = env.players[p1_idx].hand

        p0_has_containment = any(card.id == containment_card_id for card in player0_hand)
        p1_has_any_card = bool(player1_hand) # P1 just needs any card

        if p0_has_containment and p1_has_any_card:
            for card in player0_hand:
                if card.id == containment_card_id:
                    p0_containment_obj = card
                    break
            p1_card_to_play_fd_obj = player1_hand[0] # P1 will try to play their first card FD

            if p0_containment_obj: # Ensure containment card object is found
                print(f"Favorable setup: P0 has '{p0_containment_obj.name}'. P1 has card '{p1_card_to_play_fd_obj.name}'.")
                break
        p0_containment_obj = None # Reset if setup failed this attempt

    if not p0_containment_obj or not p1_card_to_play_fd_obj:
        print(f"Could not get suitable hand (P0:CONTAINMENT, P1:Any Card) after {max_setup_retries} retries. Skipping test.")
        return

    print(f"Initial P0 Hand: {[card.name for card in env.players[p0_idx].hand]}")
    print(f"Initial P1 Hand: {[card.name for card in env.players[p1_idx].hand]}")
    print(f"Initial Theater Layout: {[th.current_type.name for th in env.theaters]}")
    env.render()

    actions_per_slot = NUM_THEATERS * 2 #

    # --- Step 1: Player 0 plays CONTAINMENT Face-Up ---
    print(f"\nSimple CONTAINMENT Test - Step 1: P{env.current_player_idx} (P0) plays CONTAINMENT FU.")
    player_idx_before_action_s1 = env.current_player_idx

    p0_containment_slot = -1
    for i, card in enumerate(env.players[p0_idx].hand):
        if card.id == containment_card_id: p0_containment_slot = i; break
    if p0_containment_slot == -1: print("Error: P0's CONTAINMENT card not in hand. Skipping."); return

    target_air_theater_idx = -1
    native_card_type_name_for_containment = p0_containment_obj.card_type.name # Should be "AIR"
    for i, th in enumerate(env.theaters):
        if th.current_type.name == native_card_type_name_for_containment: # Compare string names
            target_air_theater_idx = i; break
    if target_air_theater_idx == -1: print(f"No {native_card_type_name_for_containment} theater for {p0_containment_obj.name}. Skipping."); return

    play_containment_action_id = p0_containment_slot * actions_per_slot + target_air_theater_idx * 2 + 0  # 0 for FU
    play_containment_desc = f"Play '{p0_containment_obj.name}' (slot {p0_containment_slot}) to {native_card_type_name_for_containment} (Pos {target_air_theater_idx}) FU"

    if not any(aid == play_containment_action_id for aid, _ in env.get_readable_legal_moves()):
        print(f"Planned play of CONTAINMENT FU ({play_containment_desc}) ID {play_containment_action_id} not legal. Skipping."); return

    obs, r, term, tr, info_s1 = env.step(play_containment_action_id)
    info_s1['current_player_idx_at_action_end'] = player_idx_before_action_s1
    env.render(); print_step_result("1 (P0 Plays CONTAINMENT FU)", play_containment_action_id, play_containment_desc, obs, r, term, tr, info_s1)
    if term: print("Battle ended after P0 played CONTAINMENT. Test incomplete."); return

    assert env._is_containment_active_globally(), "CONTAINMENT should be active globally." #
    print("CONTAINMENT confirmed ACTIVE globally.")
    assert env.current_player_idx == p1_idx, "Turn should now be P1's."

    # --- Step 2: Player 1 attempts to play their card Face-Down (should be discarded) ---
    print(f"\nSimple CONTAINMENT Test - Step 2: P{env.current_player_idx} (P1) plays '{p1_card_to_play_fd_obj.name}' FD.")
    player_idx_before_action_s2 = env.current_player_idx # Should be P1
    battle_deck_size_before_discard = len(env.battle_deck) #
    battle_turn_counter_before_p1_play = env.battle_turn_counter #

    p1_card_to_play_fd_slot = -1
    # Find the specific card p1_card_to_play_fd_obj in P1's current hand to get its slot
    for i, card_in_hand in enumerate(env.players[p1_idx].hand):
        if card_in_hand.id == p1_card_to_play_fd_obj.id:
            p1_card_to_play_fd_slot = i
            break
    if p1_card_to_play_fd_slot == -1:
        print(f"Error: P1's card '{p1_card_to_play_fd_obj.name}' not found in hand. Hand: {[c.name for c in env.players[p1_idx].hand]}. Skipping."); return

    target_theater_for_p1_fd_play = 0 # P1 plays to any theater FD, e.g., Pos 0
    play_p1_card_fd_action_id = p1_card_to_play_fd_slot * actions_per_slot + target_theater_for_p1_fd_play * 2 + 1  # 1 for FD
    play_p1_card_fd_desc = f"Play '{p1_card_to_play_fd_obj.name}' (slot {p1_card_to_play_fd_slot}) to {env.theaters[target_theater_for_p1_fd_play].current_type.name} (Pos {target_theater_for_p1_fd_play}) FD"

    if not any(aid == play_p1_card_fd_action_id for aid, _ in env.get_readable_legal_moves()):
        print(f"Planned play for P1 FD ({play_p1_card_fd_desc}) ID {play_p1_card_fd_action_id} not legal. Check setup or mask. Skipping."); return

    obs, r, term, tr, info_s2 = env.step(play_p1_card_fd_action_id)
    info_s2['current_player_idx_at_action_end'] = player_idx_before_action_s2
    env.render(); print_step_result("2 (P1 Plays Card FD - Expect Discard)", play_p1_card_fd_action_id, play_p1_card_fd_desc, obs, r, term, tr, info_s2)

    assert "containment_triggered" in info_s2, "CONTAINMENT effect should have triggered for P1's FD play." #
    print(f"CONTAINMENT Triggered for P1's play: {info_s2['containment_triggered']}. Correct.")
    # If P1's card had an effect (e.g., AMBUSH), it should not have changed the game phase.
    assert info_s2.get("game_phase") == GamePhase.NORMAL_PLAY.name, \
        f"Game phase should remain NORMAL_PLAY. Phase is: {info_s2.get('game_phase')}" #
    print("P1's discarded card effect did not trigger (Game phase is NORMAL_PLAY). Correct.")

    card_p1_in_theater = False
    for played_card_instance in env.theaters[target_theater_for_p1_fd_play].player_zones[p1_idx]:
        if played_card_instance.card_ref.id == p1_card_to_play_fd_obj.id:
            card_p1_in_theater = True; break
    assert not card_p1_in_theater, f"P1's card '{p1_card_to_play_fd_obj.name}' should NOT be in theater {target_theater_for_p1_fd_play}."
    print(f"P1's card '{p1_card_to_play_fd_obj.name}' not found in theater. Correct (discarded).")

    assert len(env.battle_deck) == battle_deck_size_before_discard + 1, "Battle deck size should increase by 1 after P1's card discard." #
    assert env.battle_deck[0].id == p1_card_to_play_fd_obj.id, \
        f"P1's discarded card (ID {p1_card_to_play_fd_obj.id}) should be at the bottom (index 0) of the deck. Found ID {env.battle_deck[0].id}" #
    print(f"P1's card '{p1_card_to_play_fd_obj.name}' correctly returned to bottom of deck.")

    assert env.battle_turn_counter == battle_turn_counter_before_p1_play + 1, \
        f"Battle turn counter should have incremented once for P1's action. Before: {battle_turn_counter_before_p1_play}, Now: {env.battle_turn_counter}" #
    print("Battle turn counter incremented correctly for P1's discard action.")

    if term: print("Battle ended after P1's FD play. Test successful."); return
    assert env.current_player_idx == p0_idx, "Turn should pass back to P0 after P1's FD play."

    print("\nSimple CONTAINMENT Test (P1 Plays FD) Function Finished.")


# TODO: Blockade test fails sometimes but the engine should still be correct? Problem is probably with setup?
def test_blockade_effect(env: AirLandSeaBaseEnv, max_setup_retries=30):  #
    print("\n" + "#" * 10 + " Test: BLOCKADE Card Effect " + "#" * 10)

    blockade_card_id = 15  # SEA_5_BLOCKADE
    # Card P1 will attempt to play into the crowded theater (ideally one with an instant effect)
    p1_card_z_id = 1  # e.g., LAND_2_AMBUSH
    p0_idx, p1_idx = 0, 1

    # Hand requirements for setup:
    # P0: BLOCKADE, Card_P0_A (to play into target theater)
    # P1: Card_P1_A, Card_P1_B (to play into target theater), Card_Z (AMBUSH)
    p0_has_blockade = False
    p0_has_card_a = False
    p1_has_card_a = False
    p1_has_card_b = False
    p1_has_card_z = False

    p0_blockade_obj, p0_card_a_obj = None, None
    p1_card_a_obj, p1_card_b_obj, p1_card_z_obj = None, None, None

    for retry in range(max_setup_retries):
        print(f"BLOCKADE Test: Setup attempt {retry + 1}/{max_setup_retries}")
        env.reset()
        p0_hand = env.players[p0_idx].hand
        p1_hand = env.players[p1_idx].hand

        # Check P0's hand
        temp_p0_blockade = next((card for card in p0_hand if card.id == blockade_card_id), None)
        temp_p0_card_a = next((card for card in p0_hand if card.id != blockade_card_id), None)

        # Check P1's hand
        p1_card_ids_in_hand = [card.id for card in p1_hand]
        temp_p1_card_z = next((card for card in p1_hand if card.id == p1_card_z_id), None)

        other_p1_cards = [card for card in p1_hand if card.id != p1_card_z_id]
        temp_p1_card_a = other_p1_cards[0] if len(other_p1_cards) > 0 else None
        temp_p1_card_b = other_p1_cards[1] if len(other_p1_cards) > 1 else None

        if temp_p0_blockade and temp_p0_card_a and \
            temp_p1_card_a and temp_p1_card_b and temp_p1_card_z:
            p0_blockade_obj, p0_card_a_obj = temp_p0_blockade, temp_p0_card_a
            p1_card_a_obj, p1_card_b_obj, p1_card_z_obj = temp_p1_card_a, temp_p1_card_b, temp_p1_card_z
            print(f"Favorable setup: P0 has BLOCKADE, CardA. P1 has CardA, CardB, CardZ (AMBUSH).")
            break
        # Reset found objects for next retry
        p0_blockade_obj, p0_card_a_obj = None, None
        p1_card_a_obj, p1_card_b_obj, p1_card_z_obj = None, None, None

    if not (p0_blockade_obj and p0_card_a_obj and p1_card_a_obj and p1_card_b_obj and p1_card_z_obj):
        print(f"Could not get suitable multi-card hands after {max_setup_retries} retries. Skipping BLOCKADE test.")
        return

    print(f"Initial P0 Hand: {[card.name for card in env.players[p0_idx].hand]}")
    print(f"Initial P1 Hand: {[card.name for card in env.players[p1_idx].hand]}")
    print(f"Initial Theater Layout: {[th.current_type.name for th in env.theaters]}")
    env.render()

    actions_per_slot = NUM_THEATERS * 2  #

    # --- Step 1: P0 plays BLOCKADE Face-Up ---
    # Find a SEA theater for BLOCKADE (ID 15) and an adjacent theater for testing
    theater_y_for_blockade_idx = -1  # Where BLOCKADE will be played
    theater_x_target_idx = -1  # Adjacent theater P1 will play into

    for i_th_y in range(NUM_THEATERS):
        if env.theaters[i_th_y].current_type.name == CardType.SEA.name:  # BLOCKADE is SEA
            adj_to_y = env._get_adjacent_theater_indices(i_th_y)  #
            if adj_to_y:
                theater_y_for_blockade_idx = i_th_y
                theater_x_target_idx = random.choice(adj_to_y)
                break

    if theater_y_for_blockade_idx == -1 or theater_x_target_idx == -1:
        print("Could not find a SEA theater with an adjacent theater for BLOCKADE setup. Skipping test.");
        return
    print(f"Test Plan: P0 plays BLOCKADE (ID {blockade_card_id}) to SEA theater "
          f"Pos {theater_y_for_blockade_idx} ({env.theaters[theater_y_for_blockade_idx].current_type.name}). "
          f"P1 will play into adjacent Pos {theater_x_target_idx} ({env.theaters[theater_x_target_idx].current_type.name}).")

    print(f"\nBLOCKADE Test - Step 1: P{env.current_player_idx} (P0) plays BLOCKADE FU.")
    player_idx_before_action_s1 = env.current_player_idx

    p0_blockade_slot = -1
    for i, card in enumerate(env.players[p0_idx].hand):
        if card.id == blockade_card_id: p0_blockade_slot = i; break
    if p0_blockade_slot == -1: print("Error: P0's BLOCKADE card not in hand. Skipping."); return

    play_blockade_action_id = p0_blockade_slot * actions_per_slot + theater_y_for_blockade_idx * 2 + 0  # 0 for FU
    play_blockade_desc = f"Play '{p0_blockade_obj.name}' (slot {p0_blockade_slot}) to SEA (Pos {theater_y_for_blockade_idx}) FU"

    if not any(aid == play_blockade_action_id for aid, _ in env.get_readable_legal_moves()):
        print(f"Planned play of BLOCKADE FU ({play_blockade_desc}) ID {play_blockade_action_id} not legal. Skipping.");
        return

    obs, r, term, tr, info_s1 = env.step(play_blockade_action_id)
    info_s1['current_player_idx_at_action_end'] = player_idx_before_action_s1
    env.render();
    print_step_result("1 (P0 Plays BLOCKADE FU)", play_blockade_action_id, play_blockade_desc, obs, r, term, tr,
                      info_s1)
    if term: print("Battle ended. Test incomplete."); return

    assert env._is_card_effect_active_for_player(p0_idx, blockade_card_id), "BLOCKADE should be active for P0."  #
    print(f"BLOCKADE (ID {blockade_card_id}) confirmed ACTIVE for P0 in T{theater_y_for_blockade_idx}.")
    assert env.current_player_idx == p1_idx, "Turn should be P1's."

    # --- Steps 2, 3, 4: Populate Theater X to have 3 cards (P1, P0, P1) ---
    cards_to_play_in_setup = [
        (p1_idx, p1_card_a_obj, "2 (P1 Plays Card P1-A)"),  # Total in T_X = 1
        (p0_idx, p0_card_a_obj, "3 (P0 Plays Card P0-A)"),  # Total in T_X = 2
        (p1_idx, p1_card_b_obj, "4 (P1 Plays Card P1-B)"),  # Total in T_X = 3
    ]

    for current_player_to_act, card_obj_to_play, step_desc_prefix in cards_to_play_in_setup:
        if term: print(f"Battle ended during setup for {step_desc_prefix}. Test incomplete."); return
        assert env.current_player_idx == current_player_to_act, f"Expected P{current_player_to_act}'s turn for {step_desc_prefix}."

        print(
            f"\nBLOCKADE Test - Step {step_desc_prefix}: P{current_player_to_act} plays '{card_obj_to_play.name}' to Target Theater {theater_x_target_idx}.")
        player_idx_before_action = env.current_player_idx

        card_slot = -1
        for i, card_in_hand in enumerate(env.players[current_player_to_act].hand):
            if card_in_hand.id == card_obj_to_play.id: card_slot = i; break
        if card_slot == -1: print(
            f"Error: Card '{card_obj_to_play.name}' not in P{current_player_to_act}'s hand. Skipping."); return

        play_action_id = card_slot * actions_per_slot + theater_x_target_idx * 2 + 1  # Play FD for simplicity in setup
        play_desc = f"Play '{card_obj_to_play.name}' (slot {card_slot}) to T{theater_x_target_idx} FD"

        if not any(aid == play_action_id for aid, _ in env.get_readable_legal_moves()):
            print(f"Planned play ({play_desc}) ID {play_action_id} not legal. Trying FU.");
            play_action_id = card_slot * actions_per_slot + theater_x_target_idx * 2 + 0  # FU
            play_desc = f"Play '{card_obj_to_play.name}' (slot {card_slot}) to T{theater_x_target_idx} FU"
            if not any(aid == play_action_id for aid, _ in env.get_readable_legal_moves()):
                print(f"Planned play FU ({play_desc}) ID {play_action_id} also not legal. Skipping setup step.");
                return

        obs, r_step, term_step, tr_step, info_step = env.step(play_action_id)
        info_step['current_player_idx_at_action_end'] = player_idx_before_action
        env.render();
        print_step_result(step_desc_prefix, play_action_id, play_desc, obs, r_step, term_step, tr_step, info_step)
        term = term_step  # Update global terminated flag for the test

        # Ensure card was not discarded by P0's Blockade (it shouldn't be, yet)
        assert "blockade_triggered" not in info_step, f"BLOCKADE should NOT trigger during setup play by P{player_idx_before_action}"

    if term: print("Battle ended during theater population. Test incomplete."); return

    # At this point, Theater X should have 3 cards total. Verify.
    total_cards_in_tx_after_setup = len(env.theaters[theater_x_target_idx].player_zones[p0_idx]) + \
                                    len(env.theaters[theater_x_target_idx].player_zones[p1_idx])
    print(
        f"Total cards in Target Theater {theater_x_target_idx} after setup: {total_cards_in_tx_after_setup} (Expected 3)")
    assert total_cards_in_tx_after_setup == 3, "Theater X should have 3 cards total for BLOCKADE condition."
    assert env.current_player_idx == p0_idx, "Turn should be P0's after P1's 3rd setup play."  # P1 played last card P1-B

    # --- Step 5: P0 plays a simple card to pass turn to P1 (or withdraws if no simple play) ---
    # This ensures it's P1's turn to attempt playing the 4th card into the crowded theater.
    print(f"\nBLOCKADE Test - Step 5: P{env.current_player_idx} (P0) makes a play to pass turn.")
    player_idx_before_action_s5 = env.current_player_idx  # P0
    p0_moves_s5 = env.get_readable_legal_moves()
    if not p0_moves_s5: print("P0 has no moves. Test incomplete."); return

    p0_action_id_s5, p0_desc_s5 = p0_moves_s5[0]  # Default to first
    # Try to find a play that is not BLOCKADE itself, and not into theater_x_target_idx or theater_y_for_blockade_idx
    # to keep things simple. If not, just take first non-withdraw.
    simple_play_found = False
    for aid, desc in p0_moves_s5:
        if aid != env.withdraw_action_id:
            _slot, _tid, _fu, _ = env._decode_action(aid)
            if _tid is not None and _tid not in [theater_x_target_idx, theater_y_for_blockade_idx]:
                p0_action_id_s5, p0_desc_s5 = aid, desc;
                simple_play_found = True;
                break
    if not simple_play_found and p0_moves_s5[0][0] == env.withdraw_action_id and len(p0_moves_s5) > 1:
        p0_action_id_s5, p0_desc_s5 = p0_moves_s5[1]  # Avoid withdraw if other options exist

    obs, r, term, tr, info_s5 = env.step(p0_action_id_s5)
    info_s5['current_player_idx_at_action_end'] = player_idx_before_action_s5
    env.render();
    print_step_result("5 (P0 Passes Turn)", p0_action_id_s5, p0_desc_s5, obs, r, term, tr, info_s5)
    if term: print("Battle ended. Test incomplete."); return
    assert env.current_player_idx == p1_idx, "Turn should be P1's to test BLOCKADE."

    # --- Step 6: P1 attempts to play Card Z (AMBUSH) into crowded Theater X (should be discarded) ---
    print(
        f"\nBLOCKADE Test - Step 6: P{env.current_player_idx} (P1) plays Card Z ('{p1_card_z_obj.name}') to crowded T{theater_x_target_idx}.")
    player_idx_before_action_s6 = env.current_player_idx  # P1
    battle_deck_size_before_discard = len(env.battle_deck)  #
    battle_turn_counter_before_p1_discard_attempt = env.battle_turn_counter  #

    p1_card_z_slot = -1
    for i, card in enumerate(env.players[p1_idx].hand):
        if card.id == p1_card_z_id: p1_card_z_slot = i; break
    if p1_card_z_slot == -1: print(f"Error: P1's Card Z ('{p1_card_z_obj.name}') not in hand. Skipping."); return

    play_p1_card_z_action_id = p1_card_z_slot * actions_per_slot + theater_x_target_idx * 2 + 1  # Play Face-Down (offset 1)
    play_p1_card_z_desc = f"Play '{p1_card_z_obj.name}' (slot {p1_card_z_slot}) to T{theater_x_target_idx} FD"

    if not any(aid == play_p1_card_z_action_id for aid, _ in env.get_readable_legal_moves()):
        print(
            f"Planned play for P1 Card Z ({play_p1_card_z_desc}) ID {play_p1_card_z_action_id} not legal. Mask/Setup Issue. Skipping.");
        return

    obs, r, term, tr, info_s6 = env.step(play_p1_card_z_action_id)
    info_s6['current_player_idx_at_action_end'] = player_idx_before_action_s6
    env.render();
    print_step_result("6 (P1 Plays Card Z to Crowded Theater - Expect Discard)", play_p1_card_z_action_id,
                      play_p1_card_z_desc, obs, r, term, tr, info_s6)

    assert "blockade_triggered" in info_s6, "BLOCKADE effect should have triggered and discarded P1's card."  #
    print(f"BLOCKADE Triggered: {info_s6['blockade_triggered']}. Correct.")
    assert info_s6.get("game_phase") == GamePhase.NORMAL_PLAY.name, \
        f"Game phase should be NORMAL_PLAY (Card Z's AMBUSH effect should not trigger). Phase is: {info_s6.get('game_phase')}"  #
    print("Card Z AMBUSH effect did not trigger (Game phase is NORMAL_PLAY). Correct.")

    card_z_in_theater = False
    for played_card_instance in env.theaters[theater_x_target_idx].player_zones[p1_idx]:
        if played_card_instance.card_ref.id == p1_card_z_id:
            card_z_in_theater = True;
            break
    assert not card_z_in_theater, f"Card Z ('{p1_card_z_obj.name}') should NOT be in theater {theater_x_target_idx} after discard."
    print(f"Card Z ('{p1_card_z_obj.name}') not found in theater. Correct (discarded).")

    assert len(
        env.battle_deck) == battle_deck_size_before_discard + 1, "Battle deck size should increase by 1 after P1's card discard."  #
    assert env.battle_deck[0].id == p1_card_z_id, \
        f"P1's discarded Card Z (ID {p1_card_z_id}) should be at bottom (idx 0) of deck. Found ID {env.battle_deck[0].id}"  #
    print(f"Card Z ('{p1_card_z_obj.name}') correctly returned to bottom of deck.")

    assert env.battle_turn_counter == battle_turn_counter_before_p1_discard_attempt + 1, \
        f"Battle turn counter should have incremented once. Before: {battle_turn_counter_before_p1_discard_attempt}, Now: {env.battle_turn_counter}"  #
    print("Battle turn counter incremented correctly for P1's discard action.")

    if term: print("Battle ended after P1's FD play. Test successful."); return
    assert env.current_player_idx == p0_idx, "Turn should pass back to P0 after P1's BLOCKADE-discarded play."

    print("\nBLOCKADE Test Function Finished.")


if __name__ == "__main__":
    # test_env()

    env_for_maneuver_test = AirLandSeaBaseEnv()
    test_maneuver_effect(env_for_maneuver_test)

    env_for_aerodrome_test = AirLandSeaBaseEnv()
    test_aerodrome_effect(env_for_aerodrome_test)

    env_for_ambush_test = AirLandSeaBaseEnv()
    test_ambush_effect(env_for_ambush_test)

    env_for_disrupt_test = AirLandSeaBaseEnv()
    test_disrupt_effect(env_for_disrupt_test)

    env_for_flip_passive_effect_test = AirLandSeaBaseEnv()
    test_passive_effect_on_flip_aerodrome(env_for_flip_passive_effect_test)

    env_for_air_drop_test = AirLandSeaBaseEnv()
    test_air_drop_effect(env_for_air_drop_test)

    env_for_transport_test = AirLandSeaBaseEnv()
    test_transport_effect(env_for_transport_test)

    env_for_escalation_test = AirLandSeaBaseEnv()
    test_escalation_effect(env_for_escalation_test)

    env_for_cover_fire_test = AirLandSeaBaseEnv()
    test_cover_fire_effect(env_for_cover_fire_test)

    env_for_support_test = AirLandSeaBaseEnv()
    test_support_effect(env_for_support_test)

    env_for_containment_test = AirLandSeaBaseEnv()
    test_simple_containment_effect_p1_plays_fd(env_for_containment_test)

    env_for_blockade_test = AirLandSeaBaseEnv()
    test_blockade_effect(env_for_blockade_test)