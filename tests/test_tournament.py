"""Tests for tournament.py -- log5 math, bracket parsing, simulation, snake order."""

import numpy as np
import pytest

from tournament import (
    BRACKET_PAIRINGS,
    TournamentTeam,
    _weighted_playin_barthag,
    compute_probabilities,
    log5,
    parse_bracket_text,
    simulate_region,
    snake_order,
)


class TestLog5:
    def test_equal_teams(self):
        assert log5(0.5, 0.5) == pytest.approx(0.5)

    def test_strong_vs_weak(self):
        p = log5(0.95, 0.2)
        assert p > 0.95

    def test_weak_vs_strong(self):
        p = log5(0.2, 0.95)
        assert p < 0.05

    def test_symmetry(self):
        p_ab = log5(0.8, 0.6)
        p_ba = log5(0.6, 0.8)
        assert p_ab + p_ba == pytest.approx(1.0)

    def test_zero_barthag(self):
        assert log5(0.0, 0.5) == 0.0
        assert log5(0.5, 0.0) == 1.0

    def test_one_barthag(self):
        assert log5(1.0, 0.5) == 1.0
        assert log5(0.5, 1.0) == 0.0

    def test_both_zero(self):
        assert log5(0.0, 0.0) == 0.0

    def test_both_one(self):
        assert log5(1.0, 1.0) == 1.0


class TestSnakeOrder:
    def test_basic_8_drafters(self):
        order = snake_order(8, 16)
        assert len(order) == 16
        # Round 1: 0-7
        assert order[:8] == [0, 1, 2, 3, 4, 5, 6, 7]
        # Round 2: 7-0
        assert order[8:16] == [7, 6, 5, 4, 3, 2, 1, 0]

    def test_more_rounds(self):
        order = snake_order(4, 12)
        assert len(order) == 12
        assert order[:4] == [0, 1, 2, 3]
        assert order[4:8] == [3, 2, 1, 0]
        assert order[8:12] == [0, 1, 2, 3]

    def test_partial_round(self):
        order = snake_order(4, 6)
        assert len(order) == 6
        assert order[:4] == [0, 1, 2, 3]
        assert order[4:6] == [3, 2]

    def test_single_drafter(self):
        order = snake_order(1, 5)
        assert order == [0, 0, 0, 0, 0]


class TestParseBracketText:
    def test_basic_parse(self):
        text = """
1 Duke
16 Norfolk St.
8 Michigan St.
9 Creighton
5 Clemson
12 Drake
4 Auburn
13 Vermont
6 BYU
11 VCU
3 Wisconsin
14 Colgate
7 Dayton
10 Colorado St.
2 Alabama
15 Robert Morris
"""
        result = parse_bracket_text(text, "South")
        assert len(result) == 16
        assert result[0]["name"] == "Duke"
        assert result[0]["seed"] == 1
        assert result[1]["name"] == "Norfolk St."
        assert result[1]["seed"] == 16
        assert result[-1]["name"] == "Robert Morris"
        assert result[-1]["seed"] == 15

    def test_playin_pair(self):
        text = "11 VCU / San Diego St."
        result = parse_bracket_text(text, "East")
        # Should find the 11 seed entry
        found = [t for t in result if t["seed"] == 11]
        assert len(found) == 1
        assert found[0]["is_playin"] is True
        assert found[0]["playin_partner"] == "San Diego St."
        assert "VCU" in found[0]["name"]
        assert "San Diego St." in found[0]["name"]

    def test_missing_seeds_get_tbd(self):
        text = "1 Duke"
        result = parse_bracket_text(text, "South")
        assert len(result) == 16
        assert result[0]["name"] == "Duke"
        assert "TBD" in result[1]["name"]

    def test_empty_input(self):
        result = parse_bracket_text("", "South")
        assert len(result) == 16
        assert all("TBD" in t["name"] for t in result)


class TestSimulateRegion:
    @staticmethod
    def _make_region(barthags: list[float]) -> list[TournamentTeam]:
        """Create 16 teams from barthag list in bracket order."""
        seeds = []
        for s1, s2 in BRACKET_PAIRINGS:
            seeds.extend([s1, s2])
        return [
            TournamentTeam(
                name=f"Team_{seeds[i]}",
                seed=seeds[i],
                region="Test",
                barthag=barthags[i],
            )
            for i in range(16)
        ]

    def test_one_seed_highest_prob(self):
        # Linear barthag by seed: 1-seed = 0.97, 16-seed = 0.2
        barthag_by_seed = {s: 0.97 - (s - 1) * 0.05 for s in range(1, 17)}
        barthags = []
        for s1, s2 in BRACKET_PAIRINGS:
            barthags.append(barthag_by_seed[s1])
            barthags.append(barthag_by_seed[s2])

        teams = self._make_region(barthags)
        probs = simulate_region(teams)

        # 1-seed should have highest P(E8)
        p_e8_1 = probs["Team_1"][-1]
        for name, rounds in probs.items():
            if name != "Team_1":
                assert p_e8_1 >= rounds[-1], f"1-seed should beat {name} in P(E8)"

    def test_sixteen_seed_lowest_prob(self):
        barthag_by_seed = {s: 0.97 - (s - 1) * 0.05 for s in range(1, 17)}
        barthags = []
        for s1, s2 in BRACKET_PAIRINGS:
            barthags.append(barthag_by_seed[s1])
            barthags.append(barthag_by_seed[s2])

        teams = self._make_region(barthags)
        probs = simulate_region(teams)

        p_e8_16 = probs["Team_16"][-1]
        for name, rounds in probs.items():
            if name != "Team_16":
                assert rounds[-1] >= p_e8_16, f"16-seed should lose to {name} in P(E8)"

    def test_probabilities_sum_to_one_per_round(self):
        barthag_by_seed = {s: 0.97 - (s - 1) * 0.05 for s in range(1, 17)}
        barthags = []
        for s1, s2 in BRACKET_PAIRINGS:
            barthags.append(barthag_by_seed[s1])
            barthags.append(barthag_by_seed[s2])

        teams = self._make_region(barthags)
        probs = simulate_region(teams)

        for rnd_idx in range(4):
            total = sum(rounds[rnd_idx] for rounds in probs.values())
            # After R64: 8 survivors, R32: 4, S16: 2, E8: 1
            expected = 2 ** (3 - rnd_idx)
            assert total == pytest.approx(expected, abs=0.001), \
                f"Round {rnd_idx}: expected sum {expected}, got {total}"

    def test_equal_teams_equal_probs(self):
        teams = self._make_region([0.5] * 16)
        probs = simulate_region(teams)

        # All teams should have equal probability at each round
        for rnd_idx in range(4):
            values = [rounds[rnd_idx] for rounds in probs.values()]
            assert max(values) - min(values) < 0.001

    def test_wrong_team_count_raises(self):
        teams = [TournamentTeam(name="A", seed=1, region="X", barthag=0.5)]
        with pytest.raises(ValueError, match="16 teams"):
            simulate_region(teams)


class TestComputeProbabilities:
    @staticmethod
    def _make_bracket() -> dict[str, list[TournamentTeam]]:
        """Create a 4-region bracket with declining barthag by seed."""
        regions = ["South", "East", "West", "Midwest"]
        bracket = {}
        # Give each region slightly different barthags so results aren't identical
        region_bonus = {"South": 0.02, "East": 0.01, "West": 0.0, "Midwest": -0.01}
        for region in regions:
            teams = []
            for s1, s2 in BRACKET_PAIRINGS:
                for s in (s1, s2):
                    b = max(0.05, min(0.99, 0.95 - (s - 1) * 0.05 + region_bonus[region]))
                    teams.append(TournamentTeam(
                        name=f"{region}_{s}",
                        seed=s,
                        region=region,
                        barthag=b,
                    ))
            bracket[region] = teams
        return bracket

    def test_champ_probs_sum_near_one(self):
        bracket = self._make_bracket()
        df = compute_probabilities(bracket, region_order=["South", "East", "West", "Midwest"])
        total = df["P(Champ)"].sum()
        assert total == pytest.approx(1.0, abs=0.01)

    def test_f4_probs_sum_near_one(self):
        bracket = self._make_bracket()
        df = compute_probabilities(bracket, region_order=["South", "East", "West", "Midwest"])
        # P(F4) = P(regional champion), should sum to 4
        total = df["P(F4)"].sum()
        assert total == pytest.approx(4.0, abs=0.01)

    def test_one_seeds_dominate(self):
        bracket = self._make_bracket()
        df = compute_probabilities(bracket, region_order=["South", "East", "West", "Midwest"])
        one_seeds = df[df["seed"] == 1]
        others = df[df["seed"] != 1]
        # Each 1-seed should have higher P(Champ) than any non-1 seed in their region
        for _, row in one_seeds.iterrows():
            region_others = others[others["region"] == row["region"]]
            for _, other in region_others.iterrows():
                assert row["P(Champ)"] > other["P(Champ)"]

    def test_sorted_by_champ(self):
        bracket = self._make_bracket()
        df = compute_probabilities(bracket, region_order=["South", "East", "West", "Midwest"])
        champs = df["P(Champ)"].values
        assert all(champs[i] >= champs[i + 1] for i in range(len(champs) - 1))


class TestPlayinWeightedBarthag:
    def test_equal_teams(self):
        result = _weighted_playin_barthag(0.5, 0.5)
        assert result == pytest.approx(0.5)

    def test_strong_team_weighted_more(self):
        result = _weighted_playin_barthag(0.9, 0.3)
        # Should be closer to 0.9 since the strong team is more likely to win
        assert result > 0.6
        assert result < 0.9

    def test_manual_calculation(self):
        a, b = 0.8, 0.4
        p_a_wins = log5(a, b)
        expected = p_a_wins * a + (1 - p_a_wins) * b
        result = _weighted_playin_barthag(a, b)
        assert result == pytest.approx(expected)
