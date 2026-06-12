"""Unit tests for deepmem.scoring module."""

import math
import sys
import os
import unittest

# Ensure the package root is on sys.path so `deepmem` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deepmem.scoring import (
    ENTITY_BOOST_WEIGHT,
    get_bm25_params,
    normalize_bm25,
    score_and_rank,
)


class TestGetBm25Params(unittest.TestCase):
    """Tests for get_bm25_params returning valid sigmoid parameter ranges."""

    def test_bm25_params_returns_valid_range(self):
        """Midpoint should be positive and steepness should be in (0, 1]."""
        for query in [
            "short",
            "a medium length query here",
            "this is a longer query with many words to test the boundary conditions",
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen",
        ]:
            midpoint, steepness = get_bm25_params(query)
            self.assertGreater(midpoint, 0, f"midpoint must be positive for query: {query}")
            self.assertGreater(steepness, 0, f"steepness must be positive for query: {query}")
            self.assertLessEqual(steepness, 1.0, f"steepness must be <= 1.0 for query: {query}")

    def test_bm25_params_short_query(self):
        """Short queries (<=3 terms) should return the smallest midpoint."""
        midpoint, steepness = get_bm25_params("hello world")
        self.assertEqual(midpoint, 5.0)
        self.assertEqual(steepness, 0.7)

    def test_bm25_params_long_query(self):
        """Long queries (>15 terms) should return the largest midpoint."""
        query = " ".join(["word"] * 20)
        midpoint, steepness = get_bm25_params(query)
        self.assertEqual(midpoint, 12.0)
        self.assertEqual(steepness, 0.5)


class TestNormalizeBm25(unittest.TestCase):
    """Tests for normalize_bm25 sigmoid normalization."""

    def test_normalize_bm25_zero_input(self):
        """A raw score of 0 should produce a value close to 0 (well below midpoint)."""
        midpoint, steepness = 5.0, 0.7
        result = normalize_bm25(0.0, midpoint, steepness)
        # sigmoid(-steepness * midpoint) = sigmoid(-3.5) which is very small
        self.assertGreater(result, 0.0)
        self.assertLess(result, 0.05)

    def test_normalize_bm25_at_midpoint(self):
        """A raw score equal to the midpoint should produce exactly 0.5."""
        midpoint, steepness = 7.0, 0.6
        result = normalize_bm25(midpoint, midpoint, steepness)
        self.assertAlmostEqual(result, 0.5)

    def test_normalize_bm25_high_score(self):
        """A raw score well above the midpoint should produce a value close to 1.0."""
        midpoint, steepness = 5.0, 0.7
        result = normalize_bm25(50.0, midpoint, steepness)
        self.assertGreater(result, 0.99)

    def test_normalize_bm25_output_range(self):
        """Output should always be in [0, 1] for any finite input."""
        for raw in [-10.0, 0.0, 1.0, 10.0, 100.0]:
            result = normalize_bm25(raw, 5.0, 0.7)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)


class TestScoreAndRank(unittest.TestCase):
    """Tests for the additive score_and_rank function."""

    def test_score_and_rank_basic(self):
        """Basic scoring with semantic + BM25 signals."""
        semantic_results = [
            {"id": "1", "content": "memory one", "scope": "semantic", "score": 0.8},
            {"id": "2", "content": "memory two", "scope": "episodic", "score": 0.6},
            {"id": "3", "content": "memory three", "scope": "procedural", "score": 0.3},
        ]
        bm25_scores = {"1": 0.5, "2": 0.7}
        entity_boosts = {}

        results = score_and_rank(
            semantic_results, bm25_scores, entity_boosts,
            threshold=0.2, top_k=10,
        )

        # All three should pass the threshold
        self.assertEqual(len(results), 3)

        # Results should be sorted by combined score descending
        self.assertEqual(results[0]["id"], "1")  # 0.8 + 0.5 = 1.3 / 2.0 = 0.65
        self.assertEqual(results[1]["id"], "2")  # 0.6 + 0.7 = 1.3 / 2.0 = 0.65

        # Memory 3 has no BM25 score: 0.3 / 2.0 = 0.15
        self.assertEqual(results[2]["id"], "3")
        self.assertAlmostEqual(results[2]["score"], 0.15)

    def test_score_and_rank_threshold_filters(self):
        """Candidates below the semantic threshold should be excluded."""
        semantic_results = [
            {"id": "1", "content": "good", "scope": "semantic", "score": 0.9},
            {"id": "2", "content": "bad", "scope": "semantic", "score": 0.1},
        ]
        bm25_scores = {"2": 1.0}  # even high BM25 should not save it
        entity_boosts = {}

        results = score_and_rank(
            semantic_results, bm25_scores, entity_boosts,
            threshold=0.5, top_k=10,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "1")

    def test_score_and_rank_top_k_limits_results(self):
        """top_k should cap the number of returned results."""
        semantic_results = [
            {"id": str(i), "content": f"mem {i}", "scope": "semantic", "score": 0.5 + i * 0.01}
            for i in range(10)
        ]
        results = score_and_rank(
            semantic_results, {}, {},
            threshold=0.0, top_k=3,
        )
        self.assertEqual(len(results), 3)

    def test_score_and_rank_normalization(self):
        """Combined score should be capped at 1.0."""
        semantic_results = [
            {"id": "1", "content": "maxed", "scope": "semantic", "score": 1.0},
        ]
        bm25_scores = {"1": 1.0}
        entity_boosts = {"1": ENTITY_BOOST_WEIGHT}

        results = score_and_rank(
            semantic_results, bm25_scores, entity_boosts,
            threshold=0.0, top_k=10,
        )

        self.assertEqual(len(results), 1)
        # (1.0 + 1.0 + 0.5) / 2.5 = 1.0
        self.assertAlmostEqual(results[0]["score"], 1.0)


class TestEntityBoostWeight(unittest.TestCase):
    """Tests for the entity boost weight constant and its effect."""

    def test_entity_boost_weight(self):
        """ENTITY_BOOST_WEIGHT should be 0.5 as documented."""
        self.assertEqual(ENTITY_BOOST_WEIGHT, 0.5)

    def test_entity_boost_increases_score(self):
        """Adding entity boost should increase a result's combined score."""
        semantic_results = [
            {"id": "1", "content": "test", "scope": "semantic", "score": 0.6},
        ]

        without_entity = score_and_rank(
            semantic_results, {}, {},
            threshold=0.0, top_k=10,
        )
        with_entity = score_and_rank(
            semantic_results, {}, {"1": ENTITY_BOOST_WEIGHT},
            threshold=0.0, top_k=10,
        )

        self.assertGreater(with_entity[0]["score"], without_entity[0]["score"])

    def test_entity_boost_max_possible_adjustment(self):
        """When entity boosts are present, max_possible should include ENTITY_BOOST_WEIGHT."""
        semantic_results = [
            {"id": "1", "content": "test", "scope": "semantic", "score": 0.8},
        ]
        bm25_scores = {"1": 0.6}
        entity_boosts = {"1": 0.3}

        results = score_and_rank(
            semantic_results, bm25_scores, entity_boosts,
            threshold=0.0, top_k=10,
        )

        # max_possible = 1.0 (semantic) + 1.0 (bm25) + 0.5 (entity) = 2.5
        # raw = 0.8 + 0.6 + 0.3 = 1.7
        # combined = 1.7 / 2.5 = 0.68
        self.assertAlmostEqual(results[0]["score"], 0.68)


if __name__ == "__main__":
    unittest.main()
