"""The Tetris harness: it has to stay the game FluidUse's demo plays, or the comparison with laya means nothing."""
import statistics
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d1ane.tetris import Game, describe, heuristic_policy, judge, landings, play, shortlist  # noqa: E402


class Harness(unittest.TestCase):
    def test_heuristic_reproduces_fluiduse_controls(self):
        """Benchmarks.md of FluidUse: 582.8 pieces / 217.9 lines unfiltered, 2,874.4 / 1,140.2 filtered with a
        5,000-piece cap, seeds 1-10."""
        games = [play(seed, heuristic_policy) for seed in range(1, 11)]
        self.assertEqual(round(statistics.mean(g.pieces for g in games), 1), 582.8)
        self.assertEqual(round(statistics.mean(g.lines for g in games), 1), 217.9)
        games = [play(seed, heuristic_policy, 5000, True) for seed in range(1, 11)]
        self.assertEqual(round(statistics.mean(g.pieces for g in games), 1), 2874.4)
        self.assertEqual(sum(g.pieces == 5000 for g in games), 3)

    def test_first_piece(self):
        game = Game(7)
        piece = game.spawn()
        cands = game.landings(piece)
        self.assertEqual((piece, len(cands), len(shortlist(cands))), ("T", 34, 8))
        self.assertEqual(describe(cands[0], piece), "The T piece dropped at column 0 leaves two holes under it, "
                         "makes the surface bumpier, and keeps the stack low.")
        self.assertEqual(describe(cands[0], piece, "graded"), "The T piece dropped at column 0 buries two cells that "
                         "cannot be reached again, keeps the stack low, and makes the surface more uneven.")
        self.assertEqual(describe(cands[0], piece, "calibrated"),
                         "Leaves two holes under it, lands at a perfectly safe height, and makes the surface bumpier.")

    def test_judges_read_the_right_option(self):
        class Fake:                                  # stands in for a Decider: probabilities in its answer format
            def probabilities_many(self, states, q):
                noul = q["type"] == "noul"
                return [[0.9, 0.1] if noul else [0, 0, 0, 0.5, 0.5] for _ in states], {"calls": len(states)}
        self.assertEqual(judge(Fake(), "d1", "clean")(["a", "b"]), ([0.9, 0.9], 2))      # a noul is [yes, no]
        self.assertEqual(judge(Fake(), "d1", "rating")(["a"]), ([0.875], 1))             # between good and excellent

    def test_line_clear_and_top_out(self):
        full_but_one = [1] * 9 + [0]                       # the floor row is missing its last cell
        best = max(landings("I", full_but_one), key=lambda c: c.lines)
        self.assertEqual((best.lines, best.column, best.rotation), (1, 9, 1))
        self.assertEqual(best.cols, [0] * 9 + [0b111])     # the row is gone, three cells of the I remain
        self.assertEqual(landings("O", [(1 << 19) - 1] + [0] * 9), [])    # a column reaches the top two rows


if __name__ == "__main__":
    unittest.main()
