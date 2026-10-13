"""Headless 10x20 Tetris for decision models: legal landings, their features, one sentence per landing.

The harness of FluidUse's laya Tetris demo (github.com/FluidInference/FluidUse, Sources/LayaTetris, Apache-2.0),
ported so two models can be compared on the same game: the same seven-bag randomizer (SplitMix64), landings in the
same order, the same features, wording and heuristic. The board is ten column bitmasks, bit r = row r from the floor.

A model never sees the board, only the sentence, and it reads sentences by how good or bad they sound, not by the
rules of Tetris. FluidUse tuned two wordings to laya ("plain", "graded"). d1 misreads both (it rates "makes the stack
taller" above "keeps the stack low"), so "calibrated" is a third wording, chosen clause by clause from d1's own
ratings of single clauses, and d1 is asked for a five-level rating instead of a yes or no (see judge).
"""
from __future__ import annotations

from dataclasses import dataclass

W, H = 10, 20
MASK64 = (1 << 64) - 1
QUESTION = "Is this a clean placement?"                      # the yes/no question of the original demo
RATING = ("How good is this Tetris move?", ("terrible", "bad", "okay", "good", "excellent"))
STYLES = ("plain", "graded", "calibrated")
ASKS = ("clean", "rating")
# landing height in steps of two rows; d1 rates these phrases in this order (0.79 down to 0.62 of the rating scale)
HEIGHTS = ("a perfectly safe", "a very safe", "a safe", "a fairly safe", "a slightly risky", "a risky", "a very risky",
           "a very dangerous", "a critically dangerous")
NUMBERS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")

# name -> rotation states as (column, row) offsets, row 0 at the top of the piece box
PIECES = {
    "I": [[(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 0), (0, 1), (0, 2), (0, 3)]],
    "O": [[(0, 0), (1, 0), (0, 1), (1, 1)]],
    "T": [[(0, 0), (1, 0), (2, 0), (1, 1)], [(0, 0), (0, 1), (0, 2), (1, 1)], [(1, 0), (0, 1), (1, 1), (2, 1)],
          [(1, 0), (0, 1), (1, 1), (1, 2)]],
    "S": [[(1, 0), (2, 0), (0, 1), (1, 1)], [(0, 0), (0, 1), (1, 1), (1, 2)]],
    "Z": [[(0, 0), (1, 0), (1, 1), (2, 1)], [(1, 0), (0, 1), (1, 1), (0, 2)]],
    "J": [[(0, 0), (0, 1), (1, 1), (2, 1)], [(0, 0), (1, 0), (0, 1), (0, 2)], [(0, 0), (1, 0), (2, 0), (2, 1)],
          [(1, 0), (1, 1), (0, 2), (1, 2)]],
    "L": [[(2, 0), (0, 1), (1, 1), (2, 1)], [(0, 0), (0, 1), (0, 2), (1, 2)], [(0, 0), (1, 0), (2, 0), (0, 1)],
          [(0, 0), (1, 0), (1, 1), (1, 2)]],
}
NAMES = list(PIECES)
# per rotation: (cells, box width, box height, lowest cell of each box column)
SHAPES = {n: [(c, max(x for x, _ in c) + 1, max(y for _, y in c) + 1,
               [max(y for x, y in c if x == dx) for dx in range(max(x for x, _ in c) + 1)]) for c in rots]
          for n, rots in PIECES.items()}


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)


@dataclass(slots=True)
class Landing:
    rotation: int
    column: int
    cols: list[int]                 # the board after the drop and any line clears
    cells: list[tuple[int, int]]    # (x, y from the top) the piece occupies before the clears
    lines: int
    new_holes: int
    landing_height: int
    max_height: int
    bumpiness: int
    bump_delta: int
    well: int

    @property
    def heuristic(self) -> float:
        """The Dellacherie-style linear evaluation of the baseline policy."""
        return (self.lines * 3.4 - self.new_holes * 7.9 - self.landing_height * 4.5 - self.bumpiness * 1.2
                - self.well * 3.4)


def _holes(cols: list[int]) -> int:
    return sum(c.bit_length() - c.bit_count() for c in cols)


def _bumpiness(h: list[int]) -> int:
    return sum(abs(a - b) for a, b in zip(h, h[1:]))


def _well(h: list[int]) -> int:
    return max(0, max(min(h[x - 1] if x else H, h[x + 1] if x < W - 1 else H) - h[x] for x in range(W)))


def landings(piece: str, cols: list[int]) -> list[Landing]:
    """Every hard drop of `piece` on a board, rotation by rotation, column by column."""
    heights = [c.bit_length() for c in cols]
    if max(heights) >= H - 1:       # a cell in the top two rows: the game is over
        return []
    holes0, bump0, out = _holes(cols), _bumpiness(heights), []
    for rotation, (cells, pw, ph, low) in enumerate(SHAPES[piece]):
        for column in range(W - pw + 1):
            row = min(H - 1 - heights[column + dx] - low[dx] for dx in range(pw))
            if row < 0:
                continue
            new = list(cols)
            for dx, dy in cells:
                new[column + dx] |= 1 << (H - 1 - row - dy)
            full = new[0]
            for c in new[1:]:
                full &= c
            lines = full.bit_count()
            while full:             # drop the cleared rows, highest first
                r = full.bit_length() - 1
                low_bits = (1 << r) - 1
                new = [(c & low_bits) | ((c >> (r + 1)) << r) for c in new]
                full ^= 1 << r
            h = [c.bit_length() for c in new]
            bump = _bumpiness(h)
            out.append(Landing(rotation, column, new, [(column + dx, row + dy) for dx, dy in cells], lines,
                               max(0, _holes(new) - holes0), H - row - ph // 2, max(h), bump, bump - bump0, _well(h)))
    return out


def shortlist(cands: list[Landing]) -> list[Landing]:
    """The harness constraint: withhold landings that bury a cell when a hole-free one exists."""
    clean = [c for c in cands if c.new_holes == 0]
    return clean or cands


def _plural(n: int, noun: str) -> str:
    return f"{NUMBERS[n] if n < len(NUMBERS) else n} {noun}{'' if n == 1 else 's'}"


def describe(c: Landing, piece: str, style: str = "plain") -> str:
    """One sentence per landing, the only thing the model sees. "plain" is the demo's original wording, "graded"
    its later one, which stays discriminative on a tall board; "calibrated" is the one for d1, without the piece and
    the column (they only add noise to its rating)."""
    if style == "calibrated":
        clauses = [f"clears {_plural(c.lines, 'line')}"] if c.lines else []
        clauses.append(f"leaves {_plural(c.new_holes, 'hole')} under it" if c.new_holes else "leaves no holes")
        clauses.append(f"lands at {HEIGHTS[min(len(HEIGHTS) - 1, (c.landing_height - 1) // 2)]} height")
        clauses.append("makes the surface much smoother" if c.bump_delta <= -3 else "makes the surface smoother"
                       if c.bump_delta < 0 else "keeps the surface flat" if c.bump_delta == 0 else
                       "makes the surface bumpier" if c.bump_delta <= 2 else "makes the surface much bumpier"
                       if c.bump_delta <= 4 else "leaves a very uneven surface")
        if c.well >= 3:
            clauses.append("leaves a dangerously deep well" if c.well >= 5 else "leaves a deep well")
        body = ", ".join(clauses[:-1]) + ", and " + clauses[-1] + "."
        return body[0].upper() + body[1:]
    if style == "graded":
        clauses = [f"clears {_plural(c.lines, 'line')}"] if c.lines else []
        clauses.append(f"buries {_plural(c.new_holes, 'cell')} that cannot be reached again" if c.new_holes
                       else "buries nothing")
        room = H - c.max_height
        clauses.append("fills the board to the very top" if room <= 1 else f"leaves only {room} rows of space" if room <= 3
                       else f"leaves {room} rows of space" if room <= 6 else "leaves comfortable room above" if room <= 10
                       else "keeps the stack low")
        clauses.append("makes the surface much more uneven" if c.bump_delta > 2 else "makes the surface more uneven"
                       if c.bump_delta > 0 else "evens out the surface" if c.bump_delta < 0
                       else "leaves the surface as even as before")
        if c.well >= 3:
            clauses.append(f"leaves a deep gap {c.well} rows deep")
    else:
        clauses = [f"leaves {_plural(c.new_holes, 'hole')} under it" if c.new_holes else "leaves no holes"]
        clauses.append("makes the surface much bumpier" if c.bump_delta > 2 else "makes the surface bumpier"
                       if c.bump_delta > 0 else "makes the surface flatter" if c.bump_delta < 0
                       else "keeps the surface flat")
        clauses.append("the stack is getting dangerously tall" if c.max_height >= 15 else "makes the stack taller"
                       if c.landing_height > 8 else "keeps the stack low")
        if c.well >= 3:
            clauses.append("leaves a deep well")
        if c.lines:
            clauses.append(f"clears {_plural(c.lines, 'line')}")
    body = ", ".join(clauses[:-1]) + ", and " + clauses[-1] if len(clauses) > 1 else clauses[0]
    return f"The {piece} piece dropped at column {c.column} {body}."


class Game:
    def __init__(self, seed: int):
        self.cols, self.lines, self.pieces, self.over = [0] * W, 0, 0, False
        self.rng, self.bag, self.upcoming = SplitMix64(seed), [], []

    def _draw(self) -> str:
        if not self.bag:
            self.bag = list(range(len(NAMES)))
            for i in range(len(self.bag) - 1, 0, -1):
                j = self.rng.next() % (i + 1)
                self.bag[i], self.bag[j] = self.bag[j], self.bag[i]
        return NAMES[self.bag.pop()]

    def spawn(self) -> str | None:
        """The next piece of the seven-bag, or None once the stack has topped out."""
        if self.over:
            return None
        while len(self.upcoming) < 2:
            self.upcoming.append(self._draw())
        return self.upcoming.pop(0)

    @property
    def next_piece(self) -> str | None:
        return self.upcoming[0] if self.upcoming else None

    def landings(self, piece: str) -> list[Landing]:
        return landings(piece, self.cols)

    def apply(self, c: Landing) -> None:
        self.cols, self.lines, self.pieces = c.cols, self.lines + c.lines, self.pieces + 1
        self.over = max(x.bit_length() for x in self.cols) >= H - 1

    def render(self) -> str:
        return "\n".join("".join("█" if self.cols[x] >> (H - 1 - y) & 1 else "·" for x in range(W)) for y in range(H))


def play(seed: int, choose, max_pieces: int = 0, filtered: bool = False) -> Game:
    """One game. choose(game, piece, candidates) -> the landing to play."""
    game = Game(seed)
    while (not max_pieces or game.pieces < max_pieces) and (piece := game.spawn()):
        cands = game.landings(piece)
        if filtered:
            cands = shortlist(cands)
        if not cands:
            break
        game.apply(choose(game, piece, cands))
    return game


def judge(model, kind: str, ask: str = "rating"):
    """states -> (a value per state, model calls made); the landing with the highest value is played.

    ask="clean" is the original demo's yes/no question and the value is P(yes). ask="rating" is a five-way choice,
    "How good is this Tetris move?", and the value is the expected level, 0 (terrible) to 1 (excellent).
    kind: "d1" (a runtime.Decider) or "laya" (a laya.Laya)."""
    instructions, levels = RATING
    scale = [i / (len(levels) - 1) for i in range(len(levels))]
    if kind == "laya":
        if ask == "clean":
            return lambda states: ([model.probabilities(s, "noul", QUESTION)[0][1] for s in states], len(states))
        return lambda states: ([sum(a * b for a, b in zip(model.probabilities(s, "choice", instructions, list(levels))[0],
                                                           scale)) for s in states], len(states))
    q = ({"type": "noul", "instructions": QUESTION} if ask == "clean" else
         {"type": "choice", "instructions": instructions, "criteria": dict.fromkeys(levels)})

    def d1(states):
        probs, t = model.probabilities_many(states, q)
        return [p[0] if ask == "clean" else sum(a * b for a, b in zip(p, scale)) for p in probs], t["calls"]
    return d1


def heuristic_policy(game: Game, piece: str, cands: list[Landing]) -> Landing:
    return max(cands, key=lambda c: c.heuristic)


def random_policy(seed: int):
    rng = SplitMix64(seed + 1)
    return lambda game, piece, cands: cands[rng.next() % len(cands)]
