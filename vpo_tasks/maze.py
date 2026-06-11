"""maze task: grid navigation collecting gold/diamonds and avoiding lava.

The reward is a 4-dim vector ``[completion, gold, diamond, avoid_lava]``, each
in [0, 1]. The vector is all zeros unless the trajectory reaches the exit E;
the trajectory ends the moment the agent steps on E. ``gold``/``diamond`` are
the fraction of those tiles collected before E; ``avoid_lava`` is one minus the
fraction of lava tiles stepped on.

Contains the move parser, trajectory simulator and scorer, the multi-route
parser, the prompt templates, the maze generator (``preprocess``), and the
held-out eval (``evaluate``).
"""

from __future__ import annotations

import re
from collections import deque

from vpo.task import SubRewrite, Task, pad_matrix, register
from vpo.utils.eval_metrics import (
    best_of_k_curve,
    diversity,
    feature_spread,
    pareto_count,
)

# ---------------------------------------------------------------------------
# Grid constants + simulation
# ---------------------------------------------------------------------------

EMPTY = "."
START = "S"
END = "E"
GOLD = "G"
DIAMOND = "D"
LAVA = "L"
WALL = "#"
BONUS = "B"  # stepping on a bonus cell grants a score multiplier (convex mode)

WALKABLE = {EMPTY, START, END, GOLD, DIAMOND, LAVA, BONUS}
ITEMS = {GOLD, DIAMOND, LAVA, BONUS}

DIRECTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

SUB_OBJECTIVE_NAMES = ("completion", "gold", "diamond", "avoid_lava")
ZERO_SCORES = [0.0] * len(SUB_OBJECTIVE_NAMES)
SCORE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)


def parse_grid(grid_text: str) -> list[list[str]]:
    """Parse whitespace-separated grid string into list[list[str]]."""
    return [row.split() for row in grid_text.strip().split("\n")]


def extract_moves(response: str) -> list[str]:
    """Pull move sequence from <answer>...</answer>. Falls back to whole text."""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    text = match.group(1) if match else response
    return re.findall(r"\b(UP|DOWN|LEFT|RIGHT)\b", text.upper())


def parse_multi_routes(response: str, m: int) -> list[str]:
    """Pull m route strings from numbered <route_i>...</route_i> tags."""
    routes = []
    for i in range(1, m + 1):
        match = re.search(
            rf"<route_{i}>(.*?)</route_{i}>", response, re.DOTALL | re.IGNORECASE
        )
        if match:
            routes.append(match.group(1))
    # A single matched tag is still an explicit route — falling through to the
    # whole-response fallback would pick up direction words from the prose
    # around the tag and simulate a different trajectory.
    if routes:
        return (routes + [""] * m)[:m]

    parts = [p.strip() for p in response.split("---") if p.strip()]
    if len(parts) >= 2:
        return (parts + [""] * m)[:m]

    answers = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if len(answers) >= 2:
        return (answers + [""] * m)[:m]

    return [response] + [""] * (m - 1)


def dedup_routes(routes: list[str]) -> list[str]:
    """Drop empty and duplicate move sequences (by normalized move-list)."""
    seen = set()
    out = []
    for r in routes:
        moves = tuple(extract_moves(r))
        if not moves:
            continue
        if moves in seen:
            continue
        seen.add(moves)
        out.append(r)
    return out


def simulate(grid, start, end, moves, max_steps) -> dict:
    """Run trajectory. Walls/borders block movement; lava is walkable but
    counted. Trajectory ENDS the moment the agent steps on E."""
    r, c = start
    H = len(grid)
    W = len(grid[0]) if H else 0
    steps = 0
    visited_items = set()
    reached_end = False

    for move in moves[:max_steps]:
        dr, dc = DIRECTIONS.get(move, (0, 0))
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != WALL:
            r, c = nr, nc
        steps += 1

        cell = grid[r][c]
        # NB: check END *before* counting current cell as a collected item.
        if cell == END:
            reached_end = True
            break
        if cell in ITEMS and (r, c) not in visited_items:
            visited_items.add((r, c))

    # Degenerate: agent already at E at step 0 (e.g., empty move list)
    if not reached_end and (r, c) == end:
        reached_end = True

    gold_got = sum(1 for (rr, cc) in visited_items if grid[rr][cc] == GOLD)
    diamond_got = sum(1 for (rr, cc) in visited_items if grid[rr][cc] == DIAMOND)
    lava_hit = sum(1 for (rr, cc) in visited_items if grid[rr][cc] == LAVA)
    bonus_hit = sum(1 for (rr, cc) in visited_items if grid[rr][cc] == BONUS)

    return {
        "steps": steps,
        "gold": gold_got,
        "diamond": diamond_got,
        "lava": lava_hit,
        "bonus": bonus_hit,
        "reached_end": reached_end,
        "final_r": r,
        "final_c": c,
    }


def _clamp(x):
    return max(0.0, min(1.0, x))


def score_trajectory(sim: dict, num_gold: int, num_diamond: int,
                     num_lava: int, score_mode: str = "linear") -> list[float]:
    """Return 4-dim reward vector. Zero vector if not reached_end.

    score_mode: "linear" (collected/total per dim) or "convex_bonus"
    ((collected/total)**2 per dim, ×1.5 if any BONUS cell visited; clamped).
    """
    if not sim["reached_end"]:
        return list(ZERO_SCORES)

    gold_frac = (sim["gold"] / num_gold) if num_gold > 0 else 0.0
    diam_frac = (sim["diamond"] / num_diamond) if num_diamond > 0 else 0.0
    safe_frac = (1.0 - sim["lava"] / num_lava) if num_lava > 0 else 1.0

    if score_mode == "convex_bonus":
        mult = 1.5 if sim.get("bonus", 0) > 0 else 1.0
        gold = mult * (gold_frac ** 2)
        diamond = mult * (diam_frac ** 2)
        avoid_lava = mult * (safe_frac ** 2)
        return [1.0, _clamp(gold), _clamp(diamond), _clamp(avoid_lava)]

    return [1.0, _clamp(gold_frac), _clamp(diam_frac), _clamp(safe_frac)]


def score_route(response: str, gt: dict) -> list[float]:
    """Score a single solution string → 4-dim reward vector."""
    grid = parse_grid(gt["grid_text"])
    start = tuple(gt["start"])
    end = tuple(gt["end"])
    moves = extract_moves(response)
    if not moves:
        return list(ZERO_SCORES)
    sim = simulate(grid, start, end, moves, gt["max_steps"])
    return score_trajectory(
        sim, gt["num_gold"], gt["num_diamond"], gt["num_lava"],
        score_mode=gt.get("score_mode", "linear"),
    )


def bfs_distance(grid, start, end, lava_blocked=True) -> float:
    """Shortest path S→E. Returns inf if unreachable."""
    H, W = len(grid), len(grid[0])
    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == end:
            return dist[(r, c)]
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if grid[nr][nc] == WALL:
                continue
            if lava_blocked and grid[nr][nc] == LAVA:
                continue
            if (nr, nc) in dist:
                continue
            dist[(nr, nc)] = dist[(r, c)] + 1
            q.append((nr, nc))
    return float("inf")


# ---------------------------------------------------------------------------
# Prompt templates (preprocess + multi-solution rewrite)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
Navigate a {size}x{size} maze from S to E. Collect gold and diamonds, avoid lava.

Grid:
{grid_text}

- Move: UP, DOWN, LEFT, RIGHT. # is a wall — you cannot enter it.
- Do not leave the grid.
- Collect: G (Gold), D (Diamond), B (Bonus) tiles by stepping on them.
- Avoid: L (Lava) tiles. Stepping on lava costs you.
- Visiting a B cell multiplies your other scores — explore!
- You MUST reach E. If you don't reach E, your score is zero everywhere.
- Items only count if collected BEFORE you reach E (the trajectory ends at E).
- You have {max_steps} steps.

This maze has {num_gold} Gold, {num_diamond} Diamond, {num_lava} Lava, and {num_bonus} Bonus tiles.
Output moves in <answer>...</answer> tags, e.g., <answer>UP UP RIGHT</answer>."""

_ROUTE_TAGS = ", ".join(f"<route_{i}>...</route_{i}>" for i in range(1, 4))

_MULTI_TEMPLATE = (
    "Reason briefly about the maze, then provide {m} genuinely different "
    "routes from S to E. Each route is a sequence of UP/DOWN/LEFT/RIGHT "
    f"moves (space-separated). Wrap each route in numbered tags ({_ROUTE_TAGS}). "
    "Inside each tag put ONLY moves (no arrows, no coordinates, no prose); "
    "any reasoning goes outside the tags. Each route must independently reach "
    "E within the per-route step budget stated above (score is zero if it "
    "doesn't).\n"
    "Format example (m=3):\n"
    "  <route_1>RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN</route_1>\n"
    "  <route_2>DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT</route_2>\n"
    "  <route_3>RIGHT DOWN RIGHT DOWN RIGHT DOWN RIGHT DOWN</route_3>"
)

_SINGLE_OUTPUT_PATTERN = re.compile(
    r"Output moves in <answer>\.\.\.</answer> tags, "
    r"e\.g\., <answer>UP UP RIGHT</answer>\.",
)
# Shared follow-up fix applied on the substitution path (single→multi).
_STEPS_PER_ROUTE = (re.compile(r"You have (\d+) steps\."), r"You have \1 steps per route.")


# ---------------------------------------------------------------------------
# The Task
# ---------------------------------------------------------------------------


class MazeTask(Task):
    name = "maze"
    data_source = "maze"
    objectives = [
        ("completion", "Reaching the exit (E)"),
        ("gold", "Collecting Gold (G) tiles"),
        ("diamond", "Collecting Diamond (D) tiles"),
        ("avoid_lava", "Avoiding Lava (L) tiles"),
    ]
    multi_rewrite = SubRewrite(
        pattern=_SINGLE_OUTPUT_PATTERN,
        template=_MULTI_TEMPLATE,
        extra_subs=[_STEPS_PER_ROUTE],
    )

    def weighted_scalar(self, vector):
        return sum(w * s for w, s in zip(SCORE_WEIGHTS, vector)) / sum(SCORE_WEIGHTS)

    # ----- scoring -----
    def score_one(self, solution_str, ground_truth, extra_info):
        moves = extract_moves(solution_str)
        sub = score_route(solution_str, ground_truth)
        return {
            "sub_scores": sub,
            "channels": self.named_channels(sub),
            "scalar": self.weighted_scalar(sub),
            "extra": {},
            "num_test_executions": 0,
            "num_programs_executed": 1 if moves else 0,
        }

    def score_multi(self, solution_str, m, ground_truth, extra_info):
        parsed = parse_multi_routes(solution_str, m)
        num_parsed = sum(1 for s in parsed if s and s.strip())
        unique = dedup_routes(parsed)
        unique_scores = [score_route(r, ground_truth) for r in unique]
        padded = pad_matrix(unique_scores, m, ZERO_SCORES)

        # Pool diversity metrics over the dedup'd routes (logged, not reward) —
        # the shared, tested definitions from vpo/utils/eval_metrics.
        extra = self.multi_channel_extras(padded)
        extra["route_diversity"] = diversity(unique_scores)
        extra["pareto_count"] = float(pareto_count(unique_scores))
        extra["feature_spread"] = feature_spread(unique_scores)

        return {
            "sub_scores": padded,
            "scalar": sum(self.weighted_scalar(row) for row in padded),
            "num_parsed": num_parsed,
            "num_test_executions": 0,
            "extra": extra,
        }

    # ----- preprocess -----
    def add_preprocess_args(self, p):
        p.add_argument("--local_save_dir", required=True)
        p.add_argument("--train_size", type=int, default=1000)
        p.add_argument("--test_size", type=int, default=100)
        p.add_argument("--train_seed", type=int, default=42)
        p.add_argument("--test_seed", type=int, default=4242)

    def preprocess(self, args):
        import os
        from collections import OrderedDict

        import pyarrow as pa
        import pyarrow.parquet as pq

        os.makedirs(args.local_save_dir, exist_ok=True)
        print(f"Generating maze (convex_bonus + 4-corner, {V8_GRID_SIZE}x{V8_GRID_SIZE}) ...")
        train_rows = _make_split(args.train_size, args.train_seed)
        test_rows = _make_split(args.test_size, args.test_seed)

        def to_table(rows):
            cols = OrderedDict()
            for k in ["data_source", "prompt", "ability", "reward_model", "extra_info"]:
                cols[k] = [r[k] for r in rows]
            return pa.Table.from_pydict(cols)

        pq.write_table(to_table(train_rows), os.path.join(args.local_save_dir, "train.parquet"))
        pq.write_table(to_table(test_rows), os.path.join(args.local_save_dir, "test.parquet"))
        print(f"Wrote train={len(train_rows)} test={len(test_rows)}")
        print("\n=== Sample maze ===")
        print(train_rows[0]["prompt"][0]["content"])

    # ----- eval -----
    def add_eval_args(self, p):
        from vpo.eval_harness import add_sampling_eval_args

        p.add_argument("--method", required=True,
                       choices=["grpo", "gdpo", "maxrl", "multi_rlvr", "vpo", "goal_cond"])
        p.add_argument("--data-dir", required=True, help="dir containing test.parquet")
        p.add_argument("--output", required=True,
                       help="JSON path for summary; .npy tensor written alongside")
        add_sampling_eval_args(p, n_chains=10, num_solutions=None,
                               max_tokens=1024, temperature=0.7, seed=0)

    def evaluate(self, args):
        import json
        from pathlib import Path

        import numpy as np

        from vpo.eval_harness import EvalHarness

        is_multi = args.method in {"vpo", "multi_rlvr"}
        m = args.num_solutions if args.num_solutions is not None else (3 if is_multi else 1)

        harness = EvalHarness(args.model, max_model_len=2048, seed=args.seed)
        t = harness.load_test(args.data_dir)
        n_mazes = len(t["prompt"])

        prompts = []
        for i in range(n_mazes):
            base = t["prompt"][i][0]["content"]
            if is_multi:
                base = self.rewrite_multi_solution(base, m)
            prompts.append(harness.render(base, enable_thinking=False))

        out = harness.generate(prompts, n=args.n_chains, temperature=args.temperature,
                               max_tokens=args.max_tokens, seed=args.seed)

        tensor = np.zeros((n_mazes, args.n_chains, m, 4), dtype=np.float32)
        parse_fail = 0
        for i in range(n_mazes):
            gt = t["reward_model"][i]["ground_truth"]
            for j in range(args.n_chains):
                text = out[i].outputs[j].text
                if is_multi:
                    routes = parse_multi_routes(text, m)
                    # parse_multi_routes always pads to exactly m entries;
                    # a missing route is an empty string, not a short list.
                    parse_fail += sum(1 for r in routes if not r.strip())
                    for k in range(m):
                        tensor[i, j, k] = score_route(routes[k] if k < len(routes) else "", gt)
                else:
                    tensor[i, j, 0] = score_route(text, gt)

        npy_path = Path(args.output).with_suffix(".npy")
        np.save(npy_path, tensor)

        flat = tensor.reshape(n_mazes, -1, 4)
        scalar = flat.mean(axis=-1)
        pool_size = scalar.shape[1]
        # best-of-k over a random pool subset (shared estimator), averaged over
        # mazes — not a generation-order prefix.
        rng = np.random.default_rng(args.seed)
        best_curve = np.mean(
            [best_of_k_curve(scalar[i], rng.permutation(pool_size)) for i in range(n_mazes)],
            axis=0,
        )
        best_at = {f"best@{k}": float(best_curve[k - 1])
                   for k in [1, 3, 5, 10, pool_size] if k <= pool_size}

        summary = {
            "model": args.model,
            "method": args.method,
            "num_mazes": n_mazes,
            "n_chains": args.n_chains,
            "num_solutions": m,
            "completion_rate": float((flat[..., 0] > 0).mean()),
            "per_objective_mean": {
                name: float(flat[..., k].mean())
                for k, name in enumerate(["completion", "gold", "diamond", "avoid_lava"])
            },
            "best_at_k": best_at,
            "parse_fail_rate": parse_fail / max(1, n_mazes * args.n_chains * m),
            "tensor_path": str(npy_path),
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Dataset generator (designed-conflict 9x9 maze)
# ---------------------------------------------------------------------------

V8_GRID_SIZE = 9
V8_BUDGET_SLACK = 7  # allows one corner + bonus center detour


def _prims_maze(rng, size):
    grid = [[WALL] * size for _ in range(size)]
    sr, sc = rng.randrange(size), rng.randrange(size)
    grid[sr][sc] = EMPTY
    frontier = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < size and 0 <= nc < size:
            frontier.append((nr, nc))
    while frontier:
        idx = rng.randrange(len(frontier))
        r, c = frontier.pop(idx)
        if grid[r][c] != WALL:
            continue
        en = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                 if 0 <= r + dr < size and 0 <= c + dc < size and grid[r + dr][c + dc] == EMPTY)
        if en != 1:
            continue
        grid[r][c] = EMPTY
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == WALL:
                frontier.append((nr, nc))
    return grid


def _add_cycles(grid, rng, n_cycles):
    size = len(grid)
    cands = []
    for r in range(size):
        for c in range(size):
            if grid[r][c] != WALL:
                continue
            en = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                     if 0 <= r + dr < size and 0 <= c + dc < size and grid[r + dr][c + dc] == EMPTY)
            if en >= 2:
                cands.append((r, c))
    rng.shuffle(cands)
    for r, c in cands[:n_cycles]:
        grid[r][c] = EMPTY


def _bfs_path(grid, start, end, avoid=None):
    size = len(grid)
    avoid = avoid or set()
    parent = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == end:
            path = []
            cur = (r, c)
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return list(reversed(path))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < size and 0 <= nc < size):
                continue
            if grid[nr][nc] == WALL:
                continue
            if (nr, nc) in avoid:
                continue
            if (nr, nc) in parent:
                continue
            parent[(nr, nc)] = (r, c)
            q.append((nr, nc))
    return None


def _zone(corner, size, radius=2):
    r0, c0 = corner
    return [(r, c) for r in range(size) for c in range(size)
            if abs(r - r0) + abs(c - c0) <= radius]


def _build_maze(seed):
    import random

    rng = random.Random(seed)
    size = V8_GRID_SIZE
    grid = _prims_maze(rng, size)
    _add_cycles(grid, rng, n_cycles=rng.randint(18, 28))

    se_axes = [(0, 0, size - 1, size - 1), (0, size - 1, size - 1, 0)]
    sr, sc, er, ec = rng.choice(se_axes)
    start = (sr, sc)
    end = (er, ec)
    perp = [c for c in [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
            if c not in [start, end]]
    rng.shuffle(perp)
    gold_corner, diam_corner = perp
    center = (size // 2, size // 2)

    for c in [start, end, gold_corner, diam_corner, center]:
        grid[c[0]][c[1]] = EMPTY

    if _bfs_path(grid, start, end) is None:
        return None

    via_gold = (len(_bfs_path(grid, start, gold_corner) or [99] * 99) - 1) + (len(_bfs_path(grid, gold_corner, end) or [99] * 99) - 1)
    via_diam = (len(_bfs_path(grid, start, diam_corner) or [99] * 99) - 1) + (len(_bfs_path(grid, diam_corner, end) or [99] * 99) - 1)
    via_center = (len(_bfs_path(grid, start, center) or [99] * 99) - 1) + (len(_bfs_path(grid, center, end) or [99] * 99) - 1)
    if via_gold > 90 or via_diam > 90 or via_center > 90:
        return None

    budget = max(via_gold, via_diam) + V8_BUDGET_SLACK
    via_gold_then_diam = (len(_bfs_path(grid, start, gold_corner) or [99] * 99) - 1) \
        + (len(_bfs_path(grid, gold_corner, diam_corner) or [99] * 99) - 1) \
        + (len(_bfs_path(grid, diam_corner, end) or [99] * 99) - 1)
    if via_gold_then_diam <= budget:
        return None

    grid[start[0]][start[1]] = START
    grid[end[0]][end[1]] = END
    grid[center[0]][center[1]] = BONUS

    gold_zone = [c for c in _zone(gold_corner, size, radius=2) if grid[c[0]][c[1]] == EMPTY]
    diam_zone = [c for c in _zone(diam_corner, size, radius=2) if grid[c[0]][c[1]] == EMPTY]
    if len(gold_zone) < 3 or len(diam_zone) < 3:
        return None
    n_gold = rng.randint(3, min(5, len(gold_zone)))
    n_diam = rng.randint(3, min(5, len(diam_zone)))
    for r, c in rng.sample(gold_zone, n_gold):
        grid[r][c] = GOLD
    for r, c in rng.sample(diam_zone, n_diam):
        grid[r][c] = DIAMOND

    interior = [(r, c) for r in range(2, size - 2) for c in range(2, size - 2)
                if grid[r][c] == EMPTY]
    n_lava = min(rng.randint(3, 5), len(interior))
    for r, c in rng.sample(interior, n_lava):
        grid[r][c] = LAVA

    return {
        "grid": grid, "start": start, "end": end,
        "num_gold": n_gold, "num_diamond": n_diam, "num_lava": n_lava,
        "num_bonus": 1, "size": size, "max_steps": budget,
        "gold_corner": gold_corner, "diam_corner": diam_corner,
        "bonus_cell": center,
        "via_gold": via_gold, "via_diam": via_diam, "via_center": via_center,
        "both_corners_detour": via_gold_then_diam, "seed": seed,
    }


def _build_record(seed):
    maze = _build_maze(seed)
    if maze is None:
        return None
    d_safe = bfs_distance(maze["grid"], maze["start"], maze["end"], lava_blocked=True)
    d_safe = int(d_safe) if d_safe != float("inf") else -1
    if not (d_safe != -1 and d_safe <= maze["max_steps"]):
        return None
    grid_text = "\n".join(" ".join(row) for row in maze["grid"])
    prompt = PROMPT_TEMPLATE.format(
        size=maze["size"], grid_text=grid_text, max_steps=maze["max_steps"],
        num_gold=maze["num_gold"], num_diamond=maze["num_diamond"],
        num_lava=maze["num_lava"], num_bonus=maze["num_bonus"],
    )
    return {
        "data_source": "maze",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "spatial",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "grid_text": grid_text,
                "start": list(maze["start"]), "end": list(maze["end"]),
                "max_steps": maze["max_steps"],
                "num_gold": maze["num_gold"], "num_diamond": maze["num_diamond"],
                "num_lava": maze["num_lava"],
                "score_mode": "convex_bonus",
            },
        },
        "extra_info": {
            "seed": maze["seed"], "d_safe": d_safe,
            "gold_corner": list(maze["gold_corner"]),
            "diam_corner": list(maze["diam_corner"]),
            "bonus_cell": list(maze["bonus_cell"]),
            "via_gold": maze["via_gold"], "via_diam": maze["via_diam"],
            "via_center": maze["via_center"],
            "both_corners_detour": maze["both_corners_detour"],
        },
    }


def _make_split(n, base_seed):
    rows = []
    offset = 0
    attempts = 0
    while len(rows) < n:
        rec = _build_record(base_seed + offset)
        offset += 1
        attempts += 1
        if rec is not None:
            rows.append(rec)
        if attempts > n * 300:
            print(f"WARN: only {len(rows)}/{n} after {attempts} attempts")
            break
    return rows


TASK = register(MazeTask())
