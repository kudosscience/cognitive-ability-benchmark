from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterable

MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 5
DIFFICULTY_LEVELS = (1, 2, 3, 4, 5)
DEFAULT_DATASET_SEED = 20260415

ALPHABET = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ALPHABET_SIZE = len(ALPHABET)
PATH_JOINER = " > "
DEFAULT_FORWARD_DELTA = 1
BASE_HABIT_STEPS = 4

OP_ADD = "ADD"
OP_MUL = "MUL"
OP_XOR = "XOR"
OP_TYPES = (OP_ADD, OP_MUL, OP_XOR)
BASE_RULE_OPERATIONS = 5
ADD_BASE_MAX = 3
ADD_SCALE = 2
MUL_BASE_MIN = 2
MUL_BASE_MAX = 3
XOR_BASE_MAX = 8
XOR_SCALE = 4
INITIAL_STATE_MIN = 10
INITIAL_STATE_BASE_MAX = 20
INITIAL_STATE_SCALE = 5

GOAL_ACTION_ID = "GOAL"
GOAL_RESOURCE = "MISSION_COMPLETE"
BASE_CHAIN_LENGTH = 3
BASE_TRAP_COUNT = 2
ACTION_ID_PREFIX_CHAIN = "C"
ACTION_ID_PREFIX_TRAP = "T"

RESOURCE_POOL = (
    "SEED",
    "ORE",
    "GEAR",
    "TOKEN",
    "MAP",
    "SEAL",
    "CIPHER",
    "CRYSTAL",
    "KEY",
    "SIGIL",
    "RUNE",
    "CORE",
    "PASS",
    "GLYPH",
    "SPARK",
    "ANCHOR",
)

JUNK_RESOURCE_POOL = (
    "DUST",
    "ASH",
    "SCRAP",
    "NOISE",
    "STATIC",
    "SILT",
    "CHAFF",
    "RUST",
)


@dataclass(frozen=True)
class HabitOverrideSample:
    prompt: str
    expected_output: str
    difficulty: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RuleShiftSample:
    prompt: str
    expected_output: str
    difficulty: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ConflictPlanningSample:
    prompt: str
    expected_output: str
    difficulty: int
    metadata: dict[str, Any]


def _validate_difficulty(difficulty: int) -> None:
    if difficulty < MIN_DIFFICULTY or difficulty > MAX_DIFFICULTY:
        raise ValueError(
            f"Difficulty must be between {MIN_DIFFICULTY} and {MAX_DIFFICULTY}."
        )


def _difficulty_for_index(index: int) -> int:
    return DIFFICULTY_LEVELS[index % len(DIFFICULTY_LEVELS)]


def _letter_at(index: int) -> str:
    return ALPHABET[index % ALPHABET_SIZE]


def _simulate_habit_path(
    start_index: int,
    steps: int,
    override_letters: set[str],
    override_delta: int,
) -> list[str]:
    current_index = start_index
    visited_letters = [_letter_at(current_index)]

    for _ in range(steps):
        current_letter = _letter_at(current_index)
        delta = override_delta if current_letter in override_letters else DEFAULT_FORWARD_DELTA
        current_index = (current_index + delta) % ALPHABET_SIZE
        visited_letters.append(_letter_at(current_index))

    return visited_letters


def generate_habit_override_sample(
    difficulty: int,
    rng: random.Random,
) -> HabitOverrideSample:
    _validate_difficulty(difficulty)

    steps = BASE_HABIT_STEPS + difficulty
    start_index = rng.randrange(ALPHABET_SIZE)
    override_count = min(ALPHABET_SIZE - 1, difficulty + 1)
    override_letters = tuple(sorted(rng.sample(ALPHABET, k=override_count)))
    override_delta = -(1 + (difficulty // 2))

    visited_letters = _simulate_habit_path(
        start_index=start_index,
        steps=steps,
        override_letters=set(override_letters),
        override_delta=override_delta,
    )

    prompt_lines = [
        "Executive Functions / Inhibitory Control Task",
        "Navigate an alphabet ring while overriding habitual forward motion.",
        f"Start letter: {_letter_at(start_index)}",
        f"Total moves: {steps}",
        f"Default move: +{DEFAULT_FORWARD_DELTA} (next alphabet letter)",
        f"Override letters: {', '.join(override_letters)}",
        f"Override move when on an override letter: {override_delta}",
        "Output format: return the full visited path including the start letter,",
        f"joined exactly by '{PATH_JOINER}'.",
    ]

    expected_output = PATH_JOINER.join(visited_letters)
    metadata = {
        "start_letter": _letter_at(start_index),
        "steps": steps,
        "override_letters": list(override_letters),
        "override_delta": override_delta,
        "visited_letters": visited_letters,
    }

    return HabitOverrideSample(
        prompt="\n".join(prompt_lines),
        expected_output=expected_output,
        difficulty=difficulty,
        metadata=metadata,
    )


def _draw_operation_value(operation: str, difficulty: int, rng: random.Random) -> int:
    if operation == OP_ADD:
        add_max = ADD_BASE_MAX + (difficulty * ADD_SCALE)
        return rng.randint(1, add_max)

    if operation == OP_MUL:
        mul_max = MUL_BASE_MAX + difficulty
        return rng.randint(MUL_BASE_MIN, mul_max)

    if operation == OP_XOR:
        xor_max = XOR_BASE_MAX + (difficulty * XOR_SCALE)
        return rng.randint(1, xor_max)

    raise ValueError(f"Unsupported operation: {operation}")


def _apply_rule_shift_operation(
    state: int,
    operation: str,
    value: int,
    shifted: bool,
) -> int:
    if operation == OP_ADD:
        return state - value if shifted else state + value

    if operation == OP_MUL:
        return state // value if shifted else state * value

    if operation == OP_XOR:
        return state | value if shifted else state ^ value

    raise ValueError(f"Unsupported operation: {operation}")


def generate_rule_shift_sample(
    difficulty: int,
    rng: random.Random,
) -> RuleShiftSample:
    _validate_difficulty(difficulty)

    operation_count = BASE_RULE_OPERATIONS + difficulty
    shift_after_index = rng.randint(2, operation_count - 2)
    initial_state_max = INITIAL_STATE_BASE_MAX + (difficulty * INITIAL_STATE_SCALE)
    initial_state = rng.randint(INITIAL_STATE_MIN, initial_state_max)

    operations: list[tuple[str, int]] = []
    for _ in range(operation_count):
        operation = rng.choice(OP_TYPES)
        value = _draw_operation_value(operation=operation, difficulty=difficulty, rng=rng)
        operations.append((operation, value))

    state = initial_state
    trace = [state]
    for index, (operation, value) in enumerate(operations, start=1):
        shifted = index > shift_after_index
        state = _apply_rule_shift_operation(
            state=state,
            operation=operation,
            value=value,
            shifted=shifted,
        )
        trace.append(state)

    operation_lines = [
        f"{index}. {operation} {value}"
        for index, (operation, value) in enumerate(operations, start=1)
    ]

    prompt_lines = [
        "Executive Functions / Cognitive Flexibility Task",
        "Compute the final state after a dynamic rule shift.",
        f"Initial state: {initial_state}",
        f"Rule shift: after step {shift_after_index}, operation semantics change.",
        "Semantics before shift:",
        f"- {OP_ADD} x => state + x",
        f"- {OP_MUL} x => state * x",
        f"- {OP_XOR} x => state XOR x",
        "Semantics after shift:",
        f"- {OP_ADD} x => state - x",
        f"- {OP_MUL} x => state // x",
        f"- {OP_XOR} x => state OR x",
        "Operations:",
        *operation_lines,
        "Output format: return only the final integer.",
    ]

    metadata = {
        "initial_state": initial_state,
        "shift_after_index": shift_after_index,
        "operations": [
            {"index": index, "operation": operation, "value": value}
            for index, (operation, value) in enumerate(operations, start=1)
        ],
        "trace": trace,
    }

    return RuleShiftSample(
        prompt="\n".join(prompt_lines),
        expected_output=str(state),
        difficulty=difficulty,
        metadata=metadata,
    )


def _render_action_line(action: dict[str, str]) -> str:
    action_id = action["action_id"]
    consumes = action["consumes"]
    produces = action["produces"]
    return f"- {action_id}: consume {consumes} -> produce {produces}"


def parse_action_sequence(sequence: str) -> list[str]:
    return [segment.strip().upper() for segment in sequence.split(",") if segment.strip()]


def generate_conflict_planning_sample(
    difficulty: int,
    rng: random.Random,
) -> ConflictPlanningSample:
    _validate_difficulty(difficulty)

    chain_length = BASE_CHAIN_LENGTH + difficulty
    resource_count = chain_length + 1
    chain_resources = rng.sample(RESOURCE_POOL, k=resource_count)
    start_resource = chain_resources[0]

    chain_action_ids = [
        f"{ACTION_ID_PREFIX_CHAIN}{index}" for index in range(1, chain_length + 1)
    ]
    trap_count = BASE_TRAP_COUNT + difficulty
    trap_action_ids = [
        f"{ACTION_ID_PREFIX_TRAP}{index}" for index in range(1, trap_count + 1)
    ]

    actions: list[dict[str, str]] = []

    for index, action_id in enumerate(chain_action_ids):
        actions.append(
            {
                "action_id": action_id,
                "consumes": chain_resources[index],
                "produces": chain_resources[index + 1],
                "kind": "chain",
            }
        )

    actions.append(
        {
            "action_id": GOAL_ACTION_ID,
            "consumes": chain_resources[-1],
            "produces": GOAL_RESOURCE,
            "kind": "goal",
        }
    )

    for index, trap_action_id in enumerate(trap_action_ids):
        consumed_resource = start_resource if index == 0 else rng.choice(chain_resources[:-1])
        produced_resource = rng.choice(JUNK_RESOURCE_POOL)
        actions.append(
            {
                "action_id": trap_action_id,
                "consumes": consumed_resource,
                "produces": produced_resource,
                "kind": "trap",
            }
        )

    shuffled_actions = list(actions)
    rng.shuffle(shuffled_actions)
    action_lines = [_render_action_line(action) for action in shuffled_actions]

    canonical_plan = chain_action_ids + [GOAL_ACTION_ID]
    expected_output = ",".join(canonical_plan)

    prompt_lines = [
        "Executive Functions / Complex Planning Task",
        "Find the shortest valid action sequence that reaches mission success.",
        f"Start inventory: {start_resource}=1",
        f"Goal: obtain {GOAL_RESOURCE}",
        "Constraints:",
        "- Each action can be used at most once.",
        "- You may execute an action only when its consumed resource is available.",
        "- Return the shortest successful plan.",
        "Action catalog:",
        *action_lines,
        "Output format: comma-separated action IDs (example: C1,C2,GOAL).",
    ]

    metadata = {
        "start_inventory": {start_resource: 1},
        "actions": shuffled_actions,
        "canonical_plan": canonical_plan,
        "trap_action_ids": trap_action_ids,
    }

    return ConflictPlanningSample(
        prompt="\n".join(prompt_lines),
        expected_output=expected_output,
        difficulty=difficulty,
        metadata=metadata,
    )


def simulate_conflict_plan(
    sample: ConflictPlanningSample,
    action_sequence: str | Iterable[str],
) -> bool:
    if isinstance(action_sequence, str):
        action_ids = parse_action_sequence(action_sequence)
    else:
        action_ids = [action_id.strip().upper() for action_id in action_sequence if action_id]

    actions_by_id = {
        action["action_id"].upper(): action for action in sample.metadata["actions"]
    }

    inventory = dict(sample.metadata["start_inventory"])
    used_actions: set[str] = set()

    for action_id in action_ids:
        action = actions_by_id.get(action_id.upper())
        if action is None:
            return False

        if action_id in used_actions:
            return False

        consumed_resource = action["consumes"]
        if inventory.get(consumed_resource, 0) < 1:
            return False

        inventory[consumed_resource] -= 1
        if inventory[consumed_resource] == 0:
            del inventory[consumed_resource]

        produced_resource = action["produces"]
        inventory[produced_resource] = inventory.get(produced_resource, 0) + 1
        used_actions.add(action_id)

    return inventory.get(GOAL_RESOURCE, 0) > 0


def generate_habit_override_dataset(
    num_samples: int,
    seed: int = DEFAULT_DATASET_SEED,
) -> list[HabitOverrideSample]:
    rng = random.Random(seed)
    return [
        generate_habit_override_sample(difficulty=_difficulty_for_index(index), rng=rng)
        for index in range(num_samples)
    ]


def generate_rule_shift_dataset(
    num_samples: int,
    seed: int = DEFAULT_DATASET_SEED,
) -> list[RuleShiftSample]:
    rng = random.Random(seed)
    return [
        generate_rule_shift_sample(difficulty=_difficulty_for_index(index), rng=rng)
        for index in range(num_samples)
    ]


def generate_conflict_planning_dataset(
    num_samples: int,
    seed: int = DEFAULT_DATASET_SEED,
) -> list[ConflictPlanningSample]:
    rng = random.Random(seed)
    return [
        generate_conflict_planning_sample(difficulty=_difficulty_for_index(index), rng=rng)
        for index in range(num_samples)
    ]


def build_cogniflex_dataset(
    num_samples_per_task: int,
    seed: int = DEFAULT_DATASET_SEED,
) -> dict[str, list[Any]]:
    habit_seed_offset = 11
    rule_seed_offset = 29
    planning_seed_offset = 47

    return {
        "habit_override": generate_habit_override_dataset(
            num_samples=num_samples_per_task,
            seed=seed + habit_seed_offset,
        ),
        "rule_shift": generate_rule_shift_dataset(
            num_samples=num_samples_per_task,
            seed=seed + rule_seed_offset,
        ),
        "conflict_planning": generate_conflict_planning_dataset(
            num_samples=num_samples_per_task,
            seed=seed + planning_seed_offset,
        ),
    }
