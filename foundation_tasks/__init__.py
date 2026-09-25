"""Foundation-model task battery for the hierarchical AL-RNN.

This package is the home for the foundation battery: the dynamically-rich
keystone tasks, the modifier engine, and the registry that assembles them with
the existing supervised suite (from `tasks/`). Every task conforms to the shared
interface (input <= BASE_INPUT_DIM=7, output OUTPUT_DIM=5) and is registered as
its own task_id / subject. Mode variants (pro/anti, match/nonmatch, n-back
depth, +delay, ...) are distinct tasks, as intended.

Layout:
    foundation_tasks/
        __init__.py        <- this registry (FOUNDATION_TASKS_MAP, TAGS, ...)
        modifiers.py       <- TaskModifier, DelayModifier (+delay)
        flip_flop.py       <- FlipFlop keystone
        sine_generation.py <- SineGeneration keystone
        n_back.py          <- NBack keystone
        <new task files>   <- batches encoded toward the ~80-100 target

Run `python -m foundation_tasks` to print the tagged battery table.

RL / value-based tasks (bandits, reversal, Daw two-step, IGT) and pure ML-RNN
benchmarks are deliberately excluded from the supervised battery.
"""

# Base supervised suite (stays in tasks/; the CognitiveTask base lives there too).
from tasks import (
    Arithmetics,
    CategoryDecision,
    ContextIntegration,
    CopyTask,
    DelayComparison,
    DelayedMatchToSample,
    DelayedResponse,
    DurationEstimation,
    GoNogo,
    IntervalDiscrimination,
    MultiSensoryIntegration,
    NoiseCleaner,
    PerceptualDecisionMaking,
)

# Foundation-battery tasks (encoded in this package).
from .flip_flop import FlipFlop
from .sine_generation import SineGeneration
from .modifiers import (
    with_delay,
    with_seq,
    with_int,
    DelayModifier,
    SeqModifier,
    IntModifier,
    TaskModifier,
)

# Working-memory probe batch.
from .sternberg import Sternberg
from .delay_paired_association import DelayPairedAssociation
from .spatial_wm_retro import SpatialWMRetro
from .change_detection import ChangeDetection

# Go/Anti batch.
from .reaction_time_response import RTResponse
from .stop_signal import StopSignal
from .anti_reach import AntiReach1D

# Cognitive-control batch.
from .stroop import Stroop
from .flanker import Flanker
from .simon import Simon
from .ax_cpt import AXCPT
from .task_switch import TaskSwitch

# Timing batch (scalar interval production -- the one timing modality base lacks;
# short/long duration judgment is already covered by DurationEstimation).
from .interval_reproduction import IntervalReproduction
from .one_two_three_go import OneTwoThreeGo

# Navigation batch.
from .head_direction import HeadDirection
from .path_integration import PathIntegration1D

# Yang-coverage batch (delay-match-to-category).
from .delay_match_category import DelayMatchCategory

# Additional distinct cognitive tasks.
from .dms_distractor import DMSDistractor
from .hierarchical_reasoning import HierarchicalReasoning
from .numerosity import Numerosity

# Literature-gap batch: 2D continuous attractor, categorical timing, attention.
from .path_integration_2d import PathIntegration2D
from .temporal_bisection import TemporalBisection
from .visual_search import VisualSearch
from .posner_cueing import PosnerCueing

# Animal-paradigm batch: pulse accumulation, deviance detection, auditory map.
from .poisson_clicks import PoissonClicks
from .oddball_detection import OddballDetection
from .sound_localization import SoundLocalization

# Classical-psychology batch: one task per distinct cognition type.
from .mental_rotation import MentalRotation
from .backward_masking import BackwardMasking
from .signal_detection import SignalDetection
from .serial_order_memory import SerialOrderMemory
from .conditional_reasoning import ConditionalReasoning
from .lexical_decision import LexicalDecision
from .attentional_blink import AttentionalBlink
from .wm_updating import WMUpdating

# Extension batch (validated individually; admitted to the battery).
from .relational_match import RelationalMatch
from .prospective_memory import ProspectiveMemory
from .temporal_order_judgment import TemporalOrderJudgment
from .metacog_optout import MetacognitiveOptOut
from .prior_integration import PriorIntegration
from .simultaneity_judgment import SimultaneityJudgment
from .gap_detection import GapDetection
from .simultaneity_judgment import SimultaneityJudgment
from .gap_detection import GapDetection

# Argmax / categorical batch (K-way selection; brings the battery to 100).
from .plurality_vote import PluralityVote
from .odd_one_out import OddOneOut

# Same phase-duration ranges used by the existing experiments.
duration_params = {
    "context": (5, 10),
    "stimulus": (10, 15),
    "delay": (10, 15),
    "response": (5, 10),
}


def _tag(
    accuracy_type,
    sequence_output=False,
    has_delay=False,
    requires_memory=False,
    requires_rotation=False,
    wm_load=0,
    n_fixed_points=0,
):
    return dict(
        accuracy_type=accuracy_type,
        sequence_output=sequence_output,
        has_delay=has_delay,
        requires_memory=requires_memory,
        requires_rotation=requires_rotation,
        wm_load=wm_load,
        n_fixed_points=n_fixed_points,
    )


# --- BASE backbone: the existing supervised suite (each variant a distinct task) ---
BASE_TASKS_MAP = {
    "NoiseCleaner": NoiseCleaner(duration_params, noise_level=0.5),
    "DelayPro": DelayedResponse(duration_params, mode="pro"),
    "DelayAnti": DelayedResponse(duration_params, mode="anti"),
    "CatPro": CategoryDecision(duration_params, mode="pro"),
    "CatAnti": CategoryDecision(duration_params, mode="anti"),
    "Match2Sample": DelayedMatchToSample(duration_params, mode="match"),
    "NonMatch2Sample": DelayedMatchToSample(duration_params, mode="nonmatch"),
    "CtxIntMod1": ContextIntegration(duration_params, relevant_modality=1),
    "CtxIntMod2": ContextIntegration(duration_params, relevant_modality=2),
    "ArithMultiply": Arithmetics(duration_params, mode="multiply"),
    "ArithAdd": Arithmetics(duration_params, mode="avg"),
    "CopyTask": CopyTask(duration_params, seq_len=8, num_symbols=4, delay_range=0),
    "GoNogo": GoNogo(duration_params),
    "PerceptualDM": PerceptualDecisionMaking(duration_params),
    "DelayedComparison": DelayComparison(duration_params),
    "DurationPro": DurationEstimation(duration_params, mode="pro"),
    "DurationAnti": DurationEstimation(duration_params, mode="anti"),
    "IntDisc": IntervalDiscrimination(duration_params),
    "MultiSens": MultiSensoryIntegration(duration_params),
}

# --- KEYSTONE dynamical tasks (break the near-linear regime) ---
KEYSTONE_TASKS_MAP = {
    "FlipFlop2": FlipFlop(duration_params, n_bits=2),  # 4 fixed points
    "FlipFlop3": FlipFlop(duration_params, n_bits=3),  # 8 fixed points / bitcode
    "SineGen": SineGeneration(duration_params),  # limit cycle / rotation
    # --- WM-probe batch ---
    "Sternberg": Sternberg(duration_params, set_size=3),
    "PairedAssoc": DelayPairedAssociation(duration_params),
    "SpatialWMRetro": SpatialWMRetro(duration_params),
    "ChangeDetect": ChangeDetection(duration_params, array_size=3),
    # --- Go/Anti batch ---
    "RTGo": RTResponse(duration_params, mode="pro"),
    "RTAnti": RTResponse(duration_params, mode="anti"),
    "StopSignal": StopSignal(duration_params),
    "AntiReach1D": AntiReach1D(duration_params),
    # --- Cognitive-control batch ---
    "Stroop": Stroop(duration_params),
    "Flanker": Flanker(duration_params),
    "Simon": Simon(duration_params),
    "AXCPT": AXCPT(duration_params),
    "TaskSwitch": TaskSwitch(duration_params),
    # --- Timing batch ---
    "IntervalReproduction": IntervalReproduction(duration_params),
    "OneTwoThreeGo": OneTwoThreeGo(duration_params),
    # --- Navigation batch (continuous-attractor probes) ---
    "HeadDirection": HeadDirection(duration_params),
    "PathIntegration1D": PathIntegration1D(duration_params),
    # --- Yang-coverage batch ---
    "DMC": DelayMatchCategory(duration_params, mode="match"),
    "DNMC": DelayMatchCategory(duration_params, mode="nonmatch"),
    # --- additional distinct cognitive tasks ---
    "DMSDistractor": DMSDistractor(duration_params),
    "HierReason": HierarchicalReasoning(duration_params),
    "Numerosity": Numerosity(duration_params),
    # --- literature-gap batch ---
    "PathIntegration2D": PathIntegration2D(duration_params),  # 2D continuous attractor
    "TemporalBisection": TemporalBisection(duration_params),  # categorical timing
    "VisualSearch": VisualSearch(duration_params, set_size=4),  # attention / search
    "PosnerCueing": PosnerCueing(duration_params),  # covert spatial attention
    # --- animal-paradigm batch ---
    "PoissonClicks": PoissonClicks(duration_params),  # pulse evidence accumulation
    "OddballDetection": OddballDetection(duration_params),  # deviance detection
    "SoundLocalization": SoundLocalization(duration_params),  # auditory azimuth
    # --- classical-psychology batch (one per cognition type) ---
    "MentalRotation": MentalRotation(duration_params),  # mental imagery
    "BackwardMasking": BackwardMasking(duration_params),  # perception / masking
    "SignalDetection": SignalDetection(duration_params),  # psychophysics / detection
    "SerialOrderMemory": SerialOrderMemory(duration_params),  # memory for order
    "ConditionalReasoning": ConditionalReasoning(duration_params),  # deduction
    "LexicalDecision": LexicalDecision(duration_params),  # language / lexical access
    "AttentionalBlink": AttentionalBlink(duration_params),  # temporal attention
    "WMUpdating": WMUpdating(duration_params),  # executive updating
    # --- extension batch (validated individually; admitted to the battery) ---
    "RelationalMatch": RelationalMatch(duration_params),  # relations-of-relations
    "ProspectiveMemory": ProspectiveMemory(duration_params),  # deferred intention
    "TemporalOrderJudgment": TemporalOrderJudgment(duration_params),  # order timing
    "MetacognitiveOptOut": MetacognitiveOptOut(duration_params),  # uncertainty opt-out
    "PriorIntegration": PriorIntegration(duration_params),  # prior x likelihood
    "SimultaneityJudgment": SimultaneityJudgment(duration_params),  # coincidence timing
    "GapDetection": GapDetection(duration_params),  # temporal acuity
    # --- argmax / categorical batch (K-way selection) ---
    "PluralityVote": PluralityVote(duration_params),  # 3-way temporal mode
    "OddOneOut": OddOneOut(duration_params),  # 4-way deviance localization
}

# --- MODIFIER-generated tasks (the scaling engine) ---
# Any base or keystone task can be modified; each variant is its own task_id.
_modifiable = {**BASE_TASKS_MAP, **KEYSTONE_TASKS_MAP}

# +delay : insert a retention gap before response (reactive -> working memory).
_DELAY_BASES = [
    "GoNogo",
    "PerceptualDM",
    "CatPro",
    "CatAnti",
    "NoiseCleaner",
    "ArithAdd",
    "ArithMultiply",
    "MultiSens",
    "CopyTask",  # gap inserted before recall -> delayed-recall copy
    "Stroop",
    "Flanker",
    "Simon",
    "TaskSwitch",  # delayed conflict resolution
]
DELAYED_TASKS_MAP = {
    f"{name}+delay": with_delay(_modifiable[name]) for name in _DELAY_BASES
}

# +seq : rotate the angular response over the window (time-varying output).
# +int : rotate the remembered direction across an inserted delay (integration).
# Both apply only to single-target angular bases (loss_channels == [1, 2]).
# +seq/+int append an omega-cue channel, so the base must have a spare channel
# within BASE_INPUT_DIM=7. CtxIntMod1/2 already use 7 channels and are excluded.
_ANGULAR_BASES = [
    "DelayPro",
    "DelayAnti",
    "NoiseCleaner",
    "MultiSens",
    "PerceptualDM",
    "RTGo",
    "RTAnti",
    "SpatialWMRetro",
]
SEQ_TASKS_MAP = {f"{name}+seq": with_seq(_modifiable[name]) for name in _ANGULAR_BASES}
INT_TASKS_MAP = {f"{name}+int": with_int(_modifiable[name]) for name in _ANGULAR_BASES}
# int x seq composition: rotate memory across a delay AND emit a moving output.
# Mod-Cog's headline compositional demand (distinct from int or seq alone).
INTSEQ_TASKS_MAP = {
    f"{name}+int+seq": with_seq(with_int(_modifiable[name])) for name in _ANGULAR_BASES
}

# Tasks that stay encoded but are held OUT of the joint foundation battery.
# CopyTask+delay: persistent failure (~20%), still generated by the modifier
# engine. Arithmetic (+ its +delay variants): ArithAdd collapsed to ~0%
# (degenerate readout), the rest persistently weak -- files kept, held out.
# (n-back, ChangeReport, TransitiveInference, ProactiveInterference, TimeToContact
# have been deleted entirely.)
EXCLUDED_FROM_BATTERY = {
    "CopyTask+delay",
    "ArithAdd",
    "ArithMultiply",
    "ArithAdd+delay",
    "ArithMultiply+delay",
}

FOUNDATION_TASKS_MAP = {
    k: v
    for k, v in {
        **BASE_TASKS_MAP,
        **KEYSTONE_TASKS_MAP,
        **DELAYED_TASKS_MAP,
        **SEQ_TASKS_MAP,
        **INT_TASKS_MAP,
        **INTSEQ_TASKS_MAP,
    }.items()
    if k not in EXCLUDED_FROM_BATTERY
}

# --- analysis covariates per task (editable; modifiers extend these) ---
TAGS = {
    "NoiseCleaner": _tag("angular", requires_memory=True),
    "DelayPro": _tag("angular", has_delay=True, requires_memory=True),
    "DelayAnti": _tag("angular", has_delay=True, requires_memory=True),
    "CatPro": _tag("binary"),
    "CatAnti": _tag("binary"),
    "Match2Sample": _tag("binary", has_delay=True, requires_memory=True, wm_load=1),
    "NonMatch2Sample": _tag("binary", has_delay=True, requires_memory=True, wm_load=1),
    "CtxIntMod1": _tag("angular", requires_memory=True),
    "CtxIntMod2": _tag("angular", requires_memory=True),
    "ArithMultiply": _tag("binary", requires_memory=True),
    "ArithAdd": _tag("binary", requires_memory=True),
    "CopyTask": _tag(
        "copytask",
        sequence_output=True,
        has_delay=True,
        requires_memory=True,
        wm_load=8,
    ),
    "GoNogo": _tag("binary"),
    "PerceptualDM": _tag("angular"),
    "DelayedComparison": _tag(
        "binary", has_delay=True, requires_memory=True, wm_load=1
    ),
    "DurationPro": _tag("binary", requires_memory=True),
    "DurationAnti": _tag("binary", requires_memory=True),
    "IntDisc": _tag("binary", requires_memory=True),
    "MultiSens": _tag("angular"),
    "FlipFlop2": _tag(
        "flipflop",
        sequence_output=True,
        requires_memory=True,
        wm_load=2,
        n_fixed_points=4,
    ),
    "FlipFlop3": _tag(
        "flipflop",
        sequence_output=True,
        requires_memory=True,
        wm_load=3,
        n_fixed_points=8,
    ),
    "SineGen": _tag("copytask", sequence_output=True, requires_rotation=True),
    # WM-probe batch
    "Sternberg": _tag("binary", has_delay=True, requires_memory=True, wm_load=3),
    "PairedAssoc": _tag("binary", has_delay=True, requires_memory=True, wm_load=1),
    "SpatialWMRetro": _tag("angular", has_delay=True, requires_memory=True, wm_load=2),
    "ChangeDetect": _tag("binary", has_delay=True, requires_memory=True, wm_load=3),
    # Go/Anti batch
    "RTGo": _tag("angular"),
    "RTAnti": _tag("angular"),
    "StopSignal": _tag("binary", requires_memory=True),
    "AntiReach1D": _tag("scalar"),
    # Cognitive-control batch
    "Stroop": _tag("binary"),
    "Flanker": _tag("binary"),
    "Simon": _tag("binary"),
    "AXCPT": _tag("binary", has_delay=True, requires_memory=True, wm_load=1),
    "TaskSwitch": _tag("binary"),
    # Timing batch
    "IntervalReproduction": _tag("scalar", requires_memory=True),
    "OneTwoThreeGo": _tag("scalar", requires_memory=True),
    # Navigation batch (continuous attractors)
    "HeadDirection": _tag(
        "copytask", sequence_output=True, requires_memory=True, requires_rotation=True
    ),
    "PathIntegration1D": _tag("scalar", requires_memory=True),
    # Yang-coverage batch
    "DMC": _tag("binary", has_delay=True, requires_memory=True, wm_load=1),
    "DNMC": _tag("binary", has_delay=True, requires_memory=True, wm_load=1),
    # additional distinct cognitive tasks
    "DMSDistractor": _tag("binary", has_delay=True, requires_memory=True, wm_load=1),
    "HierReason": _tag("binary"),
    "Numerosity": _tag("scalar", requires_memory=True),
    # literature-gap batch
    "PathIntegration2D": _tag(
        "vector", requires_memory=True, requires_rotation=False
    ),
    "TemporalBisection": _tag("binary", requires_memory=True),
    "VisualSearch": _tag("binary", requires_memory=True, wm_load=1),
    "PosnerCueing": _tag("angular", has_delay=True, requires_memory=True),
    # animal-paradigm batch
    "PoissonClicks": _tag("binary", requires_memory=True),
    "OddballDetection": _tag("binary", requires_memory=True),
    "SoundLocalization": _tag("scalar", requires_memory=True),
    # classical-psychology batch
    "MentalRotation": _tag(
        "binary", has_delay=True, requires_memory=True, requires_rotation=True
    ),
    "BackwardMasking": _tag("binary"),
    "SignalDetection": _tag("binary"),
    "SerialOrderMemory": _tag("binary", requires_memory=True, wm_load=4),
    "ConditionalReasoning": _tag("binary"),
    "LexicalDecision": _tag("binary", requires_memory=True),
    "AttentionalBlink": _tag("binary", requires_memory=True),
    "WMUpdating": _tag("angular", requires_memory=True, wm_load=2),
    # extension batch
    "RelationalMatch": _tag("binary", requires_memory=True),
    "ProspectiveMemory": _tag("binary", requires_memory=True),
    "TemporalOrderJudgment": _tag("binary", requires_memory=True),
    "MetacognitiveOptOut": _tag("binary"),
    "PriorIntegration": _tag("angular", requires_memory=True),
    "SimultaneityJudgment": _tag("binary", requires_memory=True),
    "GapDetection": _tag("binary", requires_memory=True),
    # argmax / categorical batch
    "PluralityVote": _tag("argmax", has_delay=True, requires_memory=True, wm_load=3),
    "OddOneOut": _tag("argmax"),  # simultaneous perceptual outlier judgment (no delay)
}

# Auto-derive tags for modifier variants from the base tags (no hand-tuning).
for _name in _DELAY_BASES:
    _t = dict(TAGS[_name])
    _t.update(has_delay=True, requires_memory=True)
    TAGS[f"{_name}+delay"] = _t
for _name in _ANGULAR_BASES:
    # +seq: time-varying rotating output.
    _t = dict(TAGS[_name])
    _t.update(accuracy_type="copytask", sequence_output=True, requires_rotation=True)
    TAGS[f"{_name}+seq"] = _t
    # +int: rotate the memory across an inserted delay (integration).
    _t2 = dict(TAGS[_name])
    _t2.update(has_delay=True, requires_memory=True, requires_rotation=True)
    TAGS[f"{_name}+int"] = _t2
    # +int+seq: composition (rotate memory across delay + moving output).
    _t3 = dict(TAGS[_name])
    _t3.update(
        accuracy_type="copytask",
        sequence_output=True,
        has_delay=True,
        requires_memory=True,
        requires_rotation=True,
    )
    TAGS[f"{_name}+int+seq"] = _t3

assert set(TAGS) == set(FOUNDATION_TASKS_MAP) | EXCLUDED_FROM_BATTERY, (
    "TAGS and tasks out of sync"
)
