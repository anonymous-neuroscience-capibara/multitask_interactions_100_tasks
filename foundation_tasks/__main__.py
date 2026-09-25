"""Print the tagged foundation battery: `python -m foundation_tasks`."""

from foundation_tasks import (
    BASE_TASKS_MAP,
    KEYSTONE_TASKS_MAP,
    DELAYED_TASKS_MAP,
    SEQ_TASKS_MAP,
    INT_TASKS_MAP,
    INTSEQ_TASKS_MAP,
    FOUNDATION_TASKS_MAP,
    TAGS,
)

print(f"Base tasks:     {len(BASE_TASKS_MAP)}")
print(f"Keystone tasks: {len(KEYSTONE_TASKS_MAP)}")
print(f"+delay tasks:   {len(DELAYED_TASKS_MAP)}")
print(f"+seq tasks:     {len(SEQ_TASKS_MAP)}")
print(f"+int tasks:     {len(INT_TASKS_MAP)}")
print(f"+int+seq tasks: {len(INTSEQ_TASKS_MAP)}")
print(f"Total battery:  {len(FOUNDATION_TASKS_MAP)}\n")

hdr = (f"{'task':20s} {'accuracy':9s} {'seq':4s} {'delay':6s} "
       f"{'mem':4s} {'rot':4s} {'wm':3s} {'fp':3s}")
print(hdr)
print("-" * len(hdr))
for name in FOUNDATION_TASKS_MAP:
    t = TAGS[name]
    print(f"{name:20s} {t['accuracy_type']:9s} "
          f"{str(t['sequence_output']):4s} {str(t['has_delay']):6s} "
          f"{str(t['requires_memory']):4s} {str(t['requires_rotation']):4s} "
          f"{t['wm_load']:>3d} {t['n_fixed_points']:>3d}")
