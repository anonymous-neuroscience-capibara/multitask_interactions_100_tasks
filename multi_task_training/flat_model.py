"""The flat, P-free PLRNN, in one place.

`FlatPLRNN` is `rnn_model.PLRNN` with the joint pipeline's call signature. The scoring
and loss helpers in `hierachical_model_task.rnn_model` call `model(inputs, task_ids)`
(lines 195/236/575) because the hierarchical model needs the subject index to select its
p_vector; the flat model has no per-task parameters, so it accepts and ignores that
argument. Nothing else differs -- same A/W/h/C/D, same dynamics, same native
initialisation, same parameter count. It is a subclass rather than an edit to
PLRNN.forward so that `rnn_model.PLRNN` itself stays untouched for its existing callers.

Task identity, where a flat model actually needs it (joint training over many tasks),
arrives as a one-hot appended to the INPUT by `tasks.dataset.MultiTaskDataset`, which
widens C from (M, 7) to (M, 7 + n_tasks). It does NOT arrive through `subject`.

NOTE: `scripts/train_indiv_models.py` and `foundation_model/finetune_flat_transfer.py`
still define their own identical copies of this class. They are left alone deliberately:
train_indiv_models.py is the script behind the 12k individual runs and is invoked by
sbatch on the cluster, so adding an import of a new file would mean the cluster copy of
the repo must have that file before any array resubmission. Migrate them to this module
whenever they are next edited for other reasons.
"""

from .rnn_model import PLRNN


class FlatPLRNN(PLRNN):
    """PLRNN with the joint pipeline's call signature: forward(inputs, subject=None)."""

    def forward(self, inputs, subject=None):
        return super().forward(inputs)
