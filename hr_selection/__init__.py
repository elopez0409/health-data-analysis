"""HR source selection pipeline.

Train a model that, given multiple wearable HR readings for the same time
window, predicts which source is closest to a reference (ECG / Polar H10).

Synthetic data is written in the exact on-disk formats of the real datasets
(BigIdeasLab_STEP, GalaxyPPG, PPG-DaLiA) so real data can be dropped in
unchanged through the same adapters.
"""

from hr_selection import config

__all__ = ["config"]
