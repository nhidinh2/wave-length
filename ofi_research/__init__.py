"""ofi_research: order-flow-imbalance microstructure research pipeline.

This package tests whether order-flow imbalance (OFI) and related
microstructure signals predict future midprice movement out of sample,
and whether any edge survives realistic transaction costs.

It deliberately stops at a clean linear/microstructure baseline. The code is
organized so that an impulse-response / Green's-function layer can be added
later (see README "Extension points") WITHOUT touching the feature, target,
split, or backtest contracts defined here.
"""

__version__ = "0.1.0"
