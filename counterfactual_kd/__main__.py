"""
``python -m counterfactual_kd`` — entry point for the config-driven runner.

Usage::

    python -m counterfactual_kd run configs/round1_rerun.yaml
    python -m counterfactual_kd run configs/e1_random_init.yaml --stages stage1 stage2
    python -m counterfactual_kd validate configs/round1_rerun.yaml
    python -m counterfactual_kd show configs/e1_random_init.yaml
"""

from counterfactual_kd.cli import main

main()
