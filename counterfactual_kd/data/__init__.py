from counterfactual_kd.data.loaders import load_data, DATASET_LOADERS, DATASET_NUM_LABELS, DATASET_TARGET_LABELS


def __getattr__(name):
    """Lazy import for torch-dependent classes."""
    if name in ("TextClassificationDataset", "make_dataloader", "make_dataloaders",
                "collate_fn_factory"):
        from counterfactual_kd.data import dataset
        return getattr(dataset, name)
    raise AttributeError(f"module 'counterfactual_kd.data' has no attribute {name}")
