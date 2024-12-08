import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    # example of collate_fn
    result_batch["true_wav"] = torch.vstack(
        [elem["wav"] for elem in dataset_items]
    )
    result_batch["mel_spec"] = torch.vstack(
        [elem["mel_spec"] for elem in dataset_items]
    )

    return result_batch
