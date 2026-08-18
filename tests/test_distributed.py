import pytest

from semicon_restore.distributed import DistributedInfo, ShardSampler, reduce_sum


def shards(count: int, world_size: int) -> list[list[int]]:
    return [list(ShardSampler(count, DistributedInfo(True, rank, rank, world_size, False)))
            for rank in range(world_size)]


def test_single_process_info_is_the_identity_case():
    info = DistributedInfo()
    assert info.primary and info.world_size == 1 and not info.enabled
    assert list(ShardSampler(5, info)) == [0, 1, 2, 3, 4]


def test_shards_are_disjoint_and_cover_every_index():
    # A repeated index would be counted twice in the summed validation total and a missing one would be
    # dropped, so exact partition is the property that makes sharded validation equal the single-process
    # metric rather than approximate it.
    for count in (0, 1, 7, 16, 31):
        for world_size in (1, 2, 3, 4):
            parts = shards(count, world_size)
            flat = [index for part in parts for index in part]
            assert sorted(flat) == list(range(count))
            assert len(flat) == len(set(flat))


def test_shard_lengths_differ_by_at_most_one_and_are_not_padded():
    # DistributedSampler would pad the short shard by repeating samples to equalise the lengths; the
    # ragged split is deliberate, so an odd count must produce unequal shards.
    lengths = [len(part) for part in shards(31, 4)]
    assert lengths == [8, 8, 8, 7]
    assert max(lengths) - min(lengths) <= 1


def test_summing_shard_totals_reproduces_the_single_process_mean():
    values = [0.1 * index for index in range(31)]
    reference = sum(values) / len(values)
    for world_size in (2, 3, 4):
        parts = shards(31, world_size)
        totals = [{"sum": sum(values[index] for index in part), "count": float(len(part))} for part in parts]
        pooled = {key: sum(total[key] for total in totals) for key in ("sum", "count")}
        assert pooled["sum"] / pooled["count"] == pytest.approx(reference)


def test_reduce_sum_is_a_copy_when_distribution_is_disabled():
    # The disabled path is what a single-process run takes, and it must not touch a process group that
    # was never created. Returning a copy also keeps the caller's dict from being mutated in place.
    values = {"loss": 1.5, "count": 4.0}
    reduced = reduce_sum(values, DistributedInfo())
    assert reduced == values and reduced is not values
    assert reduce_sum(values, DistributedInfo(False, 3, 3, 4, False)) == values


def test_device_wraps_the_local_rank_over_the_visible_gpu_count():
    # Two ranks sharing one card is how a multi-process run is smoke tested on a single GPU. With one
    # rank per GPU the wrap is the identity, so the same expression covers both layouts.
    assert DistributedInfo(True, 1, 1, 2, False).device.type == "cpu"
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to check device placement")
    visible = torch.cuda.device_count()
    for local_rank in range(4):
        device = DistributedInfo(True, local_rank, local_rank, 4, True).device
        assert device.index == local_rank % visible
