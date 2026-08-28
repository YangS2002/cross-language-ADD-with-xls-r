import math
import random
from collections import defaultdict

import torch
from torch.utils.data import Sampler


def _to_scalar(x):
    if torch.is_tensor(x):
        if x.numel() != 1:
            raise ValueError(
                f"Expected scalar tensor, got shape {tuple(x.shape)}"
            )
        return x.item()

    return x


def _optional_int(x):
    if x is None:
        return None

    if isinstance(x, str):
        x = x.strip()

        if x.lower() in {
            "",
            "none",
            "null"
        }:
            return None

    return int(x)


def extract_metadata_from_dataset(dataset):
    if not hasattr(dataset, "spks"):
        raise RuntimeError(
            "Dataset does not contain `spks`."
        )

    if not hasattr(dataset, "synthesizerlabels"):
        raise RuntimeError(
            "Dataset does not contain `synthesizerlabels`."
        )

    if not hasattr(dataset, "synthesizers"):
        raise RuntimeError(
            "Dataset does not contain `synthesizers`."
        )

    speaker_ids = [
        str(_to_scalar(x))
        for x in dataset.spks
    ]

    vocoder_ids = list(
        dataset.synthesizerlabels
    )

    labels = []

    for vocoder_id in vocoder_ids:
        vocoder_label = dataset.synthesizers[
            vocoder_id
        ]

        labels.append(
            1 if vocoder_label == 0 else 0
        )

    if len(speaker_ids) != len(dataset):
        raise RuntimeError(
            f"Speaker metadata length mismatch: "
            f"{len(speaker_ids)} vs {len(dataset)}"
        )

    if len(labels) != len(dataset):
        raise RuntimeError(
            f"Label metadata length mismatch: "
            f"{len(labels)} vs {len(dataset)}"
        )

    return speaker_ids, labels, vocoder_ids


class SameSpeakerBatchSampler(Sampler):
    """
    Each speaker contributes:
        [real1, real2, fake1, fake2]

    The sampler is only used to keep training balanced across real/fake
    and speakers. Genuine-normalized teacher targets are computed from
    the precomputed genuine speaker bank, not from these two random reals.
    """

    def __init__(
        self,
        dataset,
        speakers_per_batch=4,
        batches_per_epoch=None,
        seed=1234
    ):
        self.dataset = dataset
        self.speakers_per_batch = int(
            speakers_per_batch
        )
        self.batches_per_epoch = _optional_int(
            batches_per_epoch
        )
        self.seed = int(seed)
        self.epoch = 0

        if self.speakers_per_batch <= 0:
            raise ValueError(
                "speakers_per_batch must be > 0."
            )

        speaker_ids, labels, vocoder_ids = (
            extract_metadata_from_dataset(
                dataset
            )
        )

        real_pool = defaultdict(list)
        fake_pool = defaultdict(list)
        fake_by_vocoder = defaultdict(
            lambda: defaultdict(list)
        )

        for idx, (
            spk,
            label,
            vocoder
        ) in enumerate(
            zip(
                speaker_ids,
                labels,
                vocoder_ids
            )
        ):
            if label == 1:
                real_pool[spk].append(idx)
            else:
                fake_pool[spk].append(idx)
                fake_by_vocoder[spk][
                    vocoder
                ].append(idx)

        self.real_pool = real_pool
        self.fake_pool = fake_pool
        self.fake_by_vocoder = fake_by_vocoder

        self.valid_speakers = sorted([
            spk
            for spk in real_pool
            if (
                len(real_pool[spk]) >= 2
                and len(fake_pool[spk]) >= 2
            )
        ])

        if len(self.valid_speakers) == 0:
            raise RuntimeError(
                "No speaker has at least "
                "2 bonafide and 2 spoof samples."
            )

        if self.batches_per_epoch is None:
            batch_size = (
                self.speakers_per_batch * 4
            )

            self.batches_per_epoch = max(
                1,
                math.ceil(
                    len(dataset) / batch_size
                )
            )

        print(
            "\n========== "
            "SameSpeakerBatchSampler "
            "=========="
        )
        print(
            f"Valid speakers: "
            f"{len(self.valid_speakers)}"
        )
        print(
            f"Speakers per batch: "
            f"{self.speakers_per_batch}"
        )
        print(
            f"Effective batch size: "
            f"{self.speakers_per_batch * 4}"
        )
        print(
            f"Batches per epoch: "
            f"{self.batches_per_epoch}"
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batches_per_epoch

    def _sample_two_fakes(
        self,
        spk,
        rng
    ):
        by_vocoder = self.fake_by_vocoder[
            spk
        ]

        vocoders = [
            vocoder
            for vocoder, ids in (
                by_vocoder.items()
            )
            if len(ids) > 0
        ]

        if len(vocoders) >= 2:
            vocoder1, vocoder2 = (
                rng.sample(
                    vocoders,
                    2
                )
            )

            return (
                rng.choice(
                    by_vocoder[vocoder1]
                ),
                rng.choice(
                    by_vocoder[vocoder2]
                )
            )

        return tuple(
            rng.sample(
                self.fake_pool[spk],
                2
            )
        )

    def __iter__(self):
        rng = random.Random(
            self.seed + self.epoch
        )

        for _ in range(
            self.batches_per_epoch
        ):
            if (
                len(self.valid_speakers)
                >= self.speakers_per_batch
            ):
                speakers = rng.sample(
                    self.valid_speakers,
                    self.speakers_per_batch
                )
            else:
                speakers = [
                    rng.choice(
                        self.valid_speakers
                    )
                    for _ in range(
                        self.speakers_per_batch
                    )
                ]

            batch = []

            for spk in speakers:
                real1, real2 = rng.sample(
                    self.real_pool[spk],
                    2
                )

                fake1, fake2 = (
                    self._sample_two_fakes(
                        spk,
                        rng
                    )
                )

                batch.extend([
                    real1,
                    real2,
                    fake1,
                    fake2
                ])

            yield batch