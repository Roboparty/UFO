import json
import tempfile
import unittest
from pathlib import Path

import torch

from humanoidverse.perception.terrain_dataset import (
    TerrainPerceptionChunkWriter,
    TerrainPerceptionFrameBatch,
    TerrainPerceptionSequenceDataset,
)
from humanoidverse.train_terrain_perception import (
    ChunkGroupedShuffleSampler,
    train_terrain_perception,
)


def make_frame(step: int, episode_ids: tuple[int, ...] = (0, 0)) -> TerrainPerceptionFrameBatch:
    batch_size = len(episode_ids)
    partial = torch.full((batch_size, 273), float("nan"))
    mask = torch.zeros((batch_size, 273), dtype=torch.bool)
    partial[:, 50:70] = 0.5 + step * 0.01
    mask[:, 50:70] = True
    return TerrainPerceptionFrameBatch(
        partial_map=partial,
        visible_mask=mask,
        pelvis_pos_w=torch.tensor([[step * 0.01, env, 0.8] for env in range(batch_size)]),
        heading_yaw_w=torch.zeros(batch_size),
        timestamp_s=torch.full((batch_size,), step * 0.1),
        proprio=torch.full((batch_size, 5), float(step)),
        gt_terrain_actor=torch.full((batch_size, 273), 0.8),
        episode_id=torch.tensor(episode_ids),
        env_id=torch.arange(batch_size),
        terrain_type=torch.arange(batch_size),
    )


class TerrainPerceptionDatasetTest(unittest.TestCase):
    def test_writer_uses_projected_schema_and_atomic_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            with TerrainPerceptionChunkWriter(directory, chunk_steps=2, metadata={"camera": "clean"}) as writer:
                self.assertIsNone(writer.append(make_frame(0)))
                output = writer.append(make_frame(1))
                self.assertIsNotNone(output)
                writer.append(make_frame(2))

            root = Path(directory)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual([item["steps"] for item in manifest["chunks"]], [2, 1])
            self.assertEqual(manifest["metadata"], {"camera": "clean"})
            self.assertFalse(any(path.name.startswith(".") for path in root.iterdir()))
            payload = torch.load(root / "chunk_000000.pt", weights_only=True)
            self.assertNotIn("depth", payload)
            self.assertEqual(tuple(payload["partial_map"].shape), (2, 2, 273))

    def test_sequence_dataset_never_crosses_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = TerrainPerceptionChunkWriter(directory, chunk_steps=8)
            writer.append(make_frame(0))
            writer.append(make_frame(1))
            writer.append(make_frame(2, (1, 0)))
            writer.append(make_frame(3, (1, 0)))
            writer.close()

            dataset = TerrainPerceptionSequenceDataset(directory, sequence_steps=3, history_seconds=0.6)

            self.assertEqual(len(dataset), 2)
            for sample in dataset:
                self.assertEqual(tuple(sample["partial_map"].shape), (3, 273))
                self.assertEqual(tuple(sample["proprio"].shape), (3, 5))
                self.assertTrue(torch.all(sample["timestamp_s"][1:] >= sample["timestamp_s"][:-1]))

    def test_writer_rejects_invalid_visible_value(self):
        frame = make_frame(0)
        frame.visible_mask[:, 0] = True
        with tempfile.TemporaryDirectory() as directory:
            writer = TerrainPerceptionChunkWriter(directory)
            with self.assertRaisesRegex(ValueError, "visible partial-map"):
                writer.append(frame)

    def test_writer_does_not_overwrite_existing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = TerrainPerceptionChunkWriter(directory)
            writer.append(make_frame(0))
            writer.close()
            with self.assertRaises(FileExistsError):
                TerrainPerceptionChunkWriter(directory)

    def test_one_epoch_supervised_training_writes_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "dataset"
            writer = TerrainPerceptionChunkWriter(dataset_dir, chunk_steps=8)
            for step in range(8):
                writer.append(make_frame(step))
            writer.close()

            summary = train_terrain_perception(
                dataset_dir=dataset_dir,
                output_dir=root / "model",
                validation_dataset_dir=None,
                sequence_steps=2,
                history_seconds=0.6,
                epochs=1,
                batch_size=4,
                learning_rate=1.0e-3,
                hidden_channels=2,
                device="cpu",
                seed=3,
            )

            self.assertEqual(len(summary["history"]), 1)
            self.assertTrue((root / "model" / "latest.pt").is_file())
            self.assertTrue((root / "model" / "best.pt").is_file())

    def test_training_sampler_does_not_interleave_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = TerrainPerceptionChunkWriter(directory, chunk_steps=4)
            for step in range(8):
                writer.append(make_frame(step))
            writer.close()
            dataset = TerrainPerceptionSequenceDataset(
                directory,
                sequence_steps=2,
                history_seconds=0.6,
            )

            order = list(ChunkGroupedShuffleSampler(dataset, seed=7))
            chunk_order = [dataset.chunk_index_for_sample(index) for index in order]
            transitions = sum(left != right for left, right in zip(chunk_order, chunk_order[1:]))

            self.assertEqual(sorted(order), list(range(len(dataset))))
            self.assertLessEqual(transitions, 1)


if __name__ == "__main__":
    unittest.main()
