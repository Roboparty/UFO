import json
import tempfile
import unittest
from pathlib import Path

import torch

from humanoidverse.perception.terrain_dataset import (
    OdometryFreeTerrainPerceptionFrameBatch,
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


def make_odometry_free_frame(
    step: int,
    episode_ids: tuple[int, ...] = (0, 0),
) -> OdometryFreeTerrainPerceptionFrameBatch:
    world_frame = make_frame(step, episode_ids)
    return OdometryFreeTerrainPerceptionFrameBatch(
        partial_map=world_frame.partial_map,
        visible_mask=world_frame.visible_mask,
        timestamp_s=world_frame.timestamp_s,
        proprio=world_frame.proprio,
        gt_terrain_actor=world_frame.gt_terrain_actor,
        episode_id=world_frame.episode_id,
        env_id=world_frame.env_id,
        terrain_type=world_frame.terrain_type,
        frame_valid=torch.ones(len(episode_ids), dtype=torch.bool),
    )


class TerrainPerceptionDatasetTest(unittest.TestCase):
    def test_odometry_free_schema_contains_no_world_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            with TerrainPerceptionChunkWriter(directory, chunk_steps=2, odometry_free=True) as writer:
                writer.append(make_odometry_free_frame(0))
                writer.append(make_odometry_free_frame(1))

            root = Path(directory)
            manifest = json.loads((root / "manifest.json").read_text())
            payload = torch.load(root / "chunk_000000.pt", weights_only=True)
            dataset = TerrainPerceptionSequenceDataset(root, sequence_steps=2, history_seconds=0.6)
            sample = dataset[0]

            self.assertEqual(manifest["version"], 3)
            self.assertEqual(manifest["schema"], "odometry_free_local")
            self.assertTrue(dataset.odometry_free)
            self.assertTrue(sample["frame_valid"].all())
            for forbidden in ("pelvis_pos_w", "heading_yaw_w", "camera_pos_w", "camera_quat_w"):
                self.assertNotIn(forbidden, payload)
                self.assertNotIn(forbidden, sample)

    def test_v3_sequences_compact_30hz_frames_from_50hz_control_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            with TerrainPerceptionChunkWriter(
                directory,
                chunk_steps=64,
                odometry_free=True,
            ) as writer:
                for step in range(50):
                    frame = make_odometry_free_frame(step)
                    fresh = step == 0 or ((step * 3) // 5 != ((step - 1) * 3) // 5)
                    frame.frame_valid[:] = fresh
                    if not fresh:
                        frame.visible_mask.zero_()
                        frame.partial_map.fill_(float("nan"))
                    writer.append(frame)

            dataset = TerrainPerceptionSequenceDataset(
                directory,
                sequence_steps=10,
                history_seconds=0.6,
            )
            sample = dataset[-1]
            self.assertTrue(sample["frame_valid"].all())
            self.assertTrue(torch.all(sample["timestamp_s"][1:] > sample["timestamp_s"][:-1]))
            # Compacted camera histories retain the real 30 Hz time spacing,
            # rather than inserting fake duplicate control frames.
            self.assertGreater(float(sample["timestamp_s"][-1] - sample["timestamp_s"][0]), 0.25)

    def test_v3_sequences_cross_chunk_boundaries_without_crossing_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            with TerrainPerceptionChunkWriter(
                directory,
                chunk_steps=4,
                odometry_free=True,
            ) as writer:
                for step in range(8):
                    episodes = (0, 0) if step < 6 else (1, 0)
                    writer.append(make_odometry_free_frame(step, episodes))

            dataset = TerrainPerceptionSequenceDataset(
                directory,
                sequence_steps=3,
                history_seconds=0.6,
            )
            # env 0 has four samples before its reset; env 1 has six samples.
            self.assertEqual(len(dataset), 10)
            crossing = [
                dataset[index]
                for index in range(len(dataset))
                if dataset.chunk_index_for_sample(index) == 1
                and int(dataset[index]["env_id"]) == 1
            ]
            self.assertTrue(crossing)
            self.assertTrue(
                any(
                    torch.equal(sample["timestamp_s"], torch.tensor([0.2, 0.3, 0.4]))
                    for sample in crossing
                )
            )
            self.assertFalse(
                any(
                    int(sample["env_id"]) == 0 and int(sample["episode_id"]) == 1
                    for sample in dataset
                )
            )

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
            writer = TerrainPerceptionChunkWriter(
                dataset_dir,
                chunk_steps=8,
                metadata={
                    "terrain_component_names": [
                        "flat",
                        "slope",
                        "stairs_up",
                        "stairs_down",
                        "rough",
                        "platforms",
                    ]
                },
            )
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
            self.assertEqual(summary["stairs_terrain_ids"], [2, 3])
            self.assertTrue((root / "model" / "latest.pt").is_file())
            self.assertTrue((root / "model" / "best.pt").is_file())

    def test_odometry_free_training_uses_age_only_motion_feature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "dataset"
            writer = TerrainPerceptionChunkWriter(
                dataset_dir,
                chunk_steps=8,
                metadata={"terrain_component_names": ["flat", "slope", "stairs_up"]},
                odometry_free=True,
            )
            for step in range(8):
                writer.append(make_odometry_free_frame(step))
            writer.close()

            train_terrain_perception(
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
                history_mode="no_odometry",
            )
            checkpoint = torch.load(root / "model" / "best.pt", map_location="cpu", weights_only=True)

            self.assertEqual(checkpoint["config"]["history_mode"], "no_odometry")
            self.assertEqual(checkpoint["config"]["motion_feature_dim"], 1)

            with self.assertRaisesRegex(ValueError, "requires history_mode='no_odometry'"):
                train_terrain_perception(
                    dataset_dir=dataset_dir,
                    output_dir=root / "invalid_model",
                    validation_dataset_dir=None,
                    sequence_steps=2,
                    history_seconds=0.6,
                    epochs=1,
                    batch_size=4,
                    learning_rate=1.0e-3,
                    hidden_channels=2,
                    device="cpu",
                    seed=3,
                    history_mode="egomotion_warp",
                )

    def test_supervised_training_resumes_model_optimizer_and_sampler_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "dataset"
            model_dir = root / "model"
            writer = TerrainPerceptionChunkWriter(
                dataset_dir,
                chunk_steps=8,
                metadata={"terrain_component_names": ["flat", "slope", "stairs_up"]},
                odometry_free=True,
            )
            for step in range(8):
                writer.append(make_odometry_free_frame(step))
            writer.close()

            train_terrain_perception(
                dataset_dir=dataset_dir,
                output_dir=model_dir,
                validation_dataset_dir=None,
                sequence_steps=2,
                history_seconds=0.6,
                epochs=1,
                batch_size=4,
                learning_rate=1.0e-3,
                hidden_channels=2,
                device="cpu",
                seed=3,
                history_mode="no_odometry",
            )
            first = torch.load(model_dir / "latest.pt", map_location="cpu", weights_only=False)
            first_optimizer_steps = {
                int(state["step"].item()) for state in first["optimizer"]["state"].values()
            }

            summary = train_terrain_perception(
                dataset_dir=dataset_dir,
                output_dir=model_dir,
                validation_dataset_dir=None,
                sequence_steps=2,
                history_seconds=0.6,
                epochs=2,
                batch_size=4,
                learning_rate=1.0e-3,
                hidden_channels=2,
                device="cpu",
                seed=3,
                history_mode="no_odometry",
                resume_checkpoint=model_dir / "latest.pt",
            )
            resumed = torch.load(model_dir / "latest.pt", map_location="cpu", weights_only=False)
            resumed_optimizer_steps = {
                int(state["step"].item()) for state in resumed["optimizer"]["state"].values()
            }

            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(summary["start_epoch"], 2)
            self.assertEqual(len(summary["history"]), 1)
            self.assertEqual(Path(summary["resumed_from"]), (model_dir / "latest.pt").resolve())
            self.assertIsNotNone(summary["resume_best_validation_loss"])
            self.assertIn("best_validation_loss", resumed)
            self.assertGreater(min(resumed_optimizer_steps), min(first_optimizer_steps))

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

    def test_training_sampler_limits_compute_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = TerrainPerceptionChunkWriter(directory, chunk_steps=4)
            for step in range(12):
                writer.append(make_frame(step))
            writer.close()
            dataset = TerrainPerceptionSequenceDataset(
                directory,
                sequence_steps=2,
                history_seconds=0.6,
            )
            first = list(ChunkGroupedShuffleSampler(dataset, seed=17, num_samples=7))
            second = list(ChunkGroupedShuffleSampler(dataset, seed=17, num_samples=7))
            self.assertEqual(len(first), 7)
            self.assertEqual(len(set(first)), 7)
            self.assertEqual(first, second)
            with self.assertRaises(ValueError):
                ChunkGroupedShuffleSampler(dataset, seed=17, num_samples=len(dataset) + 1)

    def test_dataset_filters_sequence_endpoints_by_terrain_id(self):
        with tempfile.TemporaryDirectory() as directory:
            with TerrainPerceptionChunkWriter(directory, chunk_steps=4) as writer:
                for step in range(8):
                    writer.append(make_frame(step))
            dataset = TerrainPerceptionSequenceDataset(
                directory,
                sequence_steps=2,
                history_seconds=0.6,
            )

            terrain_one = dataset.sample_indices_for_terrain_ids((1,))

            self.assertTrue(terrain_one)
            self.assertTrue(all(int(dataset[index]["terrain_type"]) == 1 for index in terrain_one))
            self.assertEqual(len(terrain_one), len(dataset) // 2)
            with self.assertRaises(ValueError):
                dataset.sample_indices_for_terrain_ids(())

    def test_model_only_initialization_supports_independent_terrain_curriculum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("train", "validation"):
                with TerrainPerceptionChunkWriter(
                    root / name,
                    chunk_steps=8,
                    metadata={"terrain_component_names": ["flat", "slope"]},
                    odometry_free=True,
                ) as writer:
                    for step in range(8):
                        writer.append(make_odometry_free_frame(step))

            train_terrain_perception(
                dataset_dir=root / "train",
                output_dir=root / "source",
                validation_dataset_dir=root / "validation",
                sequence_steps=2,
                history_seconds=0.6,
                epochs=1,
                batch_size=4,
                learning_rate=1.0e-3,
                hidden_channels=2,
                device="cpu",
                seed=3,
                history_mode="no_odometry",
            )
            source = root / "source" / "best.pt"
            summary = train_terrain_perception(
                dataset_dir=root / "train",
                output_dir=root / "curriculum",
                validation_dataset_dir=root / "validation",
                sequence_steps=2,
                history_seconds=0.6,
                epochs=1,
                batch_size=4,
                learning_rate=1.0e-4,
                hidden_channels=2,
                device="cpu",
                seed=4,
                history_mode="no_odometry",
                init_checkpoint=source,
                train_terrain_ids=(1,),
                use_grid_coordinates=True,
                global_context_dim=3,
            )
            checkpoint = torch.load(
                root / "curriculum" / "best.pt",
                map_location="cpu",
                weights_only=False,
            )

            self.assertEqual(summary["start_epoch"], 1)
            self.assertEqual(summary["train_terrain_ids"], [1])
            self.assertEqual(Path(summary["initialized_from"]), source.resolve())
            self.assertEqual(checkpoint["config"]["train_terrain_ids"], (1,))
            self.assertTrue(checkpoint["config"]["use_grid_coordinates"])
            self.assertEqual(checkpoint["config"]["global_context_dim"], 3)
            self.assertEqual(checkpoint["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
