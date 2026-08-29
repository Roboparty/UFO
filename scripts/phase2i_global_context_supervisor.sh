#!/usr/bin/env bash
set -euo pipefail

root=/data/xue/UFO/evaluations/phase2i_v2_no_odometry_20260827
report_dir="$root/reports"
log_dir="$root/logs"
mkdir -p "$report_dir" "$log_dir"

tags=(
  g32_full_lr1e4
  g64_full_lr1e4
  g128_full_lr1e4
  g64_full_lr3e4
  g64_stairs_lr1e4
  g128_stairs_lr1e4
  g64_stairplat_lr1e4
  g64_geometry_lr1e4
)

while [[ $(find "$root" -maxdepth 2 -path "$root/global32_*/summary.json" | wc -l) -lt ${#tags[@]} ]]; do
  printf '[%s] waiting for global-context training\n' "$(date '+%F %T')"
  sleep 30
done

for tag in "${tags[@]}"; do
  test -s "$root/global32_$tag/best.pt"
  test -s "$root/global32_$tag/latest.pt"
  test -s "$root/global32_$tag/summary.json"
done

evaluate_tag() {
  local gpu=$1
  local tag=$2
  local checkpoint_kind mode terrain dataset output
  for checkpoint_kind in best latest; do
    for mode in clean dr; do
      for terrain in stairs_up stairs_down mixed; do
        dataset="$root/holdout_${mode}_${terrain}_16cm"
        output="$report_dir/global32_${tag}_${checkpoint_kind}_${mode}_${terrain}.json"
        CUDA_VISIBLE_DEVICES="$gpu" /home/xue/UFO/.venv/bin/python \
          /home/xue/UFO/humanoidverse/evaluate_terrain_perception.py \
          --checkpoint "$root/global32_$tag/$checkpoint_kind.pt" \
          --dataset-dir "$dataset" \
          --output "$output" \
          --batch-size 512 \
          --device cuda:0 \
          --num-workers 4 >/dev/null
      done
    done
  done
}

pids=()
for gpu in "${!tags[@]}"; do
  tag=${tags[$gpu]}
  evaluate_tag "$gpu" "$tag" >"$log_dir/eval_global32_$tag.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

/home/xue/UFO/.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path('/data/xue/UFO/evaluations/phase2i_v2_no_odometry_20260827')
tags = [
    'g32_full_lr1e4',
    'g64_full_lr1e4',
    'g128_full_lr1e4',
    'g64_full_lr3e4',
    'g64_stairs_lr1e4',
    'g128_stairs_lr1e4',
    'g64_stairplat_lr1e4',
    'g64_geometry_lr1e4',
]
holdouts = [
    ('clean', 'stairs_up'),
    ('clean', 'stairs_down'),
    ('clean', 'mixed'),
    ('dr', 'stairs_up'),
    ('dr', 'stairs_down'),
    ('dr', 'mixed'),
]
branches = []
for tag in tags:
    summary = json.loads((root / f'global32_{tag}' / 'summary.json').read_text())
    for checkpoint_kind in ('best', 'latest'):
        metrics = {}
        ordinary_terrain_deltas_cm = {}
        checkpoint_epoch = None
        for mode, terrain in holdouts:
            path = root / 'reports' / f'global32_{tag}_{checkpoint_kind}_{mode}_{terrain}.json'
            report = json.loads(path.read_text())
            checkpoint_epoch = int(report['checkpoint_epoch'])
            values = report['metrics']
            baseline_path = root / 'reports' / f'v2loss32_s7907_{mode}_{terrain}.json'
            baseline_values = json.loads(baseline_path.read_text())['metrics']
            metrics[f'{mode}_{terrain}'] = {
                'edge_mae_cm': 100.0 * values['predicted_stairs_edge_mae'],
                'underfoot_mae_cm': 100.0 * values['predicted_stairs_underfoot_mae'],
                'edge_never_observed_mae_cm': 100.0 * values['predicted_stairs_edge_never_observed_mae'],
                'overall_mae_cm': 100.0 * values['predicted_mae'],
                'baseline_overall_mae_cm': 100.0 * baseline_values['predicted_mae'],
                'overall_regression_cm': 100.0 * (
                    values['predicted_mae'] - baseline_values['predicted_mae']
                ),
            }
            if terrain == 'mixed':
                for ordinary in ('flat', 'slope', 'rough', 'platforms'):
                    key = f'predicted_terrain_{ordinary}_mae'
                    delta = 100.0 * (values[key] - baseline_values[key])
                    ordinary_terrain_deltas_cm[f'{mode}_{ordinary}'] = {
                        'mae_cm': 100.0 * values[key],
                        'baseline_mae_cm': 100.0 * baseline_values[key],
                        'regression_cm': delta,
                    }
        worst_edge = max(value['edge_mae_cm'] for value in metrics.values())
        worst_underfoot = max(value['underfoot_mae_cm'] for value in metrics.values())
        worst_overall_regression = max(value['overall_regression_cm'] for value in metrics.values())
        worst_ordinary_regression = max(
            value['regression_cm'] for value in ordinary_terrain_deltas_cm.values()
        )
        dr_edge_regressions = {
            terrain: metrics[f'dr_{terrain}']['edge_mae_cm'] - metrics[f'clean_{terrain}']['edge_mae_cm']
            for terrain in ('stairs_up', 'stairs_down', 'mixed')
        }
        worst_dr_edge_regression = max(dr_edge_regressions.values())
        passes_accuracy = worst_edge <= 3.0 and worst_underfoot <= 3.0
        passes_regression = worst_overall_regression <= 0.2 and worst_ordinary_regression <= 0.2
        passes_dr_stability = worst_dr_edge_regression <= 0.5
        branches.append({
            'tag': tag,
            'checkpoint_kind': checkpoint_kind,
            'checkpoint': str(root / f'global32_{tag}' / f'{checkpoint_kind}.pt'),
            'checkpoint_epoch': checkpoint_epoch,
            'best_validation_loss': summary['best_validation_loss'],
            'metrics': metrics,
            'ordinary_terrain_deltas_cm': ordinary_terrain_deltas_cm,
            'dr_edge_regressions_cm': dr_edge_regressions,
            'worst_edge_mae_cm': worst_edge,
            'worst_underfoot_mae_cm': worst_underfoot,
            'worst_overall_regression_cm': worst_overall_regression,
            'worst_ordinary_regression_cm': worst_ordinary_regression,
            'worst_dr_edge_regression_cm': worst_dr_edge_regression,
            'passes_accuracy_gate': passes_accuracy,
            'passes_regression_gate': passes_regression,
            'passes_dr_stability_gate': passes_dr_stability,
            'passes_estimator_gate': passes_accuracy and passes_regression and passes_dr_stability,
        })

branches.sort(
    key=lambda item: (
        not item['passes_estimator_gate'],
        item['worst_edge_mae_cm'],
        item['worst_underfoot_mae_cm'],
    )
)
payload = {
    'external_input_contract': 'partial_maps+visible_masks+timestamps+frame_valid+64D_proprio; no odometry',
    'metric_source': 'predicted_clearance (network output, without noisy current-visible bypass)',
    'edge_gate_cm': 3.0,
    'underfoot_gate_cm': 3.0,
    'maximum_material_regression_cm': 0.2,
    'maximum_dr_edge_regression_cm': 0.5,
    'regression_baseline': 'convgru32 Phase-2I v2 seed7907 on identical holdouts',
    'branches': branches,
    'selected': branches[0]['checkpoint'] if branches[0]['passes_estimator_gate'] else None,
    'phase2j_allowed': any(branch['passes_estimator_gate'] for branch in branches),
}
output = root / 'reports' / 'global32_context_gate_report.json'
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, indent=2, sort_keys=True))
PY

printf '[%s] global-context holdouts and gate report complete\n' "$(date '+%F %T')"
