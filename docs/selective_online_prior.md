# Expert-anchored selective online prior

This optional `fb_depth` training mode replaces the assumption that every
policy state is a discriminator negative. It does not use a plane collector
and it does not add stair demonstrations to the expert set.

Enable it with:

```bash
python -m humanoidverse.train \
  --agent fb_depth \
  --selective-prior \
  --prior-plane-envs 0
```

## Data contract

- Fixed mocap expert frames are positive (`+1`).
- Policy windows accepted by the slow, D-independent verifier are positive
  (`+1`) with a lower loss weight.
- Only high-confidence pathological policy windows are negative (`-1`).
- `UNKNOWN` policy frames do not enter D, Q_D, or Actor-D.
- FB, Q_Aux, and Q_H continue to use all main-terrain replay.
- The main branch retains atomic BehaviorContext relabeling. The prior branch
  always uses collection-time `z`, heading, source, motion, and context IDs.

The initial verifier accepts only exact-tracking contexts. It uses a frozen
target-B snapshot, sustained semantic consistency, heading consistency, and
conservative physical-pathology checks. Random and untraceable goal contexts
remain `UNKNOWN`.

## Staged optimization

The prior state machine is reversible:

1. `BOOTSTRAP`: collect/revalidate labels; D, Q_D, and Actor-D are off.
2. `FIT_D`: fit expert/validated/bad LSGAN; Q_D and Actor-D are off.
3. `FIT_QD`: freeze a calibrated D reward snapshot and fit Q_D; Actor-D is off.
4. `ACTOR_PRIOR`: enable the weak Actor-D term only on K-step finalized
   interior samples with low ensemble uncertainty.

Support changes or degraded calibration close the Actor-D loop and return to
the appropriate fitting phase. Q_D is reset whenever the frozen D reward
snapshot changes.

## UNKNOWN and Bellman semantics

`UNKNOWN` means that the prior has no opinion. A finalized-to-UNKNOWN
transition is omitted from Q_D loss; it is not converted into an artificial
zero-value terminal. Real terminated/truncated transitions may use a terminal
target. Nonterminal Q_D samples require a finalized successor under the same
`context_id` and use the stored collection-time `z_next`.

## Replay and resume invariants

Admission metadata is stored in the main compact-depth trajectory ring. Masked
sampling reuses the existing depth-history reconstruction and does not copy
observations. Every delayed write checks the slot generation. Validated/bad
sampling is balanced over motion IDs shared by both strata.

Checkpoint state includes prior phase, bank/teacher/D/Q_D versions, the frozen
gate teacher, and the frozen D reward snapshot. Missing or incompatible replay
metadata fails closed to `BOOTSTRAP`; it never silently restores Actor-D.
