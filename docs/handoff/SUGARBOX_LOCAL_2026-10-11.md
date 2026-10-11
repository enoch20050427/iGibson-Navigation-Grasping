# Independent Sugarbox approach snapshot: 2026-10-11

Owner: enoch20050427. Publication branch: `codex/sugarbox-local-20261011`.
Base: locally available `v23-grasp-test` commit `3eb79ec`.

## Scope

This is a reviewable snapshot of the operator's independent local development
from `.codex-runs/cc7df8`, originally based on `cc7df8d1`.
It does NOT replace the team's top-level integration files, configure startup
services, deploy to Jetson, or integrate the team's mission pipeline.
No earlier dirty checkout was included or reset. No new model was trained.

Source, standalone tests, diagnostic tools and unconfirmed profile templates:
`integration/sugarbox_local/20261011/`.
Full chronological record:
[Policy / hardware audit](../operations/sugarbox_local_20261011_policy_contract_audit.md).

All seven snapshot `integration/*.py` files retain their exact local contents.
Two publication-only reference paths in a test and an offline FK diagnostic
now locate this checkout's existing v23 files rather than a sibling checkout.

## Today's work

- PPO training-contract adapter: translate LiDAR hits into chassis-center XY
  before recomputing ray angles/ranges. LiDAR offset is 0.09 m; target origin
  conversion uses the separate 0.135 m forward offset.
- Tested scale 1.05, then returned to 1.0. The improved trial used zero added
  action delay and a frozen first target acquisition. PPO was not retrained.
- One supervised trial with no added action delay had markedly less
  forward/reverse oscillation. This is operator-reported improvement, not a
  repeatability or collision-free certification.
- Opt-in rear-camera handoff, arm pixel servoing and existing guarded v23 grasp
  orchestration. Standalone PPO behavior is not automatically replaced by it.
- Correct OEM S2-only views: forward `(90,180,0,0,90,30)`, travel/line-follow
  `(90,140,0,0,90,30)`, down20 `(90,120,0,0,90,30)`.
- Actual readback reached all three poses. Forward removed rear-view arm
  occlusion; travel -> down20 changed only S2. The intended target was explicitly
  confirmed by the operator as the BLUE SMALL BOX, not the yellow item.
- Hardened actual servo feedback and sole serial ownership. A vanished transient
  motor unit is accepted for ownership transfer only after an explicit empty
  serial-device check. No real grasp was triggered by this change.

## Not finished

Final reachable pixel goal, image-control signs and front-stop calibration are
still unset. All profile templates retain `confirmed=false`; these templates
must NOT be used to claim a calibrated grasp-ready location.
The complete PPO -> travel-camera -> down20-camera -> automatic grasp sequence
has NOT been run. The OEM pose's URDF collision-model discrepancy remains
documented; a successful supervised pose trial is not a model-safety proof.
Arm motion while driving remains disabled: the current RPC stops the chassis.

The operator paused the session; control services were stopped and the arm was
left at S2=120 with an open jaw. On resuming, both known Jetson addresses timed
out while Windows was on a different subnet. No new hardware action occurred.
Recheck real feedback and connectivity rather than trusting this recorded pose.

## Offline checks

From the repository root, with the existing ig38 development dependencies:

```powershell
python -m unittest discover -s integration/sugarbox_local/20261011/tests -q
```

The independent suite has 148 tests. It uses fake hardware and AST extraction;
passing it does not authorize physical motion or confirm navigation/grasp.

Publication checks on this checkout: independent 148/148; existing v21
controller 138/138, servo read 37/37, floor guard 641/641; UI 47 passed and one
external-route test skipped (route package absent); mission FSM selftest passed.
The initial v21 controller/floor runs failed to load an existing URDF through
its Chinese absolute path; a runtime-only relative-path loader resolved this.
Controller output also needed `python -X utf8` for the Windows terminal.
No production test/controller file was edited to obtain these results.

Models, raw runtime logs, camera photographs and generated evidence JSON remain
local and are not part of this public snapshot. Historical evidence filenames
in the audit refer to those local files. Existing absolute Windows model paths
were intentionally preserved for reproducibility, NOT made portable. Any new
deployment needs the operator's model / VecNormalize pair, YOLO, SAM2 and rear
homography assets and the documented Jetson dependencies (`feedback_odom.py`,
ROS and Rosmaster_Lib). Do not substitute another checkpoint by shape alone.

Review the `*_no_amplification.py` candidate separately from the preserved
`sugarbox_rl_motor_server.py`. Neither is installed or started by this commit.
Deployment and eventual integration need a separate review against
`v23-grasp-test`; this snapshot does not merge into it.
