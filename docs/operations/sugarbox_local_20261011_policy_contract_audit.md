# Offline Policy Contract Audit

No robot motion, remote edits, GitHub fetches, or controller changes were made
for this audit. The diagnostic script does not initialize ROS/camera/motors.

## Artifacts Actually Used

- Model: C:/Users/user/OneDrive/Desktop/學校/igibson/doorway_ft_final.zip
- Model SHA256: af5e9e1d5ce0115de8b4bbb321873c6a99686c82c0c69d060f881bbe97b64f35
- Normalizer SHA256: 3cd4ef9ab5b0fa38058b5142b1cc5b5df1ba9c440f325ff81b88d5e6ca7a8cdf
- Model SB3 version: 2.2.1; saved num_timesteps: 400224.
- Model spaces: 55 observations, 2 actions. Normalizer: 55 observations,
  finite means/variances, strictly positive variances, norm_obs=True,
  clip_obs=100, stats_count=403481.0001.
- Runtime applies normalize_obs once, with training=False and norm_reward=False.
  It does not manually divide lidar ranges by four.
- Matching dimensions/statistics do NOT prove these artifacts are paired or
  prove the observation semantics match the original training environment.

## Confirmed Deployment Behavior

1. Lidar: 48 directions, -90 to +90 degrees, ascending (right to left under
   the controller's assumed frame). Each direction selects ONE nearest raw
   angular sample, not the minimum range in its angular sector.
2. Reproducible counterexample: a 2031-sample scan with a 0.30 m obstacle
   within +/-1 degree of body-forward produces all 4.0 m policy rays.
   This proves a sampling blind spot, NOT the cause of the physical collision.
3. Scan transform: body_angle = wrap(pi - raw_angle). It negates scan angular
   handedness as well as adding an offset. Physical left/right correctness
   cannot be established from source comments or a front/back-only test.
4. Policy range origin is the physical lidar sensor, without a translation
   to robot center. The lidar is reported at the chassis front edge. Whether
   training ranges originate at robot center requires the actual env source.
5. Client positive throttle conversion: min(a0 * 0.50, 0.15). Therefore a0
   values 0.3, 0.5, and 1.0 all become 0.15 m/s. Angular output is separately
   clipped at +/-0.78 rad/s. This changes curvature relative to the nominal
   action scales; it is not proportional slowdown of a commanded twist.
6. Current locally captured motor bridge further maps left=vx*200-wz*25 and
   right=vx*200+wz*25, scales both if peak exceeds 30, rounds to integers,
   and uses small command deadbands. This is not feedback velocity control.
   The observed motion need not equal the requested SI velocity.
7. MOTOR_DELAY_STEPS=0 and CONTROL_HZ=6.0. Their agreement with training
   cannot be verified without the actual environment used for this checkpoint.

## Synthetic Model Probe

Fixed goal=1m forward, zero feedback and previous action, other rays=4m:

| Input | Raw forward action | Raw angular action |
| --- | ---: | ---: |
| Clear | 0.949232 | 1.000000 |
| Front rays 19..28 at 0.12m | 0.720303 | 0.252111 |
| Right rays 0..15 at 0.12m | 1.000000 | 1.000000 |
| Left rays 32..47 at 0.12m | 1.000000 | -0.052765 |

These observations show input sensitivity, not reliable avoidance. Even the
front case maps to 0.15 m/s under current deployment clipping. These artificial
observations are not a full simulated rollout and may be out of distribution.

## Missing Evidence / Next Checks

The local training-config JSON was found, but not the actual environment
implementation that produced doorway_ft_final.zip. Local teammate deployment
documentation is not accepted as proof of that checkpoint's training contract.
Need the original env's observation builder, lidar generator, action execution,
control timing and checkpoint-normalizer save provenance before altering input
semantics or declaring retraining necessary. Physical frame/sign checks should
be stationary before another navigation trial. No navigation retest authorized
by this audit alone.

## Follow-up Local Archive Audit

Inspected the local July 29 handoff and August 23-28 progress notes, plus file
inventories inside navigation_exact_export_20260804_172242.zip and
igibson_teammate_handoff_core_20260803.zip. The navigation files identified
there are deployment/probe scripts, not the checkpoint's training environment.
The July notes say SB3 2.3.0, while the actual model stores 2.2.1. This is evidence
of stale or mixed documentation, not proof of a broken runtime or wrong model.

Added six characterization tests loading actual pure controller and locally
captured motor bridge functions through AST, without their hardware imports.
All 34 tests in the independent runtime's test directory passed in ig38.
They demonstrate sampling gaps, throttle clipping, 55-D raw observation layout,
and wheel saturation. Passing these tests does not validate obstacle avoidance.
For vx=0.15, wz=0.78, the current default mapping yields (6,6,30,30);
it does not establish that the real robot achieves 0.78 rad/s.

Throttle clipping does not by itself prove collision causality: lowering forward
speed while preserving yaw can also tighten a turn. Sector-minimum lidar would
likewise change observation semantics and must not be assumed to equal training.
No unverified fixes to lidar frame, ray origins, action gains, or odometry were
deployed.

## GitHub Training Source Located

Repository: https://github.com/koala915/igibson_x3_test
Inspected commit: 12f53c7d4e0c1ed06ac3a13def711e63aed01288

The deployment repository README links this separate training repository.
Downloaded artifact bytes were hashed in memory, not executed. All three
artifacts exactly match the local runtime's copies:

- nav_best_model/DOORWAY_FT_final_door1.00_chain0.90/final_model.zip:
  af5e9e1d5ce0115de8b4bbb321873c6a99686c82c0c69d060f881bbe97b64f35
- Same directory vecnormalize.pkl:
  3cd4ef9ab5b0fa38058b5142b1cc5b5df1ba9c440f325ff81b88d5e6ca7a8cdf
- Same directory training_config.json:
  e7b37f01d5451f6fe1e9171b0355d294338a076d85eb9760d4650e4b08338c92

This resolves artifact-pairing provenance, but does not establish that the
current main-branch environment is byte-identical to the original training run.
The handoff calls the checkpoint 398304; the matching saved model metadata
contains 400224. Artifact hashes, not filename step counts, identify the model.

### Source-Level Differences

training/navigation_env_room.py at the inspected commit:

- Lines 469-481: observation speed is sqrt(world_vx**2 + world_vy**2),
  nonnegative even when reversing. Runtime uses signed odom.vx and omits vy.
- Lines 506-527: 48 ideal rays originate on an outward radius around robot
  center; returns center-to-hit distance, min radius 0.33 m under the matched
  config, no-hit 4 m. Runtime uses physical front-mounted sensor range without
  translating the origin or enforcing the training value floor.
- Training also uses 48 discrete rays. The narrow-obstacle sampling example
  is a real limitation, but is NOT by itself proof of a deployment-only bug.
  Sector-min pooling would deliberately change the observation contract.
- Lines 306-344: two-step delay, clear-path stall assist, then near-obstacle
  linear/angular scaling, then velocity conversion. Matched config threshold
  is 0.66 m, linear scale interpolation starts at 0.24, angular at 0.9.
  Runtime PPO-only mode bypasses these mechanisms, delay is zero, and
  independently clips vx/wz before a non-feedback wheel mapping.

### Policy Limitations Documented by Training Authors

training/NAV_HANDOFF.md lines 149-167 explicitly reports DOORWAY_FT_final
failed the obstacle benchmark gate. Doorway success is not proof of general
obstacle avoidance; the record identifies collision and oscillation failure
modes and proposes obstacle-route fine-tuning. Treat this as documented
evaluation evidence, not a claim that every physical collision is explained.

### Consequence

The physical PPO-only test removed not only added deployment filters but also
behavior the training plant supplies. It does not isolate learned policy
quality. Next implementation should align the verified observation and action
contract before drawing a retraining verdict, without merging teammate mission
or grasp code, blindly flipping lidar axes, clipping away physical near-obstacle
evidence, or raising real robot speed to simulation maximums.

## Offline Contract Reference Implemented

Added diagnostics/training_contract_reference.py and optional
audit_policy_contract.py --training-contract. Neither connects to hardware;
the live approach controller and Jetson remain unchanged in this phase.

The reference reproduces unsigned planar speed, two-action delay, verified
stall-assist conditions, near-obstacle linear/angular scaling and SI action
scales. Explicit point-origin translations are included, without inventing
the actual target or lidar frame offsets. The observation floor is only for
offline comparison, not a faithful recreation of inner-radius ray geometry.
Raw physical safety distances must remain untouched.

Nine new tests plus the existing suite: 43 tests passed under ig38. The offline
comparison command ran successfully using the hash-matched local artifacts.
Proportional limiting is a separate candidate, not part of the original
training plant and not a calibrated motor controller. Example for raw [1,1]:
clear training twist [0.5,1.2], proportional limited [0.15,0.36], whereas
current independently clipped client twist is [0.15,0.78]. This shows different
commanded curvature; it does not prove the collision's physical cause.

Important: training stall assist can produce vx=0.14 even while the initial
delayed policy action is zero, if its clear-path conditions hold. It must not
be enabled blindly on hardware. Live wiring remains pending explicit sensor
and goal reference geometry and stationary frame/sign verification.

## User Mounting Clarification

The user confirms the lidar directions were already tested and reports body
center approximately 0.09m behind the lidar. Reuse that verified direction;
do not re-flip axes based on ambiguous historical comments. The offline
reference's default lidar origin is now x=+0.09m, y=0 (reported centered mount),
not the prior half-chassis-length inference of +0.16m.
Regression example: a valid 0.20m straight-ahead hit maps to center range 0.29m;
a side hit changes both range and angle, not range alone. This is geometry,
not obstacle inflation. No raw safety range or live controller was changed.
The vision front/base target origin still must be resolved independently;
the rear_cam_to_front_m=0.20 calibration is not the lidar-center offset.

## Goal Origin Confirmed and Diagnostic Wiring

User confirms Sugarbox geometry is front-referenced and center conversion
requires x += 0.135m. This is distinct from lidar x += 0.09m. For example,
front-referenced goal (0.8,0) becomes center goal (0.935,0), while a lidar
front hit of 0.8m becomes center range 0.89m. Offsets are not interchangeable.

The reference module now lives at integration/sugarbox_training_contract.py.
Opt-in SUGARBOX_TRAINING_CONTRACT=1 wires it into the independent controller:

- Vision geometry is converted once before latency compensation/target memory.
- Scan hit positions are translated to center and angular neighbors reselected;
  rear chassis returns remain excluded. This is an approximate observation
  adapter, not an exact recreation of simulator ray casting or occlusion.
- Odom vy is retained; policy speed uses hypot(vx,vy). Memory/latency motion
  propagation still models vx/wz only, so lateral drift remains a limitation.
- Training delay, assist and near-obstacle shaping are applied before separate
  candidate proportional command limiting. Original normal-mode behavior
  remains selected by default, including its 1.15 lidar input scale.
- Contract mode disables search/centering/blind-forward and rejects DRIVE
  before hardware initialization. It can only inspect sensors and compute
  commands in DRY_RUN. No physical test was performed in this implementation.
- Raw hard-stop comparison still uses physical ranges, not the policy floor.

48 offline tests passed, including distinct goal/lidar offsets, raw scan
preservation, rear-return exclusion, legacy defaults and early DRIVE rejection.
The offline audit command confirms offsets 0.09/0.135 and goal conversion 0.935m.
Live velocity mapping remains uncalibrated; do not equate these test passes
with validated obstacle avoidance or a ready-to-drive controller.

## User-Requested Physical Trial Enablement

The user explicitly requests a moving trial instead of another stationary
inspection. Contract mode can now use DRIVE only with the additional explicit
SUGARBOX_CONTRACT_ALLOW_DRIVE=1 flag. Default configuration is unchanged.
49 offline tests passed, including early rejection without authorization and
gate acceptance with authorization. Speed limits remain 0.15m/s forward,
0.08m/s reverse and 0.78rad/s yaw via proportional twist limiting; sensor
freshness, manual emergency stop and raw front hard stop remain active.
Search, final centering and blind-forward remain disabled for this trial.
Readiness confirmation is pending; no motor launch has occurred in this step.

## Contract DRIVE Trial and Physical Feedback

Log: diagnostics/drive_training_contract_20261011_143327.log.
The trial was explicitly authorized after readiness confirmation. The user also
authorized temporarily stopping grasp-service.service to release /dev/myserial.
The hash-matched existing motor bridge and bound /dev/rear_cam streamer were
started as transient services, with calibrated odometry unchanged. After the
agent ended the trial, feedback velocity was confirmed zero and the grasp
service remained stopped, as agreed.

The user reports no collision and successful obstacle bypass, with motion
heading in the correct direction. Record this as successful bypass with
approach incomplete, not overall failure and not final arrival. Logged target
distance advanced from 1.430m to 0.633m, with an intermediate oscillatory period.
The agent stopped the trial; raw hard stop did not end it.

This contract trial used distance scale 1.00, not the normal mode's 1.15.
Changing the legacy PPO_LIDAR_DISTANCE_SCALE constant would not modify the
contract adapter's rays. Among 57 logged control snapshots: 35 raw forward-axis
actions were negative, 22 positive, with 21 raw sign changes; 27 output snapshots
were forward and 27 reverse above 0.015m/s magnitude. These are sampled logs,
not every policy tick. They show policy-issued reversals, not recovery logic
forcing backups. Training two-step delay can additionally defer sign switches;
its contribution and the motor dynamics are not quantitatively isolated.

The actual center conversion is x=r*cos(theta)+0.09, y=r*sin(theta), followed
by hypot(x,y) and atan2(y,x), then center-frame angular neighbor selection.
It is not a constant 9cm range subtraction/addition on every direction.
No coefficient change was made following this report. A future 1.20 input scale
would enlarge apparent distances relative to this trial by 20%, potentially
reducing avoidance/reversal but also reducing clearance and affecting shaping.
It is an unvalidated hypothesis, not a demonstrated fix.

## Scale 1.05 Trial and Requested Revert

Log: diagnostics/drive_training_contract_scale105_20261011_144054.log.
The authorized physical trial used 1.05 only on the policy's distance input;
center-coordinate conversion, unscaled training shaping ranges, raw safety,
motor mapping and calibrated odometry were unchanged. The log ends normally
after distance-limit zero commands, and feedback velocity was confirmed zero.
The grasp service remains stopped; do not automatically restart it.

The user reports pressing stop and continued repeated oscillation during the
trial, comparable to the previous scale-1.00 run. They attribute the apparent
return of target distance toward 1m to YOLO misidentification. Treat that as
physical feedback; the distance trace is not proof of actual backward travel
or policy failure. The recorded endpoint of 0.389m is memory/odom-based, not
proof of target alignment, obstacle clearance or final arrival. Final centering
and blind-forward were disabled in this diagnostic mode. Collision status for
this trial was not explicitly reported.

At the user's request, contract mode's policy distance scale is restored to
1.00. Normal-mode scale 1.15 is unchanged. No new physical run is authorized
by this revert, and no model, target-memory rule or Jetson code is changed.
Raw policy reversals plus the documented failed obstacle benchmark make
retraining a reasonable next direction, not an isolated causal verdict:
target misidentification, inference/delay and physical actuator dynamics remain
confounders. Further coefficient changes are not a demonstrated remedy.

## Initial-Only Target Trial

The user requests one moving trial based only on the first valid YOLO/SAM
floor-position measurement and confirms readiness. Added opt-in
SUGARBOX_TARGET_INITIAL_ONLY=1, limited to training-contract diagnostic mode.
It locks the first valid measurement immediately (no three-frame average),
then ignores all later measurements, including visual freshness and metadata.
Odometry continues updating robot-relative x/y, distance and bearing; this is
not a frozen robot-relative coordinate. Only explicit C/reset of target memory
allows a new initial measurement. Default acquisition/reconfirmation behavior
is unchanged when the flag is off. The initial detection itself can still be
wrong, and odometry-only memory can drift, including the existing omission of
lateral velocity in memory propagation.

58 offline unit tests passed. This trial retains policy scale 1.00, the same
PPO model, calibrated odometry, lidar geometry, actuator mapping, delay/shaping,
raw front hard stop and manual stop. No final centering/blind-forward or search
is enabled; diagnostic center-distance cutoff remains 0.45m. Services were
verified active for RL motor and /dev/rear_cam; grasp service remains stopped.

Physical log: diagnostics/drive_initial_only_scale100_20261011_145119.log.
The trial locked its first valid measurement at center-relative
(1.331,-0.652)m, distance 1.483m, bearing -26.1deg. Subsequent detections were
marked TARGET_INITIAL_ONLY_ODOM and did not overwrite memory. After vision
lost the target (NO_YOLO), policy control continued on odometry-only memory.
Among 45 sampled CONTRACT_DRIVE snapshots, the raw forward-axis policy action
was positive 22 times and negative 23 times, with 14 nonzero sign switches.
These are sampled actions, not all policy ticks and not equal-duration A/B
statistics. Oscillation persists without later visual target corrections.

Raw front minimum reached 0.170m and triggered CONTRACT_RAW_HARD_STOP;
the next sampled minimum was 0.160m, with zero commanded twist. The memory
distance then read 0.640m, not final arrival. The process ended normally and
feedback velocity was independently confirmed zero. Collision/bypass status
has not yet been reported by the user. Do not infer successful clearance from
the stop. This rules out later visual overwrites as a necessary cause of
this trial's oscillation, but does not isolate policy training from a wrong
initial target, odometry drift, delay or physical actuator response.

User feedback after this trial: oscillation was comparable to the preceding
two runs; the robot approached the obstacle more closely and nearly collided.
No actual collision was reported. Treat this as unstable/unsuccessful avoidance,
not successful clearance because the raw hard stop intervened.

Confirmed control path: PPO_ONLY_DIAGNOSTIC is true in contract mode and bypasses
anti_jitter.apply, so neither its 0.30s hard-stop recovery wait nor its 6.0s
no-progress backup causes this trial's reverse commands. Active run snapshots
contain CONTRACT_DRIVE followed by CONTRACT_RAW_HARD_STOP, not recovery-start
or recovery-backup states. Startup still prints legacy recovery settings even
though that path is inactive, which can misleadingly suggest it is enabled.
The reverse commands originate from raw policy actions after the separate
two-control-step training delay, shaping and proportional limits. At 6Hz this
adds approximately 0.33s nominal delay, distinct from the inactive 0.30s recovery
wait. Its physical contribution to oscillation has not been isolated.
No further motion or controller change was requested by this feedback.

## Zero Added Action Delay Trial

The user requests removing the two-step artificial delay and confirms readiness
for another moving trial. Live contract mode now sets CONTRACT_ACTION_DELAY_STEPS
to 0 and passes it explicitly into TrainingActionReference. The helper supports
a validated nonnegative integer delay; its default remains 2 for reproducible
offline comparison with the training reference. In the live zero-delay branch,
each current raw action immediately reaches the existing shaping and proportional
limits. Reset does not reintroduce two initial zero ticks. No legacy action-delay
constant, policy observation, motor mapping or odometry calibration is changed.

63 offline tests passed, including current-action sign propagation, reset,
unchanged near-obstacle shaping/assist, invalid-delay rejection, live zero-delay
wiring and retained training-reference default. The moving trial retains
initial-only target locking, input scale 1.00, the same PPO and normalizer, rear
camera binding, speed limits, raw front stop and manual emergency stop. Recovery,
search, final centering and blind-forward remain inactive. Sensor and actuator
latency are not removed by disabling this artificial buffer.

Physical log: diagnostics/drive_initial_only_delay0_scale100_20261011_150301.log.
The first target locked at center-relative (1.314,-0.024)m, approximately
1.314m distance; the first sampled raw front minimum was 0.326m. This differs
from the preceding trial's starting target/range arrangement, so it is not a
controlled equal-duration comparison. All 31 active sampled snapshots show
delay=0; raw forward-axis action was positive 18 times and negative 13 times,
with 11 nonzero sign switches and no raw-action versus commanded-vx sign
mismatches. There were no sampled CONTRACT_RAW_HARD_STOP states. Target memory
distance reached approximately 0.713m and last read 0.748m, not final arrival.
The process exited normally and remote feedback velocity was confirmed zero.
User feedback: avoidance was much better and physical twitching was almost gone.
The remaining end-stage problem appeared when rear vision could no longer see
Sugarbox. Sampled policy sign changes must not be equated with physical twitching
or used to override this observation. This is improvement in this arrangement,
not proof of robustness across obstacle arrangements. No further trial started.

## Arm-Camera Handoff (Local Implementation, Not Deployed)

The user now wants rear-camera PPO avoidance, then visible arm-camera pixel
approach/centering, followed by automatic grasp. No target-search rotation in
place or blind-forward is requested. Their "line-follow" posture means TRAVEL:
S1..S6=(90,140,0,0,90,30), confirmed from the existing v23 source. Teammate
files were read as interface documentation; they were not modified or merged.

Added an opt-in stopped exit-42 handoff to the independent approach controller.
The default remains disabled; the existing zero-delay/initial-target/scale-1.00
PPO path, wheel mapping, LiDAR axes and calibrated Jetson odom stay unchanged.
The handoff requires near remembered goal, lost rear usable vision and existing
stable clearance so a camera loss during ongoing bypass does not start grasp.
Manual ESC/stop exits cannot trigger automatic grasp.

New sugarbox_arm_approach.py checks travel pose, arm RPC capability and camera
stream before starting PPO. Arm tilt happens only after the stopped handoff.
Pixel control uses bbox center-X and bottom-Y with a physically confirmed marker
and measured image-control signs; it uses no homography or world-coordinate
calculation. Missing/stale/jumping arm detections command zero, not blind travel.
Off-center correction uses visible arcs, never rotation in place. The near-stage
front stop is separately calibrated because the target itself can appear in the
scan; no live PPO safety threshold was relaxed.

The new local motor RPC uses the existing sole serial owner, changes only S2,
keeps the jaw untouched, and is disabled by default. It requires stationary
feedback, empty travel pose and unchanged/reached joint feedback. It has not been
deployed to Jetson. A separate confirmed profile is mandatory and remains unset.
No real arm movement, wheel movement or automatic grasp was sent in this work.

Read-only /dev/arm_cam streaming was verified on UDP 5602 alongside rear UDP
5600. Frame saved as diagnostics/arm_preview_20261011.png. The stream is camera
only and does not own /dev/myserial. Existing grasp services remain stopped.

Offline check_arm_observation_pose.py uses the existing URDF/controller mapper:
travel S2=140 -> optical depression 40deg; S2=137 -> 43deg. Empty downward path
minimum floor clearance ~214.2mm; minimum body clearance ~1.11mm, same nearest
pair as the travel endpoint. S2=143 -> 37deg and ~3.80mm model penetration into
the body, so it is explicitly refused in the new profile/RPC (137..140 only).
This conservative mesh check is not physical clearance or calibration proof.
Windows PyBullet required a relative URDF path due to the workspace's non-ASCII
absolute path. The diagnostic imports no serial driver and sends no hardware IO.

At a confirmed pixel-ready point, the wrapper returns S2 to travel, stops the RL
serial owner and arm stream, verifies release, and starts the existing v23 grasp
service. It requests the guarded travel->E1 path, then fresh E1 vision and grasp.
The existing grasp policy's geometry/envelope/floor/hold verification remains;
only near-stage chassis steering is pixel-only. No arbitrary direct jaw-close
command is introduced. Cancel/failure stops the grasp services without restarting
RL. This handoff is implemented but has not been exercised on real hardware.

86 offline tests passed, including arm RPC authorization, camera shape/release,
YOLO selector class-name wiring, stopped serial ownership and manual-exit refusal.
Pending: supervised S2=137 view confirmation, physically graspable pixel anchor,
measured yaw/forward image signs and target-compatible raw-front stop distance.
No commit, GitHub push or full autonomous physical trial was performed.

## Supervised Arm Initialization And View Checks

The user explains that they changed the arm pose, and that the unpowered arm
sags. Startup must initialize instead of assuming travel survived a power cycle.
Initial raw feedback was consistent but outside nominal API command limits:
S3 raw=3453 and S4 raw=3134. This does NOT prove servo failure. No movement was
sent until the user explicitly confirmed an empty gripper and clear arm space.

Added a separate opt-in arm_initialize RPC through the sole serial owner. It
commands ONLY the existing named nav pose over 2000ms (as grasp/move_arm.py),
with the jaw OPEN at 30deg; it is not a collision-validated path from arbitrary
sag. It requires server opt-in and explicit operator/empty-gripper/workspace-clear
confirmation, stationary chassis feedback, and actual post-motion readback. The
new real pipeline initializes once each run before PPO; failed initialization
blocks departure. No arbitrary pose or jaw-close RPC is accepted.

Added bounded feedback reads using array responses plus raw-ID-aware missing-joint
retries. Raw conversion is proven equal to the existing v23 implementation by
offline tests. No last-command substitution, nominal-angle clamping or simulated
feedback is used. Missing feedback reports affected joints and raw replies.

The original RL motor file and feedback_odom.py were not overwritten. A separate
sugarbox_arm_observation_server.py and sugarbox_arm_feedback.py were uploaded;
temporary calibration units on TCP7001 exclusively owned /dev/myserial while
the original TCP7000 motor service was stopped. ROS was not reloaded or rebooted.
The original transient motor unit is garbage-collected when stopped; restoration
uses its exact original ExecStart and Restart settings, with no daemon-reload.

After the user's readiness approval, physical initialization read back exactly
(90,140,0,0,90,30). S2=137 also read back exactly and a live view was shown. The
user reported this view is insufficient and requests looking farther downward.
The preview returned S2=140 successfully before the next adjustment.

Offline model check for S2=140,137,134,131,130 showed depression 40,43,46,49,50deg,
minimum floor clearance at 130 ~204.76mm and path body clearance ~1.11mm (same
nearest travel-endpoint pair). Based on this check, the supervised downward
range is now 130..140, still no more than 3deg per acknowledged movement step.
The client splits the movement and return into verified small steps. No upward
S2=143 move is allowed. These model results are not physical collision proof.

Physical S2=130 read back exactly; live preview shown and first image saved to
diagnostics/arm_s2_130_20261011.png. A box at the lower frame edge was clipped,
so this is NOT a confirmed grasp-ready marker. The physical profile remains
unset/unconfirmed, and no wheel velocity or automatic grasp command was sent.
96 offline tests pass, including initialization authorization/failure, real
readback, partial/misdirected responses and max-3deg movement/return splitting.
View suitability, reachable pixel point, control signs and final front distance
remain unconfirmed. No commit or GitHub push was performed.

S2=130 preview cleanup did not fully return to travel: integer feedback changed
by 1deg between the two server reads, so the strict max-3deg check correctly
refused a requested final step. Readback afterward was S2=137, not 140. Do not
label that attempt a successful restore. The client now plans 2deg steps with
1deg settling margin, while the server's 3deg physical maximum is unchanged.
A focused regression test reproduces the 1deg settling case; 97 tests pass.

The user requests simultaneous rear and arm views to locate a handoff boundary.
Added diagnostics/preview_both_cameras.py, which opens UDP5600 and UDP5602 in
separate side-by-side windows and never connects to a motor. It saves an initial
fresh pair and additional pairs on C; timestamps are receiver times, not hardware
capture synchronization. S2 was safely stepped back to 130 and physically read
back as (90,130,0,0,90,30) while both camera previews remained active. The first
saved pair predates this adjustment, so its arm pose must NOT be called S2=130.
Any usable boundary/grasp anchor still needs explicit user confirmation.

The user confirms the S2=130 view is good for handoff: the rear camera's lower
visible region connects to the arm camera's upper region. They note structural
occlusion in the lower center of the rear image and propose travel-view handoff
followed by progressively downward viewing while approaching. This is a design
discussion, not authorization to execute a new chassis/grasp trial. No dynamic
pitch control or additional viewing angles have been enabled. Camera pitch
changes move the bbox without chassis progress, so a single fixed-pose pixel
target cannot be reused blindly across that transition.

The dual preview ended normally. Only the initial paired images were captured;
no C-key boundary pair was recorded. Using the corrected 2deg planned steps,
physical return to travel was verified twice as (90,140,0,0,90,30). Temporary
arm-view service was stopped, serial release verified, original RL motor service
restored active (PID8234). Existing grasp services remain inactive. Original
motor SHA256 a9fcff9d1671a9172879071f308ec26508eb03146a80586ab5b9f5763057269b
and feedback odom SHA256 59160f3b15ceaa132f8e460e546930c27e8c7f96726b83b5b3d8ea0f24d3ea38
match pre-checks. No wheel velocity or autonomous grasp command was sent.

## Staged Pixel Approach Implementation

The user authorizes the proposed staged approach implementation. New opt-in
rear-edge handoff can end the PPO leg before visual loss, but still requires
target memory near the goal and the existing bypass-clearance gate. Its pixel
boundary is an explicitly calibrated profile field, not a guessed live constant.
The arm stage confirms consistent fresh images before any chassis translation.
Default standalone PPO remains unchanged because ARM_HANDOFF is disabled.

Added sugarbox-arm-staged-pixels-v2 profile support alongside legacy v1. The
first stage is the physically viewed S2=130 handoff view. Later stages each
have their own confirmed pixel goal, image-control signs and front-stop setting.
They look down monotonically in at most 2deg increments; only the final pose
can authorize grasp. Intermediate "ready" requests a stopped pose change, not
arrival or grasp. After actual S2 readback, old detection history is discarded,
in-flight images are ignored for 0.35s, and fresh consistent frames are required.
Motion is based on pixel progress, never elapsed-time blind forward/tilting.

Offline URDF sweep includes 130,128,126,124,122,120: depression 50..60deg,
minimum floor clearance at 120 ~186.45mm, nearest path body pair ~1.11mm as at
travel. The local planned pose range is consequently 120..140, still max 3deg
actual server step and 2deg nominal client steps. Lower poses are MODEL-CHECKED
ONLY, not physically calibrated, and no new pose/motor/grasp command was sent.
The live original motor bridge remains the original version; the expanded local
RPC server was not deployed or started during this implementation.

The camera preview/ESC is checked between bounded individual pose requests and
again before grasp-service startup. Cancel stops further pose requests and
prevents grasp; it does not claim to interrupt the single already-sent servo
trajectory or a blocked socket read instantly. Chassis remains stopped during
all pose changes. Restore-to-travel uses the same cancellable steps.

The six-stage template (130..120) deliberately contains null pixel/sign/range
calibration and confirmed=false. It is not a runnable physical preset, and the
lowest required view/final grasp marker may change after physical calibration.
Unconfirmed profiles fail before hardware startup. No homography/world-coordinate
conversion was added to the near-stage chassis control; the existing guarded
v23 grasp still owns its own E1 geometry and hold checks.

115 offline tests cover stage ordering, fresh-frame reacquisition, different
pixel goals per pose, intermediate vs final ready, early rear-edge/clearance
handoff, runtime ESC/pose failures, and cancellation during pose steps. Python
syntax checks pass. No full real trial, GitHub push, commit or odom change.

## Mouse Handoff Calibration

Added diagnostics/calibrate_camera_handoff.py and pure draft helper. The two
bound cameras show fixed 640x480 views; mouse clicks require fresh frames and
actual six-servo feedback at S2=130. S saves only an unconfirmed draft plus
the selected original images and pose/timestamp evidence. The rear mark sets
the draft rear_handoff_v; the arm mark is only an overlap reference, never a
stage-switch or final grasp goal. Existing confirmed profiles cannot be reused
as draft inputs, and existing evidence files cannot be overwritten.

120 offline tests pass. The user explicitly confirmed empty gripper, clear arm
workspace and physical presence for this stationary calibration. Original motor
service was stopped and /dev/myserial verified free. A separate temporary
codex-sugarbox-handoff-calibration unit runs the previously uploaded observation
server on port 7001; the original motor/odom files were not modified. Actual
initialization feedback was (90,140,0,0,90,30); actual handoff feedback was
(90,130,0,0,90,30). No nonzero chassis velocity or grasp command was sent.
The GUI is awaiting operator marks; completion and service restoration must
be logged below after the window exits. ESC cancels further arm requests and
does not automatically move the arm back; a normal exit returns to travel via
the same verified bounded steps. The 120..128 views remain untested physically.

First GUI attempt was cancelled after a rear click; no draft was saved and
the arm was left at 130 without automatic movement. User reported accidental
cancellation and requested retry. The retry read actual S2=130, with no repeat
initialization. Operator saved handoff_draft_20261011_01.json plus original
images and evidence: rear=(416,470), arm=(421,85), both actual poses
(90,130,0,0,90,30). Receiver timestamps differ by ~6.83 seconds; these are
separate selected frames, not synchronized camera capture.

Visual inspection shows the marks correspond to a small blue box at the
camera overlap, not the yellow Sugarbox still farther away in the rear view.
This is overlap-reference evidence only. Stage pixel targets, signs and final
grasp marker remain null/unconfirmed; no motion profile was authorized.

After the normal retry save, a separate verified 2deg-step return read back
(90,140,0,0,90,30). Temporary calibration unit was stopped, serial release
verified, original motor unit restored active (PID26854). Rear and arm streams
remain active; grasp-service and grasp-vision remain inactive. Original motor
and odom SHA256 hashes still match the pre-checks above. No chassis motion,
autonomous grasp, original motor/odom edit, commit or GitHub push was performed.

## Two-View Bottom Switch And OEM Pose Correction

User replaces six small angle-specific approach stages with travel-view bottom
trigger then a 20deg downward change. Added opt-in schema
sugarbox-arm-bottom-switch-v3: travel S2=140 -> down20 S2=120. Initial driving
setpoint is beyond the switch line, not a grasp marker. Arm bottom_switch_v
defaults to 432/480 in the draft (configurable 360..450); three distinct stable,
fresh confident boxes crossing it request a stopped pose change even off-center.
Bottom clipping may request a view change, never grasp. Sensor holds reset edge
evidence. After actual pose confirmation, old images are discarded and fresh
post-settle detections required. Only the final pixel goal authorizes grasp.
The old v1/v2 paths and standalone PPO default are unchanged.

Named empty-arm RPC views are separate from the existing max-3deg calibration
RPC. They require stationary chassis, actual expected/known pose, open gripper,
server opt-in and reached feedback. A view change commands S2 only, 2000ms, not
six individual calibration goals. Motion during driving remains DISABLED;
the serial bridge stops the chassis while reading/moving/verifying the arm.
The real v3 pipeline remains unconfigured/unconfirmed and was NOT run.

Initially an offline collision-free horizontal candidate (90,120,60,0,90,30)
was tested under explicit operator empty-arm/clear-workspace supervision.
Physical feedback was (90,120,59,0,90,30), captured in forward_view_20261011_01.
It looked forward but raised the gripper into the rear-camera view and increased
Sugarbox occlusion. This candidate was REJECTED, not marked usable. Normal GUI
cleanup returned actual (90,140,0,0,90,30), and the temporary service was stopped.

User clarified: preserve line-follow pose and change only the lower joint, using
Rosmaster's anti-collision pose. Read-only inspection of the original Jetson
/home/jetson/Rosmaster/rosmaster/rosmaster_main.py lines 630..648 confirms:
cmd '43', pose 1 '# anti-collision' sends [90,180-0,180-180,180-180,90,30];
pose 3 '# line-follow' sends [90,180-40,180-180,180-180,90,30]. SDK endpoints are
therefore forward=(90,180,0,0,90,30), travel=(90,140,0,0,90,30), with S2 the only
different joint. App/UI S2..S4 angles are reversed by 180-angle; they must not
be mistaken for SDK values. The wrong shoulder/elbow candidate was removed
from local code and the new, separate remote named-view file was updated.

The existing URDF/mapper reports ~45.77mm convex-hull body penetration along
the S2-only travel->180 path. OEM source identification does NOT reconcile this
with the actual robot/attachments. No S2=180 hardware command was sent. Forward
automatic use requires explicit physical path validation as well as server
opt-in; the preview rejects before hardware without that validation. Profile
forward_view_confirmed remains false. Next work must validate the OEM S2-only
path, not reuse the rejected dual-joint candidate or claim model safety.

138 offline tests and syntax checks pass, including S2-only writes, OEM-path
validation gates, bottom switching without centering, duplicate/stale/jumping
boxes, post-tilt freshness, runtime cancel/failure and final-only grasp.
Original motor service restored active PID28661; both camera streams active;
grasp-service/vision inactive. Original motor/odom hashes remain unchanged.
No nonzero chassis command, automatic grasp, commit or push in this change.

## Supervised OEM Path Trial Before Full Run

User authorizes trying the full flow. The last grasp-ready pixel point/signs
remain unset, so no profile was fabricated and no full PPO/approach/grasp run
was started. A separate --allow-oem-validation RPC enables supervised empty-arm
S2-only exploration, NOT automatic path approval. Requests require matching real
pose, stationary chassis, other five joints at travel, 140..180 target, nominal
2deg client steps/max 3deg physical step (integer feedback settling margin),
500ms trajectory and reached readback. Cancel/failure prevents further steps.
The final OEM pose alone never flips path_validated to true.

First preview aborted before any arm command because both camera decoders did
not become fresh within the initial 8sec startup window. Read-only dual preview
then received a fresh pair; PC IP, bound camera names and original stream services
were unchanged. No old receiver process was found and no camera, desktop app or
computer was restarted. Increased startup allowance to 20sec without relaxing
the <=0.35sec frame-age requirement; frame availability remains required during
path movement. The independent temporary OEM service was restarted while idle.

Second supervised attempt reached actual (90,180,0,0,90,30), with per-step actual
feedback stored in diagnostics/oem_forward_20261011_01_initial.json. The real
rear image has no arm occlusion; the arm image looks forward. This corroborates
the OEM pose rather than the rejected shoulder/elbow candidate. The URDF conflict
was NOT silently reclassified as safe: final evidence retains oem_path_checked
and occlusion_confirmed=false until operator confirmation. User was asked to
confirm no body/cable contact and press S. No S arrived before timeout, so no
return motion was issued and no approval was fabricated. A separate async user
question remains pending. Subsequent status again read exact S2=180.

Fixed a full-grasp ownership-transfer bug: a stopped transient systemd motor
unit may disappear, yielding empty ActiveState. It may now pass only alongside
an explicit empty-device check (existing serial path, fuser exit 1, empty output
and error). Busy/unknown device ownership still refuses grasp startup. This was
an offline fix; actual grasp service was never started in this trial.

143 offline tests pass. Temporary OEM service stopped; serial release verified;
original motor unit restored active PID19405. Both original camera streams remain
active; grasp-service/vision inactive. Original motor and feedback odom hashes
still match the earlier checks. Arm remains at OEM forward 180, not travel 140.
No chassis movement, jaw closure, commit or push. Required remaining work:
operator OEM-path confirmation, final physically reachable pixel point/control
signs, then complete supervised PPO -> travel-view -> down20 -> grasp trial.

## Read-only Recheck And Travel / Down20 Stationary Trial

Read-only recheck obtained exact (90,180,0,0,90,30). Both bound streams were
fresh; diagnostics/oem_recheck_20261011_02_* images show the rear view free of
arm occlusion. No joint/chassis motion or jaw closure was requested. Original
motor service was restored and grasp services remained inactive.

The operator subsequently explicitly confirmed no body/cable contact, an empty
gripper, clear workspace and presence beside the robot, and authorized the
stationary S2-only sequence 180 -> 140 -> 120. Added a separate diagnostic,
check_arm_two_views.py, not a change to PPO or default deployment. It requires
fresh dual frames before movement, verifies actual initial OEM pose, uses the
existing sole-owner named-view RPC and verifies feedback after each move. No
initialization, nonzero chassis command, jaw-close or automatic return on exit.

First attempt received cancellation before camera setup and before any pose
request. Readback remained exact 180. Original motor service was restored. After
operator approval to retry, the second attempt reached exact travel
(90,140,0,0,90,30) and down20 (90,120,0,0,90,30). Evidence is stored in
diagnostics/two_view_20261011_02_{travel,down20}.json and the associated paired
images. Other five joint feedback values remained unchanged. Post-motion frames
were required; the down20 pair's receiver timestamps differ by about 0.036sec,
which is receiver timing, not hardware camera synchronization.

IMPORTANT target clarification: the user explicitly identified the BLUE SMALL
BOX as the intended target, NOT the yellow item. Earlier blue-box overlap
evidence therefore shows the intended object; it is still not a final reachable
grasp-point calibration. At S2=120 the blue box and floor contact are visible.
No detected pixel point was automatically promoted to a reachable grasp goal.
Final pixel setpoint/control signs/front-stop calibration remain unset, and
confirmed_grasp_point=false in both new view records. No full PPO/grasp run.

148 offline tests pass (143 existing, 5 new). The initial sandbox test attempt
could not resolve OneDrive paths; an escalated offline run exposed a tuple-vs-
JSON-array test fixture mismatch, corrected without weakening feedback checks.
Successful supervised run ended normally, leaving S2=120 and the jaw open.
No odom/PPO changes, teammate merge, commit or push.

## Publication Addendum

The operator subsequently asked to pause for a meal. The original motor,
temporary pose-check and grasp service/vision units were confirmed inactive;
no automatic resume, joint return or jaw closure was requested. On resuming,
SSH to 10.29.101.252 and the configured 192.168.0.201 both timed out. Windows
was on 192.168.0.103. No additional hardware test occurred; real feedback and
network connectivity must be checked again before motion.

The operator then explicitly requested publication of today's work. This
record and an isolated code/test/tool snapshot are published on
codex/sugarbox-local-20261011, based on the locally available v23-grasp-test
commit 3eb79ec. Earlier 'no commit or push' statements describe those earlier
trials, not this later publication. Production/team files and unrelated dirty
checkouts are left unchanged. Images, raw runtime logs and generated calibration
evidence remain local; evidence paths in this audit are not shipped assets.
Profile templates remain unconfirmed. Publication is not deployment or an
end-to-end hardware acceptance test.
