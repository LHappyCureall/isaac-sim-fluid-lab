# Upstream coarse-speed-only 40 mL calibration

Only coarse dosing speed varies: baseline 25.6 degrees/s and candidates 25.2/24.8 degrees/s. Both roles keep first approach 87 degrees, retry 87.2 degrees, conditional retry enabled and at most two cycles. All other coarse/micro speeds, holds, geometry, particles and solver parameters are fixed. See [the actual selected results](../../results/coarse40_20261128.json).

Six seen calibration cases include two conditions for each profile. A baseline may fail dose, spill, home or late-water task gates and remain a valid negative comparator only when its physical, mass and measurement evidence is valid. No such relaxation applies to candidate acceptance. Both candidate cases must satisfy every primary criterion. Feasible groups rank only mean raw physical action time; exact ties use 25.2 then 24.8 degrees/s. Other valid groups rank mean absolute error then total all-label peak spill. Invalid groups cannot be elites. This is finite calibration, not CEM, PPO or a trained neural policy.

One choice freezes before two fresh validation pairs. Validation can authorize twenty new paired tests or stop; it cannot retune or choose a fallback. All 6, 10 or 30 actual cases and failures remain. A stopped study includes an explicit receipt and no invented checkpoint or test outcomes.

Primary dose is 40 +/-0.5 mL; +/-1 mL and original +/-0.1% (0.04 mL) remain separate. Full return to original pregrasp with 12 quiet frames must finish within 24 raw physical seconds, peak spill across all labels must be at most one particle and another two seconds must show no late dose change. All required numerical, mass, measurement and actual-motion checks also apply.

Two known baselines are bound to prior first40 chosen-angle traces, including the valid spill-two negative example. Equality across speeds is checked only for preparation before coarse dosing; the later coarse-anchor and whole correction cycles are expected to change. The inherited time-zero diagnostic substep counter is distinct from every checked completed solver frame.

Three runtime files reference 25 unchanged V12 dependencies. Fifty protected old 20/30/first40 bindings refer to existing public files without copying them again. Original and portable SHA values remain distinct, including the existing robot path conversion. Posthoc audit dependencies retain their V12/V13 relative paths and pinned V9 reference. Reconstruct licensed resources and the original module layout under a new protocol; never rebind guards to resume private journals.

No SDK, third-party robot assets, NPZ, raw step logs, video or personal paths are included. Kinematic pregrasp and particle-mass truth are not a real scale, contact-grasp or hardware test. A fixed 40 mL task and narrow randomization do not guarantee arbitrary 20--40 mL or replace a GUI default.
