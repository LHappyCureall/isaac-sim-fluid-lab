# First-angle calibration after fixed coarse dosing

Only the first correction angle varies: baseline 87 degrees, candidates 87.05/87.1 degrees. Both roles keep coarse dosing 24.8 degrees/s, retry 87.2 degrees, conditional retry enabled and at most two cycles. All motion speeds, holds, geometry, particles and solver parameters remain fixed. See [the actual selected results](../../results/balanced40_20261128.json).

Nine seen calibration cases include three conditions for every profile. Seeds 197304, 198304 and 198403 are development data; the last was a previously observed held-out undershoot in the earlier coarse-speed study. It is not a new holdout here. All three candidate cases must satisfy every primary criterion. Feasible groups rank only mean raw physical action time; exact ties follow 87.05 then 87.1 degrees. Other valid groups rank mean absolute error then total peak spill; invalid groups cannot be elites. Numerically valid baseline task failures remain explicit negative controls.

One choice freezes before two fresh validation pairs. Validation can authorize twenty new paired tests or stop; it cannot retune or choose a fallback. All 9, 13 or 33 actual cases and failures remain. A stopped study includes an explicit receipt and no invented checkpoint or test outcomes. Finite angle calibration is not CEM, PPO or training a neural policy.

Primary dose is 40 +/-0.5 mL; +/-1 mL and original +/-0.1% (0.04 mL) remain separate. Full return to original pregrasp with 12 quiet frames must finish within 24 raw physical seconds, peak spill across all labels must be at most one particle and another two seconds must show no late dose change. All required numerical, mass, measurement and actual-motion checks also apply.

Three known baselines are bound to prior coarse 24.8 chosen-profile traces, including the negative undershoot example. Between angles, equality is checked through the common actually quiet 78 coarse endpoint before first-angle treatment; changed subsequent cycles are not required equal. The inherited time-zero diagnostic substep counter is separate from every completed solver frame.

Three runtime files reference 25 unchanged V12 dependencies. Fifty-eight protected old 20/30, first40 and coarse40 bindings refer to existing public files without copying them again. Original and portable SHA values remain distinct, including the existing robot path conversion. Posthoc audit dependencies retain their V12/V13 relative paths and pinned V9 reference. Reconstruct the licensed resources and original module layout under a new protocol; never rebind guards to resume private journals.

No SDK, third-party robot assets, NPZ, raw step logs, video or personal paths are included. Kinematic pregrasp and particle-mass truth are not a real scale, contact-grasp or hardware test. A fixed 40 mL task and narrow randomization do not guarantee arbitrary 20--40 mL or replace a GUI default.
