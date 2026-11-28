"""Fixed40mL half-mL task; keep the raw strict gate separate."""
import math

TASK_KIND = "fixed40ml_absolute05ml_24s_onegrain_v1"


def assess(result, strict):
    target, received = result["target_ml"], result["received_ml"]
    if (type(target) not in (int, float) or target != 40.
            or type(received) not in (int, float) or not math.isfinite(received)):
        raise ValueError("This task requires actual40mL and finite measurement")
    output = dict(strict)
    output.update(task_kind=TASK_KIND, tolerance_ml=.5,
                  dose_passed=target - .5 <= received <= target + .5)
    output["feasible"] = bool(output["valid_evidence"] and output["dose_passed"]
        and output["full_home_within_deadline"] and output["spill_passed"]
        and output["no_late_dose_change"])
    return output


def easy_assess(result, strict):
    output = assess(result, strict)
    target, received = result["target_ml"], result["received_ml"]
    output.update(task_kind="fixed40ml_absolute1ml_24s_onegrain_v1",
                  tolerance_ml=1., dose_passed=target - 1. <= received <= target + 1.)
    output["feasible"] = bool(output["valid_evidence"] and output["dose_passed"]
        and output["full_home_within_deadline"] and output["spill_passed"]
        and output["no_late_dose_change"])
    return output
