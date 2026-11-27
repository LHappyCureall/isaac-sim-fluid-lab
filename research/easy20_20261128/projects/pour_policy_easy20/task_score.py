"""Separate introductory task; never alters the original strict assessment."""
import math

TASK_KIND = "fixed20ml_absolute1ml_24s_onegrain_v1"


def assess(result, strict):
    if result["target_ml"] != 20. or not math.isfinite(result["received_ml"]):
        raise ValueError("This introductory task measures exactly20mL")
    output = dict(strict)
    output.update(task_kind=TASK_KIND, tolerance_ml=1.,
                  dose_passed=19. <= result["received_ml"] <= 21.)
    output["feasible"] = bool(output["valid_evidence"] and output["dose_passed"]
        and output["full_home_within_deadline"] and output["spill_passed"]
        and output["no_late_dose_change"])
    return output


def score(rows):
    if not rows or len({r["seed"] for r in rows}) != len(rows):
        raise ValueError("One complete row per distinct scheduled seed required")
    if any(not r["easy_assessment"]["valid_evidence"] for r in rows):
        return (2,)
    if all(r["easy_assessment"]["feasible"] for r in rows):
        return (0, sum(r["result"]["physical_time_s"] for r in rows) / len(rows))
    return (1, sum(abs(r["result"]["error_ml"]) for r in rows) / len(rows),
            sum(r["easy_assessment"]["maximum_all_label_spilled_particles"] for r in rows))


def select(groups):
    """Fixed insertion order resolves exact ties; test rows never enter this API."""
    if not groups:
        raise ValueError("No training/validation groups")
    choices = [(score(rows), index, name) for index, (name, rows) in enumerate(groups.items())]
    key, _, name = min(choices)
    if key[0] == 2:
        raise ValueError("No numerically valid candidate")
    return name
