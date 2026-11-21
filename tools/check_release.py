"""Small CPU-only checks for this source distribution; no Kit/GPU required."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = list(ROOT.glob("*.py")) + list((ROOT / "tools").glob("*.py"))
    sources += list((ROOT / "assets/robot_testtube_pour/scripts").glob("*.py"))
    for source in sources:
        compile(source.read_text(encoding="utf-8"), str(source), "exec")
    required = ("droplet_milkcrown_newton.py", "droplet_milkcrown_newton_surface.py",
                "robot_testtube_pour.py", "robot_target_volume_pour.py", "lab_paths.py")
    assert all((ROOT / name).is_file() for name in required), "Missing experiment source"
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.relative_to(ROOT).parts:
            continue
        rel = path.relative_to(ROOT)
        # Local runtime outputs are ignored by Git and do not affect a checkout.
        if rel.parts[0] in ("droplet_out", "test", "__pycache__"):
            continue
        if path.suffix in (".py", ".md", ".json"):
            assert path.stat().st_size < 2_000_000, f"Unexpectedly large text file: {rel}"
    data = json.loads((ROOT / "results/target_volume.json").read_text(encoding="utf-8"))
    assert data["success"] and len(data["results"]) == 3
    assert [r["target_ml"] for r in data["results"]] == [20, 30, 40]
    for result in data["results"]:
        assert abs(result["received_ml"] - result["target_ml"] - result["error_ml"]) < 1e-8
        assert abs(result["error_ml"]) < 2 and result["spilled_ml"] == 0
        assert result["mass_conserved"] and result["particle_count"] == 5330
    print(f"PASS: {len(sources)} Python sources compile; published result summary is consistent.")
    print("GPU physics is validated separately in Isaac Sim; this check does not run it.")


if __name__ == "__main__":
    main()
