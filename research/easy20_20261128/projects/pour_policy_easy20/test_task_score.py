import unittest
import task_score


def row(seed=1, received=20., time=5., spill=0, valid=True):
    result = {"target_ml":20., "received_ml":received, "error_ml":received-20., "physical_time_s":time}
    strict = {"valid_evidence":valid, "full_home_within_deadline":time<=24.,
              "spill_passed":spill<=1, "no_late_dose_change":True,
              "maximum_all_label_spilled_particles":spill}
    return {"seed":seed,"result":result,"easy_assessment":task_score.assess(result,strict)}


class Contract(unittest.TestCase):
    def test_closed_interval(self):
        for dose in (19.,21.): self.assertTrue(row(received=dose)["easy_assessment"]["feasible"])
        for dose in (18.999,21.001): self.assertFalse(row(received=dose)["easy_assessment"]["feasible"])
    def test_all_constraints(self):
        for values in ({"spill":2},{"time":24.01},{"valid":False}):
            self.assertFalse(row(**values)["easy_assessment"]["feasible"])
    def test_feasible_compares_time(self):
        self.assertEqual(task_score.select({"accurate_slow":[row(time=6)],
                                          "under_fast":[row(received=19.2,time=4)]}),"under_fast")
    def test_failure_compares_error_before_spill(self):
        self.assertEqual(task_score.select({"under":[row(received=10)],
                                          "spill":[row(received=20,spill=2)]}),"spill")
    def test_invalid_cannot_win(self):
        self.assertEqual(task_score.select({"invalid":[row(valid=False)],"valid":[row()]}),"valid")
    def test_duplicate_seed(self):
        with self.assertRaises(ValueError): task_score.score([row(),row()])


if __name__ == "__main__": unittest.main()
