"""CPU-only regression tests for exact archived-source publication bindings.

Every fixture lives in a temporary directory. No SDK, model, real experiment
output, Git operation or publication destination is read or changed.
"""

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


def _load_subject():
    path = Path(__file__).resolve().with_name("sync_from_workspace.py")
    spec = importlib.util.spec_from_file_location("sync_source_binding_subject", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = _load_subject()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class PublicationSourceBindingTests(unittest.TestCase):
    def setUp(self):
        # The Windows desktop sandbox's redirected TEMP may prohibit child
        # writes. Keep isolated, automatically removed fixtures beside this
        # writable test file instead of relying on that machine TEMP policy.
        fixture_parent = Path(__file__).resolve().parent
        self.temporary = tempfile.TemporaryDirectory(
            prefix="_sync_source_binding_", dir=fixture_parent)
        self.base = Path(self.temporary.name).resolve()
        if self.base.parent != fixture_parent:
            raise RuntimeError("Test cleanup target must stay in the selected fixture directory.")
        self.addCleanup(self.temporary.cleanup)
        self.root = self.base / "workspace"
        self.root.mkdir()
        self.relative = "projects/development/controller.py"
        self.snapshot_relative = "archive/source_snapshot/controller.py"
        self.current = self.root / self.relative
        self.snapshot = self.root / self.snapshot_relative
        self.old_bytes = b"# exact frozen source\r\nVALUE = 1\r\n"
        self.reviewed_bytes = b"# reviewed development source\nVALUE = 2\n"
        self.binding = {
            "snapshot": self.snapshot_relative,
            "sha256": digest(self.old_bytes),
            "reviewed_workspace_sha256": digest(self.reviewed_bytes),
        }
        self.bindings = {self.relative: self.binding}
        patcher = mock.patch.dict(subject.ARCHIVED_SOURCE_BINDINGS, self.bindings, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def write(path, raw):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def reviewed_fixture(self):
        self.write(self.current, self.reviewed_bytes)
        self.write(self.snapshot, self.old_bytes)

    def selected(self, relative=None):
        return subject.publication_source(self.root, self.relative if relative is None else relative)

    def test_original_current_returns_itself_without_requiring_snapshot(self):
        self.write(self.current, self.old_bytes)
        self.assertFalse(self.snapshot.exists())
        self.assertEqual(self.selected().resolve(), self.current.resolve())
        self.assertEqual(self.current.read_bytes(), self.old_bytes)

    def test_known_reviewed_development_selects_exact_frozen_snapshot(self):
        self.reviewed_fixture()
        self.assertEqual(self.selected().resolve(), self.snapshot.resolve())
        self.assertEqual(self.selected().read_bytes(), self.old_bytes)
        self.assertEqual(self.current.read_bytes(), self.reviewed_bytes)

    def test_unknown_current_edit_rejected_even_with_valid_snapshot(self):
        self.reviewed_fixture()
        self.current.write_bytes(b"# new unreviewed edit\nVALUE = 3\n")
        with self.assertRaises(subject.SyncError):
            self.selected()

    def test_current_is_rehashed_after_an_accepted_call(self):
        self.reviewed_fixture()
        self.selected()
        self.current.write_bytes(self.reviewed_bytes + b"# later edit\n")
        with self.assertRaises(subject.SyncError):
            self.selected()

    def test_reviewed_development_requires_existing_snapshot(self):
        self.write(self.current, self.reviewed_bytes)
        with self.assertRaises(subject.SyncError):
            self.selected()

    def test_corrupt_snapshot_is_rejected(self):
        self.reviewed_fixture()
        self.snapshot.write_bytes(b"# corrupt snapshot\n")
        with self.assertRaises(subject.SyncError):
            self.selected()

    def test_snapshot_line_ending_change_is_not_the_frozen_bytes(self):
        self.reviewed_fixture()
        self.snapshot.write_bytes(self.old_bytes.replace(b"\r\n", b"\n"))
        with self.assertRaises(subject.SyncError):
            self.selected()

    def test_snapshot_is_rehashed_after_an_accepted_call(self):
        self.reviewed_fixture()
        self.selected()
        self.snapshot.write_bytes(self.old_bytes + b"# later edit\n")
        with self.assertRaises(subject.SyncError):
            self.selected()

    def test_missing_current_cannot_be_replaced_by_snapshot(self):
        self.write(self.snapshot, self.old_bytes)
        with self.assertRaises(subject.SyncError):
            self.selected()

    def test_current_path_traversal_is_rejected_even_when_hash_matches(self):
        outside = self.base / "outside.py"
        self.write(outside, self.old_bytes)
        relative = "../outside.py"
        with mock.patch.dict(subject.ARCHIVED_SOURCE_BINDINGS, {relative: dict(self.binding)}):
            with self.assertRaises(subject.SyncError):
                self.selected(relative)

    def test_absolute_current_outside_root_is_rejected(self):
        outside = self.base / "outside.py"
        self.write(outside, self.old_bytes)
        relative = str(outside)
        with mock.patch.dict(subject.ARCHIVED_SOURCE_BINDINGS, {relative: dict(self.binding)}):
            with self.assertRaises(subject.SyncError):
                self.selected(relative)

    def test_snapshot_path_traversal_is_rejected_even_when_hash_matches(self):
        self.write(self.current, self.reviewed_bytes)
        self.write(self.base / "outside.py", self.old_bytes)
        binding = {**self.binding, "snapshot": "../outside.py"}
        with mock.patch.dict(subject.ARCHIVED_SOURCE_BINDINGS, {self.relative: binding}):
            with self.assertRaises(subject.SyncError):
                self.selected()

    def test_sibling_with_common_prefix_is_outside_root(self):
        self.write(self.current, self.reviewed_bytes)
        self.write(self.base / "workspace-other" / "controller.py", self.old_bytes)
        binding = {**self.binding, "snapshot": "../workspace-other/controller.py"}
        with mock.patch.dict(subject.ARCHIVED_SOURCE_BINDINGS, {self.relative: binding}):
            with self.assertRaises(subject.SyncError):
                self.selected()

    def test_current_and_snapshot_must_be_files(self):
        self.current.mkdir(parents=True)
        with self.assertRaises(subject.SyncError):
            self.selected()
        self.current.rmdir()
        self.write(self.current, self.reviewed_bytes)
        self.snapshot.mkdir(parents=True)
        with self.assertRaises(subject.SyncError):
            self.selected()

    def test_unbound_safe_source_is_returned_without_an_archive(self):
        relative = "ordinary/source.py"
        current = self.root / relative
        self.write(current, b"# unbound source\n")
        self.assertEqual(self.selected(relative).resolve(), current.resolve())


if __name__ == "__main__":
    unittest.main()
