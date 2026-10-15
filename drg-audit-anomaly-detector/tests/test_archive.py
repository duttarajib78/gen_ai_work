import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import tests.helpers  # noqa: F401,I001  (puts src/ on sys.path)

from drg_audit.archive import safe_extract_tar
from drg_audit.errors import ArtifactError


def make_tar(path, members):
    """members: list of (name, bytes | None, type)  -- bytes None means directory/link entry."""
    with tarfile.open(path, "w:gz") as tar:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                tar.addfile(info)
            elif kind == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = "a.txt"
                tar.addfile(info)


class SafeExtractTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.archive = self.root / "a.tar.gz"
        self.dest = self.root / "out"

    def test_extracts_regular_files_with_private_permissions(self):
        make_tar(self.archive, [("./", None, "dir"), ("./a.txt", b"hello", "file"), ("sub/b.txt", b"x", "file")])
        files = safe_extract_tar(self.archive, self.dest)
        self.assertEqual(sorted(files), ["a.txt", "sub/b.txt"])
        self.assertEqual((self.dest / "a.txt").read_bytes(), b"hello")
        self.assertEqual((self.dest / "a.txt").stat().st_mode & 0o777, 0o600)

    def test_path_traversal_is_refused(self):
        make_tar(self.archive, [("../evil.txt", b"x", "file")])
        with self.assertRaises(ArtifactError):
            safe_extract_tar(self.archive, self.dest)
        self.assertFalse((self.root / "evil.txt").exists())

    def test_absolute_paths_are_refused(self):
        make_tar(self.archive, [("/tmp/drg_audit_evil.txt", b"x", "file")])
        with self.assertRaises(ArtifactError):
            safe_extract_tar(self.archive, self.dest)
        self.assertFalse(Path("/tmp/drg_audit_evil.txt").exists())

    def test_links_are_refused(self):
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                make_tar(self.archive, [("a.txt", b"x", "file"), ("link", None, kind)])
                with self.assertRaises(ArtifactError):
                    safe_extract_tar(self.archive, self.dest)

    def test_bounds(self):
        make_tar(self.archive, [("a", b"1", "file"), ("b", b"2", "file")])
        with self.assertRaises(ArtifactError):
            safe_extract_tar(self.archive, self.dest, max_members=1)
        with self.assertRaises(ArtifactError):
            safe_extract_tar(self.archive, self.dest, max_total_bytes=1)

    def test_non_tar_input(self):
        self.archive.write_bytes(b"not a tar")
        with self.assertRaises(ArtifactError):
            safe_extract_tar(self.archive, self.dest)


if __name__ == "__main__":
    unittest.main()
