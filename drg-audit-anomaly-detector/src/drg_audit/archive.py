"""Safe extraction of tar archives that cross a trust boundary (for example a training job's model.tar.gz).

Only regular files and directories are extracted. Links, devices, absolute paths and any member that
would land outside the destination are refused, and member count / total size are bounded. Members are
streamed out by hand rather than via ``extractall`` so behaviour is identical on every Python version.
"""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path, PurePosixPath

from drg_audit.errors import ArtifactError


def safe_extract_tar(
    archive: str | Path,
    destination: str | Path,
    *,
    max_members: int = 10_000,
    max_total_bytes: int = 4 * 1024**3,
) -> list[str]:
    """Extract ``archive`` into ``destination``; returns the extracted file paths (relative)."""
    dest = Path(destination).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    total = 0
    try:
        tar = tarfile.open(archive, "r:*")  # noqa: SIM115  (closed in finally)
    except (tarfile.TarError, OSError) as exc:
        raise ArtifactError("archive is unreadable or not a tar file") from exc
    try:
        members = tar.getmembers()
        if len(members) > max_members:
            raise ArtifactError("archive contains too many members")
        for member in members:
            if not (member.isfile() or member.isdir()):
                raise ArtifactError("archive contains a link or special file; refusing to extract")
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or "\x00" in member.name:
                raise ArtifactError("archive member has an unsafe path")
            target = (dest / member.name).resolve()
            if target != dest and dest not in target.parents:
                raise ArtifactError("archive member escapes the destination directory")
            total += max(member.size, 0)
            if total > max_total_bytes:
                raise ArtifactError("archive expands beyond the permitted size")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ArtifactError("archive member cannot be read")
            with source, open(target, "wb") as out:
                shutil.copyfileobj(source, out)
            target.chmod(0o600)
            extracted.append(str(target.relative_to(dest)))
    finally:
        tar.close()
    return extracted
