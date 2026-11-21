"""Download only reviewed CC0 table maps, checking every output's SHA-256.

No SDK or generated robot assets are downloaded. The marble ZIP is read by
three exact member names; its contents are never extracted as filesystem paths.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "textures" / "table_options" / "sources.json"
STYLES = ("gray_concrete", "pale_oak", "white_marble")
MAPS = ("albedo", "roughness", "normal_gl")
MARBLE_MEMBERS = {
    "albedo": "Marble012_2K-JPG_Color.jpg",
    "roughness": "Marble012_2K-JPG_Roughness.jpg",
    "normal_gl": "Marble012_2K-JPG_NormalGL.jpg",
}
ALLOWED_HOSTS = {"dl.polyhaven.org", "ambientcg.com"}


def _validate_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS or parsed.username or parsed.password:
        raise ValueError(f"Not a reviewed HTTPS texture host: {url}")
    return url


def load_manifest(path=MANIFEST):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(manifest) != set(STYLES):
        raise ValueError("Texture assets must exactly match the reviewed manifest allowlist.")
    for style, asset in manifest.items():
        if asset.get("license") != "CC0 1.0" or set(asset.get("maps", {})) != set(MAPS):
            raise ValueError(f"Invalid CC0 asset or map allowlist: {style}")
        for name, item in asset["maps"].items():
            expected_name = f"{style}_{name}_2k.jpg"
            if item.get("file") != expected_name or Path(item["file"]).name != item["file"]:
                raise ValueError(f"Unexpected output filename for {style}/{name}.")
            if not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
                raise ValueError(f"Missing SHA-256 for {style}/{name}.")
            if not isinstance(item.get("bytes"), int) or item["bytes"] <= 0:
                raise ValueError(f"Missing expected byte count for {style}/{name}.")
            if style != "white_marble":
                _validate_url(item.get("download_url", ""))
        if style == "white_marble":
            _validate_url(asset.get("download_url", ""))
            if not isinstance(asset.get("archive_bytes"), int) or asset["archive_bytes"] <= 0:
                raise ValueError("Missing marble archive byte count.")
    return manifest


def verify_bytes(data, item):
    if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ValueError(f"Size or SHA-256 mismatch for {item['file']}.")


def read_archive_map(archive_bytes, name, item):
    """Read a fixed trusted member into memory, not a caller-supplied path."""
    if name not in MARBLE_MEMBERS:
        raise ValueError("Unknown marble map.")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        info = archive.getinfo(MARBLE_MEMBERS[name])
        if info.is_dir() or info.file_size != item["bytes"]:
            raise ValueError(f"Unexpected marble ZIP member size: {info.filename}")
        data = archive.read(info)
    verify_bytes(data, item)
    return data


def download(url, expected_bytes):
    request = Request(_validate_url(url), headers={"User-Agent": "isaac-sim-fluid-lab/1.0"})
    with urlopen(request, timeout=45) as response:
        if urlparse(response.geturl()).scheme != "https":
            raise ValueError("Texture download redirected away from HTTPS.")
        chunks, count = [], 0
        while True:
            chunk = response.read(min(1024 * 1024, expected_bytes + 1 - count))
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
            if count > expected_bytes:
                raise ValueError("Texture download exceeds its reviewed byte count.")
    if count != expected_bytes:
        raise ValueError("Texture download does not match its reviewed byte count.")
    return b"".join(chunks)


def _destination(directory, filename):
    destination = (directory / filename).resolve()
    if not destination.is_relative_to(directory):
        raise ValueError("Texture destination escapes the chosen directory.")
    return destination


def _save(directory, item, data):
    verify_bytes(data, item)
    directory.mkdir(parents=True, exist_ok=True)
    destination = _destination(directory, item["file"])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--styles", choices=STYLES, nargs="+", default=list(STYLES))
    parser.add_argument("--directory", type=Path, default=MANIFEST.parent)
    parser.add_argument("--dry-run", action="store_true", help="Show missing maps without downloading.")
    parser.add_argument("--verify-only", action="store_true", help="Check local maps without network access.")
    args = parser.parse_args(argv)
    if args.dry_run and args.verify_only:
        parser.error("Use either --dry-run or --verify-only.")
    manifest = load_manifest()
    directory = args.directory.expanduser().resolve()
    missing = []
    for style in dict.fromkeys(args.styles):
        asset = manifest[style]
        wanted = []
        for name in MAPS:
            item = asset["maps"][name]
            destination = _destination(directory, item["file"])
            valid = False
            if destination.is_file():
                try:
                    verify_bytes(destination.read_bytes(), item)
                    valid = True
                except ValueError:
                    pass
            print(f"{'verified' if valid else 'missing or changed'}: {item['file']}")
            if not valid:
                wanted.append(name)
                missing.append(item["file"])
        if not wanted or args.dry_run or args.verify_only:
            continue
        archive = download(asset["download_url"], asset["archive_bytes"]) if style == "white_marble" else None
        for name in wanted:
            item = asset["maps"][name]
            data = (read_archive_map(archive, name, item) if archive is not None
                    else download(item["download_url"], item["bytes"]))
            _save(directory, item, data)
            print(f"saved and verified: {item['file']}")
    return 1 if args.verify_only and missing else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Asset preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
