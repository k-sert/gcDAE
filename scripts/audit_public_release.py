"""Read-only, value-redacting heuristic scan of the worktree and reachable Git objects.

Run from the repository root. This is a fallback, not a credential detector guarantee.
No repository modules are imported and no matched values are printed.
"""
import collections
import json
import pathlib
import re
import subprocess


RULES = {
    "credential": r"(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{32,}|-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "assigned_secret": r"(?i)(?:password|passwd|api[_-]?key|access[_-]?token|client[_-]?secret|wandb[_-]?api[_-]?key)\s*[:=]\s*['\"][^'\"\r\n]{8,}['\"]",
    "personal_path": r"(?i)/(?:home|Users|scratch(?:-shared)?|projects|gpfs|lustre|mnt|work|data\d*)/[^\s'\"\\]+|[A-Z]:[\\/](?:Users|home)[\\/][^\s'\"]+",
    "email": r"[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "private_network": r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b|(?i:https?://[^\s/]+\.(?:internal|local)\b)",
    "authenticated_url": r"https?://[^\s/@:]+:[^\s/@]+@",
}
PATTERNS = {name: re.compile(rule) for name, rule in RULES.items()}


def git(*args):
    return subprocess.check_output(["git", *args])


def findings(data):
    text = data.decode("utf-8", errors="replace")
    return {name: [i for i, line in enumerate(text.splitlines(), 1) if regex.search(line)]
            for name, regex in PATTERNS.items() if regex.search(text)}


def main():
    objects = {}
    for line in git("rev-list", "--objects", "--all").decode().splitlines():
        oid, _, path = line.partition(" ")
        objects[oid] = path
    proc = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    totals = collections.Counter()
    binaries = []
    large = []
    for oid, path in objects.items():
        proc.stdin.write((oid + "\n").encode())
        proc.stdin.flush()
        header = proc.stdout.readline().decode().split()
        size = int(header[2])
        data = proc.stdout.read(size)
        assert proc.stdout.read(1) == b"\n"
        kind = header[1]
        totals[kind] += 1
        if kind not in ("blob", "commit", "tag"):
            continue
        if size > 1024 * 1024:
            large.append({"object": oid, "path": path, "bytes": size})
        if kind == "blob" and b"\0" in data:
            binaries.append({"object": oid, "path": path, "bytes": size})
        hits = findings(data)
        if hits:
            print(json.dumps({"scope": "history", "kind": kind, "object": oid,
                              "path": path, "rules": hits}))
    proc.stdin.close()
    assert proc.wait() == 0
    count = 0
    for path in sorted(pathlib.Path(".").rglob("*")):
        if ".git" in path.parts or not path.is_file() or path.is_symlink():
            continue
        count += 1
        hits = findings(path.read_bytes())
        if hits:
            print(json.dumps({"scope": "worktree", "path": path.as_posix(), "rules": hits}))
    print(json.dumps({"summary": dict(totals), "worktree_files": count,
                      "binary_blobs": binaries, "large_objects": large}))


if __name__ == "__main__":
    main()
