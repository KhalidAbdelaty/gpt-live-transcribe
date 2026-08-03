"""Commit staged changes with a clean message, no trailers. Temporary helper."""

import subprocess

MESSAGE = """Widen the silence hold and add a microphone calibration script

A 0.8s hold treated a breath or a dipping voice as the end of a turn, so
sentences committed halfway through and the model transcribed fragments.
1.5s waits out normal speech rhythm and still finalizes fast enough to read.

The right speech threshold depends on the microphone and the room, so
check_mic.py measures both and suggests a value instead of leaving it to
guesswork.
"""


def git(*args, stdin=None):
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, input=stdin, encoding="utf-8"
    )
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout.strip()


git("add", "-A")

staged = git("diff", "--cached", "--name-only").splitlines()
if any(f.strip() == ".env" for f in staged):
    raise SystemExit("refusing to commit: .env is staged")
print("staged:", ", ".join(staged))

tree = git("write-tree")
parent = git("rev-parse", "HEAD")
commit = git("commit-tree", tree, "-p", parent, stdin=MESSAGE)
git("reset", "--hard", commit)

print()
print(git("log", "--format=%an <%ae>%n%n%B", "-1"))
