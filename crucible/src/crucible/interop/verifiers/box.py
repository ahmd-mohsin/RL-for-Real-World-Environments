"""The shell the ``crucible-vf`` plugin runs in the agent's box before a shared grade (M11 lab audit R-3, P23): plain
POSIX ``sh`` over ``/proc`` with builtins only (``read``, ``case``, ``[``, ``kill``, ``set``), so a task image that lacks
``sed`` or ``head`` cannot make it skip a process, and it forks nothing but ``sleep`` between rounds. Kept apart from
``vf_taskset`` (which imports verifiers) so it can be tested in Crucible's own environment. The rules follow Crucible's
in-place Harbor quiesce (``harbor/quiesce.py``) as far as an in-box helper can.

- ``LIST_PIDS`` (after task setup, before the agent: the box is trusted then) prints ``<pid>:<start>`` for every process,
  ``<start>`` being field 22 of ``/proc/<pid>/stat`` (start time in clock ticks): the baseline. Keyed by start time, so a
  policy that cycles pids until it holds a baseline pid (or the listing shell's own, long gone) is not spared.
- ``QUIESCE`` spares this shell and its ancestors (the runtime's exec chain), PID 1, kernel threads (pid 2 or a child of
  2), single-threaded zombies and every baseline process still running with its baseline start time; every other process
  is SIGKILLed with ``$CRUCIBLE_KILL`` (``kill``; the caller always passes it, so the box's environment cannot choose it),
  up to three rounds a second apart. A thread-group leader that is a zombie while its other threads run (``Z`` with
  ``Threads`` > 1) is a target: those threads can still write. A process whose ``stat`` cannot be read is a target, and is
  listed as left when it remains: what cannot be classified never passes. Output: ``killed <pid>``, ``left <pid> <ppid>
  <state>`` (``?`` when unreadable), then ``quiesced``.
- ``verdict`` reads that as Harbor's in-place quiesce does: ``survivors`` (a target the policy started is left: scored 0,
  flag ``policy_process_survived``), ``respawned`` (only children of a baseline service other than PID 1 are left: a
  supervisor or a postmaster replacing killed workers; whose the respawn is cannot be told, so it masks), ``uninterruptible``
  (only processes in state D are left: SIGKILL pending; masks), ``error`` (the script did not run to its end; masks) or
  ``clean``.

Residual (documented, not fixable in the box): a root policy controls every binary in it, ``sh`` included, and can print
``quiesced`` itself; Crucible's in-place Harbor quiesce lets only the host's listing decide, which a verifiers runtime does
not offer. Tasks graded in verifiers' isolated (fresh-box) lifecycle do not depend on this."""

from __future__ import annotations

# Builtins only. ``st <pid>`` sets S_STATE, S_PPID, S_START from /proc/<pid>/stat (the fields after the last ") ", so a
# command name with spaces or parentheses cannot shift them); ``nthreads <pid>`` sets T from /proc/<pid>/status.
_PRELUDE = r"""
st() {
  S_STATE=; S_PPID=; S_START=
  IFS= read -r l < "/proc/$1/stat" 2>/dev/null || return 1
  r=${l##*) }
  set -f; set -- $r; set +f
  [ "$#" -ge 20 ] || return 1
  S_STATE=$1; S_PPID=$2; shift 19; S_START=$1
}
nthreads() {
  T=
  while IFS= read -r l; do
    case "$l" in Threads:*) set -f; set -- ${l#Threads:}; set +f; T=$1; return 0;; esac
  done < "/proc/$1/status" 2>/dev/null
}
"""

LIST_PIDS = _PRELUDE + r"""
for d in /proc/[0-9]*; do
  p=${d#/proc/}
  st "$p" && echo "$p:$S_START"
done
"""

QUIESCE = _PRELUDE + r"""
base=" $CRUCIBLE_BASE "
keep=" 1 "
p=$$
while [ -n "$p" ] && [ "$p" -gt 0 ] 2>/dev/null; do
  keep="$keep$p "
  st "$p" || break
  p=$S_PPID
done
target() {
  case "$keep" in *" $1 "*) return 1;; esac
  if st "$1"; then
    [ "$1" = 2 ] || [ "$S_PPID" = 2 ] && return 1
    case "$base" in *" $1:$S_START "*) return 1;; esac
    if [ "$S_STATE" = Z ]; then
      nthreads "$1"
      [ "${T:-1}" -le 1 ] 2>/dev/null && return 1
    fi
    return 0
  fi
  [ -d "/proc/$1" ]
}
round=0
while :; do
  round=$((round + 1))
  n=0
  for d in /proc/[0-9]*; do
    p=${d#/proc/}
    target "$p" || continue
    n=$((n + 1))
    $CRUCIBLE_KILL -9 "$p" 2>/dev/null && echo "killed $p"
  done
  [ "$n" -eq 0 ] && break
  sleep 1 2>/dev/null || :
  [ "$round" -ge 3 ] && break
done
for d in /proc/[0-9]*; do
  p=${d#/proc/}
  target "$p" || continue
  if [ -n "$S_STATE" ]; then echo "left $p $S_PPID $S_STATE"; else echo "left $p ? ?"; fi
done
echo quiesced
"""


def baseline(stdout: str) -> list[str]:
    """The ``<pid>:<start>`` entries ``LIST_PIDS`` printed."""
    out = set()
    for tok in (stdout or "").split():
        pid, _, start = tok.partition(":")
        if pid.isdigit() and start.isdigit():
            out.add(tok)
    return sorted(out, key=lambda t: int(t.partition(":")[0]))


def parse(stdout: str) -> tuple[list[int], list[dict], bool]:
    """(killed pids, processes left as ``{pid, ppid, state}``, whether the script ran to its end) from ``QUIESCE``."""
    killed, left, ran = [], [], False
    for line in (stdout or "").splitlines():
        f = line.split()
        if f[:1] == ["killed"] and len(f) == 2 and f[1].isdigit():
            killed.append(int(f[1]))
        elif f[:1] == ["left"] and len(f) == 4 and f[1].isdigit():
            left.append({"pid": int(f[1]), "ppid": f[2], "state": f[3]})
        elif line.strip() == "quiesced":
            ran = True
    return killed, left, ran


def verdict(left: list[dict], base: list[str], ran: bool) -> str:
    """``error`` > ``survivors`` > ``respawned`` > ``uninterruptible`` > ``clean`` (module doc)."""
    if not ran:
        return "error"
    services = {t.partition(":")[0] for t in base} - {"1"}
    survivors = [x for x in left if x["state"] != "D" and x["ppid"] not in services]
    if survivors:
        return "survivors"
    if any(x["state"] != "D" for x in left):
        return "respawned"
    return "uninterruptible" if left else "clean"
