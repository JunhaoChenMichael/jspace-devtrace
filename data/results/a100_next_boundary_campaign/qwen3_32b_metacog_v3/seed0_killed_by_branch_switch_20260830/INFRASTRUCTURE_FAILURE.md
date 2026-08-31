# Infrastructure failure, preserved

This run was killed mid-M0 by the operator switching git branches while the job
was reading source files and writing into the working tree. Both this job and
the scale-sweep array died at the same instant (13:14:21) on different nodes;
the exit code 120 is CPython's "could not flush stdout at exit", which followed
from the output directory being swept by `git stash -u`.

No scientific decision was produced: only the Explicit M0 measurement completed,
the M0 gate never ran, no training started, and no OOD battery was opened. No
OOD attempt was consumed. The run is preserved rather than deleted, and a fresh
attempt was started in `seed0/`.

Cause is operator error, not a defect in the campaign code.
