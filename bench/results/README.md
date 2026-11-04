# bench/results

One CSV per benchmark family, appended to (never rewritten). Numbers live here or in PROGRESS entries, not in report files.

Columns, in this order:

| column | meaning |
|---|---|
| `date` | ISO date of the run, `YYYY-MM-DD` |
| `commit` | short git hash of the code that was measured (add `-dirty` if uncommitted changes) |
| `device` | GPU, e.g. `RTX 3060 Laptop`, `Intel UHD` |
| `browser` | browser and version, or `native` for the native harness |
| `scene` | scene id from the test-scene manifest |
| `metric` | what was measured, with the unit in the name, e.g. `gpu_ms_probe_update` |
| `value` | the number |
