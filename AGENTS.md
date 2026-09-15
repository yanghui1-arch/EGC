# Research continuity

`docs/RESEARCH_PLAN.md` is the authoritative record of the research innovation,
experimental plan, progress, decisions, and next steps.

Before each turn involving project operations, read that file and relevant evidence.
Before the final response, check whether innovation, plan, progress, constraints or next
steps changed; update the file and decision log when they did. Do not invent progress
or rewrite historical results. Treat other reports as historical evidence.

Keep hypotheses separate from implemented methods and verified results. Do not call
structural annotation validation semantic accuracy, or smoke tests benchmark gains.
Record why an experimental direction changes, including negative findings.

The user runs local data-processing/API annotation scripts themselves. Do not launch those
jobs for them based on prior API authorization. The assistant edits code/docs, performs
necessary offline software checks, supplies commands and analyzes returned results.
For each run, give the location, exact command, input/output paths, request bound where
applicable, and files/logs to return. Do not present unimplemented commands as available.
All training and local-model inference happen on the user's server, where the user pulls
`yanghui1-arch/EGC` from GitHub.
Actual parameter counts below 7B use full training; 7B and above use LoRA.
Never write API keys into files or logs. Do not publish raw cases, API caches, or model weights.

These conventions add no approval step; continue work within existing user authorization.
