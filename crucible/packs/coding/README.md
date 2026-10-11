# coding (a Crucible pack)

The legacy GPU code migration family (DESIGN §12): four seed tasks are real migrations with kit graders in GPU
verifier images:
- task01: OpenMP offload;
- task02: Triton;
- task03: Pallas;
- task04: a PCL CUDA 12 port.

The pack also holds span tasks and traps derived from task04.

```bash
crucible pack check packs/coding --level fixture   # CPU: the dense reward's oracle = 1 / base = 0 on all four seeds
crucible label --pack coding --out label_coding    # recorded frontier runs + PCL pull requests -> traces, Φ credit, SFT/KTO
```

Live grading (the semantic check, `crucible build`) needs the lab GPU host and the seeds' verifier images. Run it as
`crucible pack check packs/coding --level semantic --allow gpu` on such a host. The GPU gates' runners are
`python -m crucible_pack_coding.verify.golden_gpu` and `python -m crucible_pack_coding.forge.forge_gpu`.
