"""Task generators (swarm-pipeline/00 §2, §15 P5): every generator emits ``factory.longtask.LongTask``s that pass the
same executed gates as a mined window. ``base`` holds the Generator Protocol, the synthetic-history builder the window
machinery runs on, the deterministic gate runner (S3, S5, S6, S8, S10) and the funnel; ``rebreak``, ``stitch`` and
``open`` are the generators."""
