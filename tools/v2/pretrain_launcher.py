"""Spawn-safe entry point for the platform pretrainer.

The guarded lazy import matters on Windows: spawn re-executes this file as
``__mp_main__``. Environment workers therefore do not import the torch-heavy learner.
"""

if __name__ == "__main__":
    import pretrain_platform

    raise SystemExit(pretrain_platform.main())
