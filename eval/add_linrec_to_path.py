try:
    import linrec
except Exception:
    import sys # Path hack
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))