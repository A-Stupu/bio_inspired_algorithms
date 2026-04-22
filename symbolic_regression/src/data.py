"""
Data loading utilities.

Parses instance files in the format:
    n
    x1 y1
    x2 y2
    ...
"""
import os
import re


def load_instance(filepath: str) -> tuple[list[tuple[float, float]], dict]:
    """
    Load an instance file.

    Returns
    -------
    data : list of (x, y) tuples
    meta : dict with keys 'n', 'type', 'id', 'filename'
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    data = []
    for line in lines[1:n + 1]:
        parts = line.split()
        data.append((float(parts[0]), float(parts[1])))

    if len(data) != n:
        raise ValueError(f"Expected {n} data points, got {len(data)}")

    # Parse metadata from filename
    fname = os.path.basename(filepath)
    meta  = _parse_filename(fname)
    meta['n'] = n

    return data, meta


def _parse_filename(fname: str) -> dict:
    """
    Extract type and id from filenames.

    Handles both simple types (sr_poly_4.txt)
    and compound types (sr_challenge_a_01.txt).
    """
    stem = fname.replace('.txt', '')
    parts = stem.split('_')

    meta        = {'filename': fname, 'type': 'unknown', 'id': '?'}
    simple_types  = {'poly', 'ratio', 'approx', 'periodic'}
    compound_types = {'challenge'}

    for i, p in enumerate(parts):
        if p in compound_types and i + 1 < len(parts):
            meta['type'] = p + '_' + parts[i + 1]   # e.g. challenge_a
            if i + 2 < len(parts):
                meta['id'] = parts[i + 2]
            break
        if p in simple_types:
            meta['type'] = p
            if i + 1 < len(parts):
                meta['id'] = parts[i + 1]
            break

    return meta


def load_all_instances(directory: str) -> list[tuple[list, dict]]:
    """Load all .txt instance files from a directory."""
    results = []
    for fname in sorted(os.listdir(directory)):
        if fname.startswith('sr_') and fname.endswith('.txt'):
            path = os.path.join(directory, fname)
            try:
                data, meta = load_instance(path)
                results.append((data, meta))
            except Exception as e:
                print(f"Warning: could not load {fname}: {e}")
    return results