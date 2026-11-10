#!/usr/bin/env python3
"""Organize completed Nighres outputs for every sub-*/ses-*/anat.

By default, PREVIEW ONLY. Pass --apply to move, or --undo MANIFEST to restore.

On filesystems rejecting renameat2(RENAME_NOREPLACE), this script uses a
no-overwrite HARD-LINK-FIRST move on the SAME filesystem:
  1. Create the destination hard link (fails if destination already exists).
  2. Verify BOTH paths reference the exact same inode and file metadata.
  3. Only then unlink the old pathname. The file's data is NOT deleted.

If interrupted between (1) and (3), both paths safely exist. --undo MANIFEST
also handles this particular interrupted state.

IMPORTANT: Run only after all relevant processing jobs have finished. Scripts
that expect the old paths will need to be updated after organization.
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

DEFAULT_ROOT = Path(
    '/data/pt_03187/data/in_vivo/bids/derivatives/LORAKS/'
    'derivatives/nighres/echoes_to_mni'
)
FIELDS = ['source', 'destination', 'device', 'inode', 'size', 'mtime_ns']


def kind(name):
    ahead = name.endswith(('_align-ahead_def-img.nii',
                           '_align-ahead_def-img.nii.gz'))
    reorient = name.endswith(('_reorient.nii', '_reorient.nii.gz'))
    lc_header = name.endswith(('_in-PDw_header.nii',
                               '_in-PDw_header.nii.gz'))
    modality = re.search(r'_acq-(PDw|T1w|MTw)_', name)
    is_lc = '_chunk-LCslab_' in name

    if ahead:
        if is_lc:
            return ('AHEAD', 'LCslab')
        if modality:
            return ('AHEAD', modality.group(1))
    if reorient and modality:
        return ('Reoriented', modality.group(1))
    if lc_header and is_lc:
        return ('LCslab_PDw_header', 'LCslab')
    if re.fullmatch(r'.*_coreg-(mtw2pdw|t1w2pdw)_ants-'
                    r'(def0|map|invmap)\.nii\.gz', name):
        contrast = 'MTw_to_PDw' if '_coreg-mtw2pdw_' in name else 'T1w_to_PDw'
        return ('Transforms', contrast)
    return None


def fingerprint(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def expect_fingerprint(path, expected):
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f'Expected a regular non-symlink file: {path}')
    result = path.stat()
    if fingerprint(result) != fingerprint(expected):
        raise RuntimeError(f'File identity or metadata changed: {path}')
    return result


def check_root(root):
    if root.name != 'echoes_to_mni' or root.is_symlink() or not root.is_dir():
        raise RuntimeError(f'Invalid root: {root} (expected a real echoes_to_mni directory)')


def ensure_folders(anat, destination, create=False):
    """Never write via a symlinked destination folder."""
    try:
        parts = destination.parent.relative_to(anat).parts
    except ValueError:
        raise RuntimeError(f'Destination escaped anatomical directory: {destination}')
    current = anat
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f'Destination directory is a symlink: {current}')
        if current.exists():
            if not current.is_dir():
                raise RuntimeError(f'Expected directory: {current}')
        elif create:
            current.mkdir()  # Only one checked level at a time


def collect(root):
    plan = []
    untouched = 0
    for subject in sorted(root.glob('sub-*')):
        if not subject.is_dir() or subject.is_symlink():
            continue
        for session in sorted(subject.glob('ses-*')):
            if not session.is_dir() or session.is_symlink():
                continue
            anat = session / 'anat'
            if not anat.is_dir() or anat.is_symlink():
                continue
            for source in sorted(anat.iterdir()):
                if source.is_dir() and not source.is_symlink():
                    continue  # Existing organizational folders
                if not source.is_file() or source.is_symlink():
                    untouched += 1
                    continue
                folders = kind(source.name)
                if folders is None:
                    untouched += 1
                    continue
                destination = anat.joinpath(*folders, source.name)
                ensure_folders(anat, destination)
                if os.path.lexists(destination):
                    raise RuntimeError(f'Existing destination, refusing overwrite:\n{destination}')
                plan.append((source, destination, source.stat()))
    return plan, untouched


def move_no_overwrite(source, destination, expected):
    """Hard-link first, verify identity, then remove ONLY the old name.

    This cannot overwrite destination. Hard linking does not duplicate MRI data.
    If unlink fails or process is interrupted, there may be TWO paths to the
    SAME data, which can be repaired using --undo with the saved manifest.
    """
    expect_fingerprint(source, expected)
    if os.path.lexists(destination):
        raise RuntimeError(f'Refusing to overwrite: {destination}')
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise RuntimeError(f'Cross-filesystem operation refused: {source}')

    # os.link fails with FileExistsError if destination appears, even in a race.
    os.link(source, destination, follow_symlinks=False)

    # Do not unlink the original path until the new path is verified.
    before = expect_fingerprint(source, expected)
    after = expect_fingerprint(destination, expected)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise RuntimeError(f'Hard-link identity check failed; source retained: {destination}')

    os.unlink(source)  # Only old directory entry, NOT the underlying file data
    expect_fingerprint(destination, expected)
    if os.path.lexists(source):
        raise RuntimeError(f'Original path unexpectedly still exists: {source}')


def write_manifest(root, plan):
    manifest = root / ('organization_manifest_'
                       f'{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}.csv')
    with manifest.open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        for source, destination, st in plan:
            writer.writerow([str(source), str(destination), st.st_dev, st.st_ino,
                             st.st_size, st.st_mtime_ns])
        f.flush()
        os.fsync(f.fileno())
    return manifest


def apply(root, plan):
    if not plan:
        print('Nothing to move.')
        return
    manifest = write_manifest(root, plan)
    print(f'Recovery manifest (KEEP THIS FILE): {manifest}', flush=True)

    for n, (source, destination, st) in enumerate(plan, start=1):
        expect_fingerprint(source, st)
        ensure_folders(source.parent, destination, create=True)
        move_no_overwrite(source, destination, st)
        if n % 50 == 0 or n == len(plan):
            print(f'Moved and verified {n}/{len(plan)}', flush=True)
    print('COMPLETE. All recognized files have been organized. Keep the manifest.')


def undo(root, manifest):
    if not manifest.is_file() or manifest.is_symlink():
        raise RuntimeError(f'Manifest missing or symlink: {manifest}')
    if manifest.parent.resolve() != root.resolve() or not manifest.name.startswith('organization_manifest_'):
        raise RuntimeError('Manifest must be directly inside the project root.')
    with manifest.open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise RuntimeError('Unexpected manifest columns; refusing undo.')
        rows = list(reader)
    if not rows:
        raise RuntimeError('Empty manifest; refusing undo.')

    actions = []
    unchanged = 0
    for row in rows:
        old = Path(row['source'])
        new = Path(row['destination'])
        if (not old.is_relative_to(root) or not new.is_relative_to(root) or
                kind(old.name) is None or
                new != old.parent.joinpath(*kind(old.name), old.name)):
            raise RuntimeError(f'Invalid manifest entry: {row}')
        expected = (int(row['device']), int(row['inode']),
                    int(row['size']), int(row['mtime_ns']))
        old_exists = os.path.lexists(old)
        new_exists = os.path.lexists(new)

        if old_exists and not new_exists:
            # This row was never moved or has already been restored.
            unchanged += 1
            continue
        if not old_exists and not new_exists:
            raise RuntimeError(f'Both paths missing (no changes made by undo): {old} / {new}')
        if new.is_symlink() or not new.is_file() or fingerprint(new.stat()) != expected:
            raise RuntimeError(f'Unexpected organized file; cannot undo safely: {new}')
        if old_exists:
            # Interrupted between link() and unlink(): both refer to SAME inode.
            if old.is_symlink() or not old.is_file() or fingerprint(old.stat()) != expected:
                raise RuntimeError(f'Undo conflict, both names exist but differ: {old} / {new}')
            actions.append(('remove_extra_link', old, new, expected))
        else:
            actions.append(('restore_original_name', old, new, expected))

    print(f'Undo preflight: {len(actions)} actions; {unchanged} already at original paths.')
    for n, (action, old, new, expected) in enumerate(actions, start=1):
        if action == 'remove_extra_link':
            # The old name already exists, so discard only the redundant new link.
            if fingerprint(old.stat()) != expected or fingerprint(new.stat()) != expected:
                raise RuntimeError(f'File changed during undo: {old} / {new}')
            os.unlink(new)
            if fingerprint(old.stat()) != expected:
                raise RuntimeError(f'Original changed unexpectedly: {old}')
        else:
            ensure_folders(old.parent, old, create=False)
            # Expected is a 4-tuple, but move_no_overwrite takes stat_result.
            current = new.stat()
            if fingerprint(current) != expected:
                raise RuntimeError(f'File changed during undo: {new}')
            move_no_overwrite(new, old, current)
        if n % 50 == 0 or n == len(actions):
            print(f'Undo processed {n}/{len(actions)}', flush=True)
    print('UNDO COMPLETE. The manifest has been retained.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT,
                        help='Directory containing sub-*/ses-*/anat')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--apply', action='store_true', help='Move files safely')
    group.add_argument('--undo', type=Path, help='Restore paths using a saved manifest')
    parser.add_argument('--verbose', action='store_true', help='List every planned move')
    args = parser.parse_args()
    root = args.root.absolute()
    check_root(root)
    if args.undo:
        undo(root, args.undo.absolute())
        return
    plan, untouched = collect(root)
    counts = Counter(str(dest.parent.relative_to(root)) for _, dest, _ in plan)
    print(f'Root: {root}')
    print(f'Mode: {"APPLY" if args.apply else "DRY RUN — NO FILES WILL CHANGE"}')
    print(f'Found {len(plan)} matching files across '
          f'{len(set(src.parent for src, _, _ in plan))} anatomical directories.')
    print(f'Unrecognized/symlinked files left untouched: {untouched}')
    for folder, count in sorted(counts.items()):
        print(f'  {count:3d}  {folder}')
    if args.verbose:
        for source, dest, _ in plan:
            print(f'  {source} -> {dest}')
    if args.apply:
        apply(root, plan)
    else:
        print('\nDRY RUN ONLY. To organize, rerun with --apply.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        print('Stopped. No destination was overwritten. Keep any manifest:',
              'it can recover moves performed before an error.', file=sys.stderr)
        sys.exit(1)
