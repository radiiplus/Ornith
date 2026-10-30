"""Validate and stage a small unified diff before touching a source repository."""

from pathlib import Path
import os
import re
import subprocess


class PatchError(RuntimeError):
    pass


def validate_patch(patch: str, allowed_paths: set) -> tuple:
    if not patch or len(patch.encode()) > 32768:
        raise PatchError('Patch must be nonempty and at most 32 KiB')
    paths = []
    current = None
    for line in patch.splitlines():
        if line.startswith('diff --git '):
            match = re.fullmatch(r'diff --git a/([A-Za-z0-9_./-]+) b/([A-Za-z0-9_./-]+)', line)
            if not match or match.group(1) != match.group(2):
                raise PatchError('Patch contains an unsupported path or rename')
            path = match.group(1)
            if path.startswith('/') or '..' in Path(path).parts or path not in allowed_paths:
                raise PatchError(f'Patch path is not approved: {path}')
            paths.append(path)
            current = path
        elif line.startswith('--- ') and line[4:] not in ('/dev/null', 'a/' + str(current)):
            raise PatchError('Patch source header disagrees with approved path')
        elif line.startswith('+++ ') and line[4:] != 'b/' + str(current):
            raise PatchError('Patch target header disagrees with approved path')
        elif line.startswith(('rename from ', 'rename to ', 'copy from ', 'copy to ',
                              'GIT binary patch', 'old mode ', 'new mode ')):
            raise PatchError('Renames, binary patches and mode changes are not permitted')
        elif line.startswith('new file mode ') and line != 'new file mode 100644':
            raise PatchError('New files must be nonexecutable')
    if not paths or len(set(paths)) > 5:
        raise PatchError('Patch must change between one and five approved files')
    return tuple(dict.fromkeys(paths))


def apply_patch(repository: Path, patch: str, allowed_paths: set) -> tuple:
    paths = validate_patch(patch, allowed_paths)
    root = Path(repository).resolve(strict=True)
    for path in paths:
        target = root / path
        if target.is_symlink() or any(parent.is_symlink() for parent in target.parents if parent != root):
            raise PatchError('Patch cannot target symlinks')
        if not str(target.resolve()).startswith(str(root) + '/'):
            raise PatchError('Patch escapes the repository')
    for flags in (('--check',), ()):
        environment = os.environ.copy()
        environment.pop('GIT_DIR', None)
        environment.pop('GIT_WORK_TREE', None)
        environment['GIT_CEILING_DIRECTORIES'] = str(root.parent)
        completed = subprocess.run(['git', 'apply', *flags, '-'], input=patch, text=True,
                                   cwd=root, env=environment, capture_output=True, timeout=10)
        if completed.returncode:
            raise PatchError('Patch did not apply cleanly: ' + completed.stderr[:1000])
    return paths


def apply_edits(repository: Path, edits: list, allowed_paths: set) -> tuple:
    if not isinstance(edits, list) or not 1 <= len(edits) <= 5:
        raise PatchError('Provide one to five exact-text edits')
    root = Path(repository).resolve(strict=True)
    pending = {}
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {'path', 'old', 'new'}:
            raise PatchError('Each edit requires path, old and new strings')
        name, old, new = edit['path'], edit['old'], edit['new']
        if (not isinstance(name, str) or name not in allowed_paths or
                not isinstance(old, str) or not isinstance(new, str) or
                not old or old == new or len(old) + len(new) > 8192):
            raise PatchError('Invalid or unapproved exact-text edit')
        target = root / name
        if target.is_symlink() or not target.is_file() or not str(target.resolve()).startswith(str(root) + '/'):
            raise PatchError(f'Edit path is unsafe: {name}')
        content = pending.get(target, target.read_text())
        if content.count(old) != 1:
            raise PatchError(f'Edit source must occur exactly once: {name}')
        pending[target] = content.replace(old, new, 1)
    for target, content in pending.items():
        target.write_text(content)
    return tuple(str(path.relative_to(root)) for path in pending)
