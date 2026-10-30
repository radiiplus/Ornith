"""Read a report from a workspace directory."""

from pathlib import Path
import sys


def read_report(workspace, name):
    base = Path(workspace).resolve()
    report = (base / name).resolve()
    assert report.is_relative_to(base), 'report is outside workspace'
    return report.read_text()


if __name__ == '__main__':
    print(read_report(sys.argv[1], sys.argv[2]))
