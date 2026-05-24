from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CRITICAL_FAILURES: list[str] = []
WARNINGS: list[str] = []


def _print_result(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"{status}: {name}{suffix}")


def _critical(name: str, condition: bool, detail: str = "") -> None:
    _print_result(name, condition, detail)
    if not condition:
        CRITICAL_FAILURES.append(name)


def _warning(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "WARN"
    suffix = f" - {detail}" if detail else ""
    print(f"{status}: {name}{suffix}")
    if not condition:
        WARNINGS.append(name)


def check_readme_repo_url(text: str) -> None:
    _critical("README clone URL uses lsd-seg", "https://github.com/Macs-Laboratory/lsd-seg.git" in text)
    _critical(
        "README install command does not use old repo name",
        "MICCAI2026-Dynamic-Sub-domain-Modeling" not in text,
    )


def check_citation(text: str) -> None:
    match = re.search(r"```bibtex\s*(.*?)```", text, flags=re.DOTALL)
    _critical("README citation block exists", match is not None)
    if match is None:
        return
    citation = match.group(1).strip()
    _critical("README citation block is not empty", bool(citation))
    _critical("README citation contains lee2026lsdseg placeholder", "lee2026lsdseg" in citation)


def check_images() -> None:
    for relative in ["assets/overview.png", "assets/main_results.png", "assets/mechanism_stability.png"]:
        _critical(f"README image exists: {relative}", (ROOT / relative).exists())


def check_local_links(text: str) -> None:
    missing: list[str] = []
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        clean = target.split("#", 1)[0]
        if clean and not (ROOT / clean).exists():
            missing.append(target)
    for target in re.findall(r'<img[^>]+src="([^"]+)"', text):
        if not (ROOT / target).exists():
            missing.append(target)
    _critical("README local links resolve", not missing, ", ".join(missing[:5]))


def check_suspicious_script_values() -> None:
    suspicious_patterns = [
        "dice = np.array(",
        '"dice": 0.',
        '"routing_entropy": 0.',
    ]
    for relative in ["scripts/prompt_sensitivity.py", "scripts/analyze_results.py"]:
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        suspicious = [pattern for pattern in suspicious_patterns if pattern in text]
        dataframe_parameter_rows = "pd.DataFrame([" in text and '"parameter":' in text
        _warning(f"No obvious synthetic metric constants in {relative}", not suspicious and not dataframe_parameter_rows)


def main() -> int:
    text = README.read_text(encoding="utf-8")
    check_readme_repo_url(text)
    check_citation(text)
    check_images()
    check_local_links(text)
    check_suspicious_script_values()
    if CRITICAL_FAILURES:
        print("\nCritical validation failures:")
        for item in CRITICAL_FAILURES:
            print(f"- {item}")
        return 1
    if WARNINGS:
        print("\nWarnings:")
        for item in WARNINGS:
            print(f"- {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
