from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _resize_for_readme(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    scale = max_width / float(image.width)
    size = (max_width, max(1, int(round(image.height * scale))))
    return image.resize(size, Image.Resampling.LANCZOS)


def _copy_or_convert(source: Path, destination: Path, max_width: int, dpi: int) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        with Image.open(source) as image:
            converted = _resize_for_readme(image.convert("RGBA"), max_width=max_width)
            converted.save(destination, format="PNG", dpi=(dpi, dpi), optimize=True)
        return

    if suffix == ".pdf":
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise RuntimeError(
                "PDF conversion requires pypdfium2 or manual export. "
                "Please export the PDF figure to PNG and rerun."
            ) from exc
        document = pdfium.PdfDocument(str(source))
        page = document[0]
        scale = dpi / 72.0
        bitmap = page.render(scale=scale).to_pil()
        converted = _resize_for_readme(bitmap.convert("RGBA"), max_width=max_width)
        converted.save(destination, format="PNG", dpi=(dpi, dpi), optimize=True)
        return

    raise RuntimeError(f"Unsupported asset format: {source.suffix}")


def _resolve_source(latex_dir: Path, relative_path: str) -> Path:
    source_path = (latex_dir / relative_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Requested source file does not exist: {source_path}")
    return source_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy selected LaTeX figures into repo-tracked README assets.")
    parser.add_argument("--latex-dir", required=True, help="Path to the local LaTeX project directory.")
    parser.add_argument("--overview", required=True, help="Relative path to the overview figure inside the LaTeX directory.")
    parser.add_argument("--main-results", required=True, help="Relative path to the main results figure inside the LaTeX directory.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG export DPI for PDF conversion.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    assets_dir = repo_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    latex_dir = Path(args.latex_dir).expanduser().resolve()
    overview_source = _resolve_source(latex_dir, args.overview)
    main_results_source = _resolve_source(latex_dir, args.main_results)

    overview_output = assets_dir / "overview.png"
    main_results_output = assets_dir / "main_results.png"

    _copy_or_convert(overview_source, overview_output, max_width=1800, dpi=args.dpi)
    _copy_or_convert(main_results_source, main_results_output, max_width=2200, dpi=args.dpi)

    print(f"overview -> {overview_output}")
    print(f"main_results -> {main_results_output}")


if __name__ == "__main__":
    main()
