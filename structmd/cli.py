"""Click-based command line interface for structmd.

Supported forms::

    structmd input.pdf -o output.md
    structmd input.pdf --json extraction.json --md output.md
    structmd input.pdf --model qwen2-vl:2b --workers 8 --dpi 200
    structmd --from-json extraction.json -o output.md
    structmd batch doc1.pdf doc2.docx doc3.png -o output_dir/
    structmd input.pdf --pages 1,3,5-10 -o output.md
    structmd --config ~/.structmd.yaml input.pdf -o out.md

``batch`` is dispatched manually rather than as a Click subcommand so that
options may follow the positional input path (e.g. ``structmd in.pdf -o o.md``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import click

from structmd.batch.processor import collect_input_files
from structmd.config import load_config
from structmd.core import StructMDError
from structmd.pipeline import StructMDPipeline

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


def parse_pages(spec: str) -> List[int]:
    """Parse ``1,3,5-10`` style page selections into a sorted unique list."""
    pages: set = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            try:
                start, end = int(start_s), int(end_s)
            except ValueError:
                raise click.BadParameter(f"Invalid page range {chunk!r} in {spec!r}") from None
            if start < 1 or end < start:
                raise click.BadParameter(f"Invalid page range {chunk!r}")
            pages.update(range(start, end + 1))
        else:
            try:
                page = int(chunk)
            except ValueError:
                raise click.BadParameter(f"Invalid page number {chunk!r}") from None
            if page < 1:
                raise click.BadParameter(f"Page numbers must be >= 1, got {page}")
            pages.add(page)
    if not pages:
        raise click.BadParameter(f"No pages found in {spec!r}")
    return sorted(pages)


def _apply_cli_overrides(config, model, url, workers, dpi, page_numbers, merge) -> None:
    """CLI flags beat config files and env vars."""
    if model:
        config.ollama_model = model
    if url:
        config.ollama_url = url
    if workers is not None:
        config.ollama_max_workers = workers
    if dpi is not None:
        config.dpi = dpi
    if page_numbers is False:
        config.include_page_numbers = False
    if merge is False:
        config.merge_continued_paragraphs = False


@click.command("structmd")
@click.version_option(package_name="structmd", prog_name="structmd")
@click.argument("tokens", nargs=-1, type=click.Path())
@click.option("-o", "--output", type=click.Path(), help="Output Markdown file path.")
@click.option("--json", "output_json", type=click.Path(), help="Save intermediate extraction JSON.")
@click.option("--md", "output_md", type=click.Path(), help="Output Markdown file (alias of -o).")
@click.option(
    "--from-json",
    is_flag=True,
    default=False,
    help="Treat INPUT as extraction JSON; skip VLM extraction entirely.",
)
@click.option("--model", default=None, help="Ollama model name (e.g. qwen2-vl:2b).")
@click.option("--url", default=None, help="Ollama base URL.")
@click.option("--workers", type=int, default=None, help="Async worker count.")
@click.option("--dpi", type=int, default=None, help="Rendering DPI.")
@click.option("--pages", default=None, help="Pages to process, e.g. 1,3,5-10.")
@click.option("--no-page-numbers", is_flag=True, default=False, help="Suppress page markers.")
@click.option(
    "--no-merge",
    is_flag=True,
    default=False,
    help="Do not merge paragraphs continued across pages.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-extract even if cached results exist (cache is refreshed).",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable the cache entirely for this run.",
)
@click.option(
    "--save-assets",
    is_flag=True,
    default=False,
    help="Crop figure regions out of PDFs as PNGs and link them in the Markdown.",
)
@click.option(
    "--config", "config_path", type=click.Path(), default=None, help="Path to config YAML."
)
@click.option(
    "-v", "--verbose", is_flag=True, default=False, help="DEBUG logging incl. raw VLM I/O."
)
def main(
    tokens: Tuple[str, ...],
    output: Optional[str],
    output_json: Optional[str],
    output_md: Optional[str],
    from_json: bool,
    model: Optional[str],
    url: Optional[str],
    workers: Optional[int],
    dpi: Optional[int],
    pages: Optional[str],
    no_page_numbers: bool,
    no_merge: bool,
    force: bool,
    no_cache: bool,
    save_assets: bool,
    config_path: Optional[str],
    verbose: bool,
) -> None:
    """Convert PDFs, Office documents, and images to structured Markdown via Ollama VLMs.

    \b
    Examples:
      structmd input.pdf -o output.md
      structmd input.pdf --json extraction.json --md output.md
      structmd --from-json extraction.json -o output.md
      structmd input.pdf --pages 1,3,5-10 -o output.md
      structmd batch doc1.pdf doc2.docx doc3.png -o output_dir/
    """
    _setup_logging(verbose)
    paths = list(tokens)

    if not paths:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        sys.exit(2)

    if paths[0] == "batch":
        _run_batch(
            paths[1:],
            output,
            model,
            url,
            workers,
            dpi,
            config_path,
            verbose,
            pages_spec=pages,
            force=force,
            no_cache=no_cache,
            save_assets=save_assets,
        )
        return

    _run_convert(
        input_path=paths[0],
        extra_paths=paths[1:],
        output=output,
        output_json=output_json,
        output_md=output_md,
        from_json=from_json,
        model=model,
        url=url,
        workers=workers,
        dpi=dpi,
        pages=pages,
        no_page_numbers=no_page_numbers,
        no_merge=no_merge,
        force=force,
        no_cache=no_cache,
        save_assets=save_assets,
        config_path=config_path,
    )


def _run_convert(
    input_path: str,
    extra_paths: List[str],
    output: Optional[str],
    output_json: Optional[str],
    output_md: Optional[str],
    from_json: bool,
    model: Optional[str],
    url: Optional[str],
    workers: Optional[int],
    dpi: Optional[int],
    pages: Optional[str],
    no_page_numbers: bool,
    no_merge: bool,
    force: bool = False,
    no_cache: bool = False,
    save_assets: bool = False,
    config_path: Optional[str] = None,
) -> None:
    if extra_paths:
        raise click.ClickException(
            f"Unexpected extra arguments: {' '.join(extra_paths)}. "
            "Use `structmd batch ...` for multiple documents."
        )

    config = load_config(config_path)
    _apply_cli_overrides(config, model, url, workers, dpi, not no_page_numbers, not no_merge)
    if no_cache:
        config.cache_enabled = False
    if save_assets:
        config.save_assets = True

    final_md = output_md or output
    page_list: Optional[List[int]] = parse_pages(pages) if pages else None

    try:
        if from_json:
            if not Path(input_path).is_file():
                raise click.ClickException(f"JSON file not found: {input_path}")
            pipeline = StructMDPipeline(config)
            markdown = pipeline.build_from_json(input_path, output_md=final_md)
        else:
            if not Path(input_path).is_file():
                raise click.ClickException(f"Input file not found: {input_path}")
            with StructMDPipeline(config) as pipeline:
                markdown = pipeline.process(
                    input_path,
                    output_json=output_json,
                    output_md=final_md,
                    pages=page_list,
                    force=force,
                )
    except StructMDError as exc:
        raise click.ClickException(str(exc)) from exc
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if final_md:
        click.echo(f"Wrote {final_md}")
    else:
        click.echo(markdown.content)


def _run_batch(
    paths: List[str],
    output_dir: Optional[str],
    model: Optional[str],
    url: Optional[str],
    workers: Optional[int],
    dpi: Optional[int],
    config_path: Optional[str],
    verbose: bool,
    pages_spec: Optional[str] = None,
    force: bool = False,
    no_cache: bool = False,
    save_assets: bool = False,
) -> None:
    if not paths:
        raise click.ClickException("batch requires at least one input file")

    files, missing = collect_input_files(paths)
    if missing:
        raise click.ClickException(f"Input files not found: {', '.join(missing)}")
    if not files:
        raise click.ClickException(
            "No convertible files found in: "
            f"{', '.join(paths)}. Supported: pdf, docx, pptx, xlsx, odt, ods, "
            "odp, png, jpg, jpeg, webp, tiff, bmp"
        )

    config = load_config(config_path)
    _apply_cli_overrides(config, model, url, workers, dpi, True, True)
    if no_cache:
        config.cache_enabled = False
    if save_assets:
        config.save_assets = True
    page_list = parse_pages(pages_spec) if pages_spec else None

    try:
        with StructMDPipeline(config) as pipeline:
            documents = pipeline.process_batch(
                files, pages=page_list, force=force, output_dir=output_dir
            )
    except StructMDError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_dir:
        out_path = Path(output_dir).expanduser()
        out_path.mkdir(parents=True, exist_ok=True)
        for document in documents:
            stem = Path(document.metadata.get("source_path") or "document").stem
            target = out_path / f"{stem}.md"
            document.save(str(target))
            click.echo(f"Wrote {target}")
    else:
        for document in documents:
            click.echo(document.content)


if __name__ == "__main__":  # pragma: no cover
    main()
