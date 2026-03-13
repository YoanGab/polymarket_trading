from __future__ import annotations

import csv
import tarfile
import time
from collections.abc import Iterable
from pathlib import Path

import httpx
import polars as pl

ARCHIVE_URL = "https://polydata-archive.s3.us-east-1.amazonaws.com/archive.tar.xz"
ARCHIVE_FILENAME = "archive.tar.xz"
EXTRACTION_SENTINEL = ".extracted.ok"

TRADE_COLUMNS = ("timestamp", "market_id", "price", "usd_amount", "token_amount", "direction")
MARKET_COLUMNS = ("market_id", "question", "volume", "closedTime")

TRADE_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "ts", "time", "datetime", "created_at", "createdat"),
    "market_id": ("market_id", "market", "condition_id", "conditionid", "marketid"),
    "price": ("price", "trade_price", "execution_price"),
    "usd_amount": ("usd_amount", "amount_usd", "notional_usd", "size_usd", "usd"),
    "token_amount": ("token_amount", "shares", "amount", "quantity", "size", "token_size"),
    "direction": ("direction", "side", "trade_side", "tradedirection"),
}
MARKET_ALIASES: dict[str, tuple[str, ...]] = {
    "market_id": ("condition_id", "conditionid", "market_id", "marketid"),
    "question": ("question", "title", "market_question"),
    "volume": ("volume", "volume_usd", "total_volume"),
    "closedTime": ("closedtime", "closed_time", "closetime", "end_time"),
}


def download_archive(dest_dir: Path, *, chunk_size: int = 8192) -> Path:
    """Download the warproxxx/poly_data archive into ``dest_dir``."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / ARCHIVE_FILENAME
    partial_path = archive_path.with_suffix(f"{archive_path.suffix}.part")

    with _build_http_client() as client:
        expected_size = _get_remote_file_size(client, ARCHIVE_URL)
        if archive_path.exists():
            local_size = archive_path.stat().st_size
            if expected_size is not None and local_size == expected_size:
                print(f"Using existing archive at {archive_path} ({_format_bytes(local_size)}).", flush=True)
                return archive_path
            if expected_size is None and local_size > 0:
                print(
                    "Using existing archive because the remote file size could not be determined: "
                    f"{archive_path} ({_format_bytes(local_size)}).",
                    flush=True,
                )
                return archive_path
            print(
                f"Existing archive size mismatch for {archive_path}; downloading a fresh copy.",
                flush=True,
            )

        partial_path.unlink(missing_ok=True)
        with client.stream("GET", ARCHIVE_URL) as response:
            response.raise_for_status()
            total_bytes = expected_size or _parse_content_length(response.headers.get("Content-Length"))
            downloaded_bytes = 0
            started_at = time.monotonic()
            last_report_at = started_at

            with partial_path.open("wb") as output:
                for chunk in response.iter_bytes(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    output.write(chunk)
                    downloaded_bytes += len(chunk)
                    now = time.monotonic()
                    if now - last_report_at >= 1.0:
                        _print_progress(downloaded_bytes, total_bytes, started_at)
                        last_report_at = now

            _print_progress(downloaded_bytes, total_bytes, started_at, force=True)

        if total_bytes is not None and downloaded_bytes != total_bytes:
            partial_path.unlink(missing_ok=True)
            raise OSError(
                f"Downloaded archive size does not match the expected size: {downloaded_bytes} != {total_bytes}"
            )

    partial_path.replace(archive_path)
    return archive_path


def extract_archive(archive_path: Path, dest_dir: Path) -> Path:
    """Extract ``archive_path`` into a deterministic directory beneath ``dest_dir``."""

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = dest_dir / archive_path.name.removesuffix(".tar.xz")
    sentinel_path = extracted_dir / EXTRACTION_SENTINEL

    if sentinel_path.exists():
        print(f"Using existing extracted archive at {extracted_dir}.", flush=True)
        return extracted_dir

    extracted_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path.name} into {extracted_dir}...", flush=True)
    with tarfile.open(archive_path, mode="r:xz") as archive:
        archive.extractall(extracted_dir, filter="data")

    sentinel_path.write_text("ok\n", encoding="utf-8")
    return extracted_dir


def parse_trades(extracted_dir: Path) -> pl.DataFrame:
    """Parse trade CSVs from an extracted archive."""

    _ensure_directory(extracted_dir)

    frames: list[pl.DataFrame] = []
    for csv_path in _iter_csv_files(extracted_dir):
        column_map = _match_columns(_read_csv_header(csv_path), TRADE_ALIASES)
        if not _is_trade_file(column_map):
            continue
        frame = _read_csv_subset(csv_path, column_map, TRADE_COLUMNS)
        if frame.is_empty():
            continue
        frame = frame.with_columns(
            pl.col("timestamp").cast(pl.String, strict=False),
            pl.col("market_id").cast(pl.String, strict=False),
            pl.col("price").cast(pl.Float64, strict=False),
            pl.col("usd_amount").cast(pl.Float64, strict=False),
            pl.col("token_amount").cast(pl.Float64, strict=False),
            pl.col("direction").cast(pl.String, strict=False).str.strip_chars().str.to_lowercase(),
        ).with_columns(
            pl.when(
                pl.col("token_amount").is_null()
                & pl.col("usd_amount").is_not_null()
                & pl.col("price").is_not_null()
                & (pl.col("price") != 0.0)
            )
            .then(pl.col("usd_amount") / pl.col("price"))
            .otherwise(pl.col("token_amount"))
            .alias("token_amount")
        )
        frame = frame.filter(
            pl.col("timestamp").is_not_null() & pl.col("market_id").is_not_null() & pl.col("price").is_not_null()
        )
        if not frame.is_empty():
            frames.append(frame.select(TRADE_COLUMNS))

    if not frames:
        return _empty_trades_frame()

    return (
        pl.concat(frames, how="vertical_relaxed")
        .sort(["timestamp", "market_id"], nulls_last=True)
        .select(TRADE_COLUMNS)
    )


def parse_markets(extracted_dir: Path) -> pl.DataFrame:
    """Parse market metadata CSVs from an extracted archive."""

    _ensure_directory(extracted_dir)

    frames: list[pl.DataFrame] = []
    for csv_path in _iter_csv_files(extracted_dir):
        column_map = _match_columns(_read_csv_header(csv_path), MARKET_ALIASES)
        if not _is_market_file(column_map):
            continue
        frame = _read_csv_subset(csv_path, column_map, MARKET_COLUMNS)
        if frame.is_empty():
            continue
        frame = (
            frame.with_columns(
                pl.col("market_id").cast(pl.String, strict=False),
                pl.col("question").cast(pl.String, strict=False),
                pl.col("volume").cast(pl.Float64, strict=False),
                pl.col("closedTime").cast(pl.String, strict=False),
            )
            .filter(pl.col("market_id").is_not_null())
            .select(MARKET_COLUMNS)
        )
        if not frame.is_empty():
            frames.append(frame)

    if not frames:
        return _empty_markets_frame()

    return (
        pl.concat(frames, how="vertical_relaxed")
        .unique(subset=["market_id"], keep="last", maintain_order=True)
        .sort("market_id", nulls_last=True)
        .select(MARKET_COLUMNS)
    )


def _build_http_client() -> httpx.Client:
    return httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(connect=30.0, read=30.0, write=30.0, pool=30.0),
    )


def _ensure_directory(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory: {path}")


def _get_remote_file_size(client: httpx.Client, url: str) -> int | None:
    try:
        response = client.head(url)
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    return _parse_content_length(response.headers.get("Content-Length"))


def _parse_content_length(raw_value: str | None) -> int | None:
    if raw_value is None:
        return None
    try:
        parsed = int(raw_value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _print_progress(downloaded_bytes: int, total_bytes: int | None, started_at: float, *, force: bool = False) -> None:
    if not force and downloaded_bytes == 0:
        return

    elapsed = max(time.monotonic() - started_at, 0.001)
    rate = downloaded_bytes / elapsed
    if total_bytes:
        percent = (downloaded_bytes / total_bytes) * 100
        print(
            f"Downloaded {percent:5.1f}% ({_format_bytes(downloaded_bytes)} / {_format_bytes(total_bytes)}) "
            f"at {_format_bytes(rate)}/s",
            flush=True,
        )
        return

    print(
        f"Downloaded {_format_bytes(downloaded_bytes)} at {_format_bytes(rate)}/s",
        flush=True,
    )


def _format_bytes(size: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _iter_csv_files(root: Path) -> list[Path]:
    csv_files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"]
    csv_files.sort()
    return csv_files


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return next(csv.reader(handle), [])


def _match_columns(header: Iterable[str], aliases: dict[str, tuple[str, ...]]) -> dict[str, str]:
    normalized_to_actual = {
        _normalize_column_name(column_name): column_name
        for column_name in header
        if column_name and _normalize_column_name(column_name)
    }
    matched: dict[str, str] = {}
    for canonical_name, options in aliases.items():
        for option in options:
            actual_name = normalized_to_actual.get(_normalize_column_name(option))
            if actual_name is not None:
                matched[canonical_name] = actual_name
                break
    return matched


def _normalize_column_name(name: str) -> str:
    return "".join(character for character in name.strip().lower() if character.isalnum())


def _is_trade_file(column_map: dict[str, str]) -> bool:
    return all(field in column_map for field in ("timestamp", "market_id", "price")) and any(
        field in column_map for field in ("usd_amount", "token_amount", "direction")
    )


def _is_market_file(column_map: dict[str, str]) -> bool:
    return "market_id" in column_map and any(field in column_map for field in ("question", "volume", "closedTime"))


def _read_csv_subset(path: Path, column_map: dict[str, str], expected_columns: tuple[str, ...]) -> pl.DataFrame:
    selected_columns = list(dict.fromkeys(column_map.values()))
    try:
        frame = pl.read_csv(
            path,
            columns=selected_columns,
            encoding="utf8-lossy",
            ignore_errors=True,
            infer_schema_length=1000,
            low_memory=True,
            truncate_ragged_lines=True,
        )
    except pl.exceptions.NoDataError:
        return pl.DataFrame(schema=_schema_for(expected_columns))

    rename_map = {
        actual_name: canonical_name
        for canonical_name, actual_name in column_map.items()
        if actual_name in frame.columns and actual_name != canonical_name
    }
    if rename_map:
        frame = frame.rename(rename_map)

    missing_columns = [column_name for column_name in expected_columns if column_name not in frame.columns]
    if missing_columns:
        frame = frame.with_columns(
            [
                pl.lit(None, dtype=_schema_for((column_name,))[column_name]).alias(column_name)
                for column_name in missing_columns
            ]
        )

    return frame.select(expected_columns)


def _empty_trades_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_schema_for(TRADE_COLUMNS))


def _empty_markets_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=_schema_for(MARKET_COLUMNS))


def _schema_for(columns: tuple[str, ...]) -> dict[str, pl.DataType]:
    type_map: dict[str, pl.DataType] = {
        "timestamp": pl.String,
        "market_id": pl.String,
        "price": pl.Float64,
        "usd_amount": pl.Float64,
        "token_amount": pl.Float64,
        "direction": pl.String,
        "question": pl.String,
        "volume": pl.Float64,
        "closedTime": pl.String,
    }
    return {column_name: type_map[column_name] for column_name in columns}


__all__ = [
    "download_archive",
    "extract_archive",
    "parse_markets",
    "parse_trades",
]
