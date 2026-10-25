#!/usr/bin/env python3
"""Compare a COPY-created dataset with the real lerobot-lancedb converter output."""

import argparse
import importlib.metadata
import json
import subprocess
import tempfile
from pathlib import Path

from lerobot_lancedb.convert import convert


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def run(args, sql, error=None):
    result = subprocess.run(
        [
            str(args.duckdb),
            "-unsigned",
            "-json",
            "-c",
            f"LOAD {quote(args.extension)}; {sql}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if error is not None:
        assert result.returncode != 0 and error in result.stderr, (
            result.stdout,
            result.stderr,
        )
        return
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout) if result.stdout.strip() else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    args = parser.parse_args()
    assert importlib.metadata.version("lerobot-lancedb") == "0.3.1"
    with tempfile.TemporaryDirectory(prefix="lerobot-lance-") as tmp:
        base = Path(tmp)
        parquet, lance = base / "parquet", base / "lance's data"
        features = {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "counter": {"dtype": "int64", "shape": [1]},
        }
        run(
            args,
            f"COPY (SELECT (i // 3)::BIGINT AS episode_index, 'move' AS task, "
            '[i, i+1]::FLOAT[2] AS "observation.state", i::FLOAT AS action, '
            '(i % 2 = 0) AS "next.done", (9007199254740993 + i)::BIGINT AS counter '
            f"FROM range(6) t(i) ORDER BY i) TO {quote(parquet)} "
            f"(FORMAT lerobot, FPS 10, FEATURES {quote(json.dumps(features))});",
        )
        convert(lance, root=parquet)
        for function, order in (
            ("scan", '"index"'),
            ("episodes", "episode_index"),
            ("tasks", "task_index"),
            ("stats", "feature"),
        ):
            reference = run(
                args,
                f"SELECT * FROM lerobot_{function}({quote(parquet)}) ORDER BY {order};",
            )
            actual = run(
                args,
                f"SELECT * FROM lerobot_{function}({quote(lance)}) ORDER BY {order};",
            )
            assert actual == reference, function
        left = run(args, f"SELECT * FROM lerobot_info({quote(parquet)});")
        right = run(args, f"SELECT * EXCLUDE(storage_format) FROM lerobot_info({quote(lance)});")
        assert left == right
        for selection in ("[0]", "[1, 0, 1]", "[]"):
            query = "SELECT * FROM lerobot_scan({}, episode_indices := " + selection + ') ORDER BY "index";'
            assert run(args, query.format(quote(parquet))) == run(args, query.format(quote(lance)))
        schema_query = "DESCRIBE SELECT * FROM lerobot_scan({});"
        assert run(args, schema_query.format(quote(parquet))) == run(args, schema_query.format(quote(lance)))

        # Loading an installed extension still works when automatic installation/loading is disabled.
        settings = "SET autoinstall_known_extensions=false; SET autoload_known_extensions=false; "
        assert run(args, settings + f"SELECT count(*) AS n FROM lerobot_scan({quote(lance)});") == [{"n": 6}]
        missing = settings + f"SET extension_directory={quote(base / 'empty-extensions')}; "
        run(
            args,
            missing + f"SELECT * FROM lerobot_scan({quote(lance)});",
            "INSTALL lance; LOAD lance;",
        )
        failed_install = (
            missing
            + "SET autoinstall_known_extensions=true; "
            + f"SET autoinstall_extension_repository={quote(base / 'missing-repository')}; "
        )
        run(
            args,
            failed_install + f"SELECT * FROM lerobot_scan({quote(lance)});",
            "INSTALL lance; LOAD lance;",
        )
        run(
            args,
            missing + f"SELECT * FROM lerobot_video_routes({quote(lance)}, [0]);",
            "INSTALL lance; LOAD lance;",
        )
        # Official DuckDB 1.5.5 Lance can scan frames but lacks the Blob filesystem.
        for call in (
            f"lerobot_video_routes({quote(lance)}, [0])",
            f"lerobot_video_frames({quote(lance)}, [0])",
            f"lerobot_video_windows({quote(lance)}, [{{request_id: 1, episode_index: 0, frame_index: 0}}])",
            f"lerobot_video_targets({quote(lance)}, (SELECT 1 AS request_id, 0 AS episode_index, "
            "0 AS frame_index, 'camera' AS video_key, 0 AS delta_index))",
        ):
            run(
                args,
                f"SELECT * FROM {call};",
                "require a lance extension with lance_blob_uri",
            )
        # Temporal targets need only meta/, so a cold cache must not open frames.lance.
        temporal = (
            "SELECT target_frame_index FROM lerobot_temporal_targets({}, "
            "(SELECT 1 AS request_id, 0 AS episode_index, 1 AS frame_index, 0 AS delta_index));"
        )
        assert run(args, missing + temporal.format(quote(lance))) == run(args, temporal.format(quote(parquet)))
        # Plain v3 never requires Lance, even with no extension installation available.
        assert run(args, missing + f"SELECT count(*) AS n FROM lerobot_scan({quote(parquet)});") == [{"n": 6}]

        # A scoped Lance secret changes cache identity without exposing credentials in the cache key.
        warm = f"LOAD lance; SELECT count(*) FROM lerobot_scan({quote(lance)}); "
        secret = f"CREATE SECRET lance_fixture (TYPE LANCE, SCOPE {quote(lance)}, HF_TOKEN 'public-fixture'); "
        check = f"SELECT cached FROM lerobot_cache_info({quote(lance)}) WHERE component='data';"
        result = subprocess.run(
            [
                str(args.duckdb),
                "-unsigned",
                "-csv",
                "-noheader",
                "-c",
                f"LOAD {quote(args.extension)}; "
                + warm
                + secret
                + check
                + f"SELECT count(*) FROM lerobot_scan({quote(lance)}); "
                + secret.replace("CREATE SECRET", "CREATE OR REPLACE SECRET").replace(
                    "public-fixture", "public-fixture-2"
                )
                + check,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [
            "6",
            "true",
            "false",
            "6",
            "true",
            "false",
        ], result.stdout
        print(
            "PASS: converter parity (5 functions), schema, episode filters, optional loading, video rejection, metadata-only temporal targets, Lance secret cache isolation"
        )


if __name__ == "__main__":
    main()
