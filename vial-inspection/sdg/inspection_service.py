#!/usr/bin/env python3
"""Watch a folder and run persistent, confidence-gated vial inspection."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))

from infer_tray import (  # noqa: E402
    BUNDLE_SCHEMA,
    InspectionEngine,
    render_annotated,
    resolve_bundle,
    result_rows,
    sha256,
    write_html_report,
)


STATE_SCHEMA = "vial-inspection-service-state-1"
RESULTS_SCHEMA = "vial-inspection-service-results-1"
IMAGE_SUFFIXES = {".jpg", ".jpeg"}
ROUTE_ACTIONS = {
    "complete_order": "accept",
    "incomplete_order": "reject",
    "unresolved": "hold",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def replace_with_retry(source, destination, attempts=10):
    """Atomically replace a path despite brief Windows sharing violations."""
    delay = 0.01
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts-1:
                raise
            time.sleep(delay)
            delay = min(delay*2, 0.25)


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path, payload):
    atomic_write_text(path, json.dumps(payload, indent=2)+"\n")


def atomic_write_csv(path, rows):
    if not rows:
        atomic_write_text(path, "")
        return
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def discover_images(input_dir):
    return sorted(
        (path for path in Path(input_dir).iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.lower())


def route_action(verdict):
    try:
        return ROUTE_ACTIONS[verdict]
    except KeyError as exc:
        raise ValueError(f"unsupported tray verdict: {verdict}") from exc


def fresh_state(bundle_hash):
    now = utc_now()
    return {
        "schema_version": STATE_SCHEMA,
        "bundle_schema_version": BUNDLE_SCHEMA,
        "bundle_manifest_sha256": bundle_hash,
        "created_at": now,
        "updated_at": now,
        "processed": {},
        "failed": {},
    }


def load_state(path, bundle_hash):
    path = Path(path)
    if not path.exists():
        return fresh_state(bundle_hash)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError("unsupported inspection service state")
    if state.get("bundle_manifest_sha256") != bundle_hash:
        raise ValueError(
            "service output belongs to a different bundle; choose a new output directory")
    if not isinstance(state.get("processed"), dict) or not isinstance(
            state.get("failed"), dict):
        raise ValueError("inspection service state is malformed")
    return state


class InspectionService:
    def __init__(self, labeler, bundle_path, input_dir, output_dir,
                 good_expected_through=30, scrap_expected_through=None):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        if not self.input_dir.is_dir():
            raise ValueError(f"input directory does not exist: {self.input_dir}")
        if self.output_dir == self.input_dir:
            raise ValueError("input and output directories must differ")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir = self.output_dir/"records"
        self.errors_dir = self.output_dir/"errors"
        self.records_dir.mkdir(exist_ok=True)
        self.errors_dir.mkdir(exist_ok=True)
        self.good_expected_through = good_expected_through
        self.scrap_expected_through = scrap_expected_through

        self.bundle = resolve_bundle(bundle_path)
        self.bundle_hash = self.bundle["manifest_sha256"]
        self.state_path = self.output_dir/"state.json"
        self.health_path = self.output_dir/"health.json"
        self.state = load_state(self.state_path, self.bundle_hash)
        self.engine = InspectionEngine(
            labeler,
            self.bundle["calibration"],
            self.bundle["position_metadata"].parent,
            self.bundle["threshold_profile"],
            self.bundle["tray_metadata"].parent,
            self.bundle,
        )
        self.write_health("ready")

    def close(self, status="stopped"):
        self.engine.close()
        self.write_health(status)

    def write_state(self):
        self.state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, self.state)

    def write_health(self, status, current_file=None, message=None):
        verdicts = {name: 0 for name in (
            "complete_order", "incomplete_order", "unresolved")}
        for record in self.state["processed"].values():
            verdict = record.get("tray_verdict")
            if verdict in verdicts:
                verdicts[verdict] += 1
        payload = {
            "schema_version": "vial-inspection-service-health-1",
            "status": status,
            "updated_at": utc_now(),
            "bundle_schema_version": BUNDLE_SCHEMA,
            "bundle_manifest_sha256": self.bundle_hash,
            "processed": len(self.state["processed"]),
            "failed": len(self.state["failed"]),
            "verdicts": verdicts,
            "current_file": current_file,
            "message": message,
        }
        atomic_write_json(self.health_path, payload)

    def rebuild_dashboard(self):
        results = []
        for record in sorted(
                self.state["processed"].values(),
                key=lambda item: (item["filename"].lower(), item["source_sha256"])):
            path = self.output_dir/record["record_path"]/"result.json"
            if not path.is_file():
                raise ValueError(f"committed result is missing: {path}")
            results.append(json.loads(path.read_text(encoding="utf-8")))

        summaries, positions = [], []
        for result in results:
            summary, rows = result_rows(result)
            summaries.append(summary)
            positions.extend(rows)
        atomic_write_json(self.output_dir/"results.json", {
            "schema_version": RESULTS_SCHEMA,
            "bundle_manifest_sha256": self.bundle_hash,
            "warning": "Predictions are automated outputs, not human review.",
            "results": results,
        })
        atomic_write_csv(self.output_dir/"summary.csv", summaries)
        atomic_write_csv(self.output_dir/"positions.csv", positions)
        temporary_report = self.output_dir/f".report.{uuid.uuid4().hex}.tmp"
        try:
            write_html_report(results, temporary_report)
            replace_with_retry(temporary_report, self.output_dir/"report.html")
        finally:
            temporary_report.unlink(missing_ok=True)

    def process_image(self, path):
        path = Path(path).resolve()
        source_hash = sha256(path)
        source_key = f"{path.name}:{source_hash}"
        if source_key in self.state["processed"]:
            return "skipped"
        if source_key in self.state["failed"]:
            return "failed-previously"

        record_name = f"{path.stem}-{source_hash[:12]}"
        record_relative = Path("records")/record_name
        final_dir = self.output_dir/record_relative
        temporary_dir = self.output_dir/f".pending-{uuid.uuid4().hex}"
        started = time.perf_counter()
        self.write_health("processing", path.name)
        try:
            if final_dir.exists():
                result = json.loads((final_dir/"result.json").read_text(
                    encoding="utf-8"))
            else:
                temporary_dir.mkdir()
                result, image = self.engine.inspect(
                    path, self.good_expected_through,
                    self.scrap_expected_through)
                duration_ms = round((time.perf_counter()-started)*1000, 3)
                result["annotated_image"] = (
                    record_relative/"annotated.jpg").as_posix()
                result["service"] = {
                    "schema_version": "vial-inspection-service-record-1",
                    "processed_at": utc_now(),
                    "duration_ms": duration_ms,
                    "source_sha256": source_hash,
                    "bundle_manifest_sha256": self.bundle_hash,
                    "route_action": route_action(result["tray_verdict"]),
                }
                render_annotated(image, result, temporary_dir/"annotated.jpg")
                (temporary_dir/"result.json").write_text(
                    json.dumps(result, indent=2)+"\n", encoding="utf-8")
                summary, positions = result_rows(result)
                atomic_write_csv(temporary_dir/"summary.csv", [summary])
                atomic_write_csv(temporary_dir/"positions.csv", positions)
                replace_with_retry(temporary_dir, final_dir)

            duration_ms = result.get("service", {}).get(
                "duration_ms", round((time.perf_counter()-started)*1000, 3))
            self.state["processed"][source_key] = {
                "filename": path.name,
                "source_sha256": source_hash,
                "record_path": record_relative.as_posix(),
                "tray_verdict": result["tray_verdict"],
                "route_action": route_action(result["tray_verdict"]),
                "duration_ms": duration_ms,
            }
            self.state["failed"].pop(source_key, None)
            self.write_state()
            atomic_write_json(self.output_dir/"latest.json", {
                "schema_version": "vial-inspection-service-latest-1",
                "updated_at": utc_now(),
                "filename": path.name,
                "source_sha256": source_hash,
                "tray_verdict": result["tray_verdict"],
                "route_action": route_action(result["tray_verdict"]),
                "record_path": record_relative.as_posix(),
                "bundle_manifest_sha256": self.bundle_hash,
            })
            self.rebuild_dashboard()
            self.write_health("ready")
            return "processed"
        except Exception as exc:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            if source_key in self.state["processed"]:
                self.write_health("degraded", path.name,
                                  f"dashboard/state refresh failed: {exc}")
                return "processed"
            failure = {
                "filename": path.name,
                "source_sha256": source_hash,
                "failed_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self.state["failed"][source_key] = failure
            self.write_state()
            atomic_write_json(self.errors_dir/f"{record_name}.json", failure)
            self.write_health("degraded", path.name, str(exc))
            return "failed"

    def poll(self):
        counts = {"processed": 0, "skipped": 0, "failed": 0,
                  "failed-previously": 0}
        for path in discover_images(self.input_dir):
            outcome = self.process_image(path)
            counts[outcome] += 1
            print(f"SERVICE {outcome.upper()} {path.name}", flush=True)
        return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeler", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--once", action="store_true",
                        help="process current images and exit")
    parser.add_argument("--good-expected-through", type=int, default=30)
    parser.add_argument("--scrap-expected-through", type=int)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("poll-seconds must be positive")
    try:
        service = InspectionService(
            args.labeler, args.bundle, args.input, args.output,
            args.good_expected_through, args.scrap_expected_through)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    try:
        while True:
            service.poll()
            if args.once:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("SERVICE STOPPED keyboard interrupt", flush=True)
    finally:
        service.close()
    print(f"SERVICE VERIFIED processed={len(service.state['processed'])} "
          f"failed={len(service.state['failed'])} output={service.output_dir}")


if __name__ == "__main__":
    main()
