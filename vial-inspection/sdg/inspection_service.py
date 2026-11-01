#!/usr/bin/env python3
"""Watch a folder and run persistent, confidence-gated vial inspection."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
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


STATE_SCHEMA = "vial-inspection-service-state-2"
RESULTS_SCHEMA = "vial-inspection-service-results-2"
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


def session_decision(frames, minimum_frames):
    """Apply conservative multi-frame consensus to a single physical tray."""
    if any(frame["tray_verdict"] == "incomplete_order" for frame in frames):
        return "incomplete_order"
    if (len(frames) >= minimum_frames and
            all(frame["tray_model_verdict"] == "complete_order"
                for frame in frames) and
            any(frame["tray_verdict"] == "complete_order" for frame in frames)):
        return "complete_order"
    return "unresolved"


def session_policy(enabled=False, minimum_frames=3, delimiter="__"):
    return {
        "enabled": bool(enabled),
        "minimum_frames": int(minimum_frames),
        "filename_delimiter": delimiter,
    }


def tray_id_from_path(path, policy):
    if not policy["enabled"]:
        return None
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != "vial-inspection-frame-1":
            raise ValueError(f"unsupported frame sidecar: {sidecar.name}")
        tray_id = metadata.get("tray_id")
    else:
        delimiter = policy["filename_delimiter"]
        if not delimiter or delimiter not in path.stem:
            raise ValueError(
                f"session confirmation requires {delimiter!r} in filename "
                "or a vial-inspection-frame-1 sidecar")
        tray_id = path.stem.split(delimiter, 1)[0]
    if not isinstance(tray_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", tray_id):
        raise ValueError(f"invalid tray_id for {path.name}")
    return tray_id


def fresh_state(bundle_hash, policy=None):
    now = utc_now()
    policy = policy or session_policy()
    return {
        "schema_version": STATE_SCHEMA,
        "bundle_schema_version": BUNDLE_SCHEMA,
        "bundle_manifest_sha256": bundle_hash,
        "created_at": now,
        "updated_at": now,
        "session_policy": policy,
        "processed": {},
        "failed": {},
        "ignored": {},
        "sessions": {},
    }


def load_state(path, bundle_hash, policy=None):
    path = Path(path)
    policy = policy or session_policy()
    if not path.exists():
        return fresh_state(bundle_hash, policy)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError("unsupported inspection service state")
    if state.get("bundle_manifest_sha256") != bundle_hash:
        raise ValueError(
            "service output belongs to a different bundle; choose a new output directory")
    if state.get("session_policy") != policy:
        raise ValueError(
            "service output uses a different session policy; choose a new output directory")
    if any(not isinstance(state.get(name), dict)
           for name in ("processed", "failed", "ignored", "sessions")):
        raise ValueError("inspection service state is malformed")
    return state


class InspectionService:
    def __init__(self, labeler, bundle_path, input_dir, output_dir,
                 good_expected_through=30, scrap_expected_through=None,
                 confirm_sessions=False, minimum_session_frames=3,
                 session_delimiter="__"):
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
        if minimum_session_frames < 2:
            raise ValueError("minimum-session-frames must be at least 2")
        self.session_policy = session_policy(
            confirm_sessions, minimum_session_frames, session_delimiter)

        self.bundle = resolve_bundle(bundle_path)
        self.bundle_hash = self.bundle["manifest_sha256"]
        self.state_path = self.output_dir/"state.json"
        self.health_path = self.output_dir/"health.json"
        self.state = load_state(
            self.state_path, self.bundle_hash, self.session_policy)
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
            verdict = record.get("frame_tray_verdict")
            if verdict in verdicts:
                verdicts[verdict] += 1
        session_verdicts = {name: 0 for name in verdicts}
        for session in self.state["sessions"].values():
            verdict = session.get("tray_verdict")
            if verdict in session_verdicts:
                session_verdicts[verdict] += 1
        payload = {
            "schema_version": "vial-inspection-service-health-2",
            "status": status,
            "updated_at": utc_now(),
            "bundle_schema_version": BUNDLE_SCHEMA,
            "bundle_manifest_sha256": self.bundle_hash,
            "processed": len(self.state["processed"]),
            "failed": len(self.state["failed"]),
            "ignored_after_session_finalization": len(self.state["ignored"]),
            "frame_verdicts": verdicts,
            "session_verdicts": session_verdicts,
            "session_policy": self.session_policy,
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
        sessions = sorted(
            self.state["sessions"].values(), key=lambda item: item["tray_id"])
        atomic_write_json(self.output_dir/"sessions.json", {
            "schema_version": "vial-inspection-sessions-1",
            "session_policy": self.session_policy,
            "sessions": sessions,
        })
        atomic_write_csv(self.output_dir/"sessions.csv", [{
            "tray_id": session["tray_id"],
            "frame_count": session["frame_count"],
            "tray_verdict": session["tray_verdict"],
            "route_action": session["route_action"],
            "finalized": session["finalized"],
            "latest_record_path": session["latest_record_path"],
            "updated_at": session["updated_at"],
        } for session in sessions])
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
        if source_key in self.state["ignored"]:
            return "session-finalized"

        record_name = f"{path.stem}-{source_hash[:12]}"
        record_relative = Path("records")/record_name
        final_dir = self.output_dir/record_relative
        temporary_dir = self.output_dir/f".pending-{uuid.uuid4().hex}"
        started = time.perf_counter()
        try:
            tray_id = tray_id_from_path(path, self.session_policy)
            previous_session = (self.state["sessions"].get(tray_id)
                                if tray_id is not None else None)
            if previous_session and previous_session.get("finalized"):
                self.state["ignored"][source_key] = {
                    "filename": path.name,
                    "source_sha256": source_hash,
                    "tray_id": tray_id,
                    "reason": "session already finalized",
                    "ignored_at": utc_now(),
                }
                self.write_state()
                self.write_health("ready")
                return "session-finalized"

            self.write_health("processing", path.name)
            if final_dir.exists():
                result = json.loads((final_dir/"result.json").read_text(
                    encoding="utf-8"))
                service_record = result["service"]
                session_verdict = service_record.get(
                    "session_tray_verdict", result["tray_verdict"])
                session_action = service_record.get(
                    "route_action", route_action(session_verdict))
                frames = list((previous_session or {}).get("frames", []))
                frame_record = {
                    "source_key": source_key,
                    "filename": path.name,
                    "tray_verdict": result["tray_verdict"],
                    "tray_model_verdict": result.get(
                        "tray_model_verdict", result["tray_verdict"]),
                    "complete_margin": result.get(
                        "tray_confidence_gate", {}).get("complete_margin"),
                }
                if not any(item["source_key"] == source_key for item in frames):
                    frames.append(frame_record)
            else:
                temporary_dir.mkdir()
                result, image = self.engine.inspect(
                    path, self.good_expected_through,
                    self.scrap_expected_through)
                duration_ms = round((time.perf_counter()-started)*1000, 3)
                result["annotated_image"] = (
                    record_relative/"annotated.jpg").as_posix()
                frame_record = {
                    "source_key": source_key,
                    "filename": path.name,
                    "tray_verdict": result["tray_verdict"],
                    "tray_model_verdict": result.get(
                        "tray_model_verdict", result["tray_verdict"]),
                    "complete_margin": result.get(
                        "tray_confidence_gate", {}).get("complete_margin"),
                }
                frames = list((previous_session or {}).get("frames", []))
                frames.append(frame_record)
                if self.session_policy["enabled"]:
                    session_verdict = session_decision(
                        frames, self.session_policy["minimum_frames"])
                else:
                    session_verdict = result["tray_verdict"]
                session_action = route_action(session_verdict)
                result["service"] = {
                    "schema_version": "vial-inspection-service-record-2",
                    "processed_at": utc_now(),
                    "duration_ms": duration_ms,
                    "source_sha256": source_hash,
                    "bundle_manifest_sha256": self.bundle_hash,
                    "frame_route_action": route_action(result["tray_verdict"]),
                    "tray_id": tray_id,
                    "session_confirmation": self.session_policy["enabled"],
                    "session_frame_count": len(frames),
                    "session_tray_verdict": session_verdict,
                    "route_action": session_action,
                    "session_finalized": session_verdict != "unresolved",
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
                "frame_tray_verdict": result["tray_verdict"],
                "tray_verdict": session_verdict,
                "route_action": session_action,
                "tray_id": tray_id,
                "duration_ms": duration_ms,
            }
            if tray_id is not None:
                self.state["sessions"][tray_id] = {
                    "tray_id": tray_id,
                    "frames": frames,
                    "frame_count": len(frames),
                    "tray_verdict": session_verdict,
                    "route_action": session_action,
                    "finalized": session_verdict != "unresolved",
                    "latest_record_path": record_relative.as_posix(),
                    "updated_at": utc_now(),
                }
            self.state["failed"].pop(source_key, None)
            self.write_state()
            atomic_write_json(self.output_dir/"latest.json", {
                "schema_version": "vial-inspection-service-latest-2",
                "updated_at": utc_now(),
                "filename": path.name,
                "source_sha256": source_hash,
                "frame_tray_verdict": result["tray_verdict"],
                "tray_id": tray_id,
                "session_frame_count": len(frames),
                "tray_verdict": session_verdict,
                "route_action": session_action,
                "session_finalized": session_verdict != "unresolved",
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
                "tray_id": locals().get("tray_id"),
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
                  "failed-previously": 0, "session-finalized": 0}
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
    parser.add_argument("--confirm-sessions", action="store_true",
                        help="require conservative multi-frame tray consensus")
    parser.add_argument("--minimum-session-frames", type=int, default=3)
    parser.add_argument("--session-delimiter", default="__",
                        help="tray/frame filename delimiter when no sidecar exists")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("poll-seconds must be positive")
    try:
        service = InspectionService(
            args.labeler, args.bundle, args.input, args.output,
            args.good_expected_through, args.scrap_expected_through,
            args.confirm_sessions, args.minimum_session_frames,
            args.session_delimiter)
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
          f"sessions={len(service.state['sessions'])} "
          f"failed={len(service.state['failed'])} output={service.output_dir}")


if __name__ == "__main__":
    main()
