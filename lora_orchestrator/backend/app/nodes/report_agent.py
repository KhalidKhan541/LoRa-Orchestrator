import os
import shutil
from backend.app.utils import report_utils


def report_node(state: dict, llm=None) -> dict:
    cfg = state.get("run_config", {})
    session_id = cfg.get("session_id", "unknown")
    decision = state.get("decision", "unknown")

    # Export final adapter
    adapter_dest = os.path.join("outputs", "adapters", session_id)
    os.makedirs(adapter_dest, exist_ok=True)
    training = state.get("training", {})
    src_adapter = training.get("adapter_path", "")
    if src_adapter and os.path.exists(src_adapter):
        shutil.copytree(src_adapter, adapter_dest, dirs_exist_ok=True)

    # Build reports
    json_report = report_utils.build_json_report(state)
    md_report = report_utils.build_markdown_report(state, llm)
    report_utils.save_reports(session_id, json_report, md_report, "outputs")

    report_dir = os.path.join("outputs", "reports", session_id)

    return {
        "report": {
            "json_path": os.path.join(report_dir, "report.json"),
            "md_path": os.path.join(report_dir, "report.md"),
            "adapter_path": adapter_dest,
        },
        "status": "done" if decision == "stop" else "failed",
        "agent_logs": [f"[Report] Report saved to {report_dir}"],
    }
