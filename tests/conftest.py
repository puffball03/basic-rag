# conftest.py: prints the evaluation-harness summary at the end of every pytest run
EVALS = {
    "test_retrieval_recall[": "Retrieval recall@3",
    "test_answer_accuracy[": "Answer accuracy",
}


def pytest_terminal_summary(terminalreporter):
    counts = {label: {"passed": 0, "failed": 0, "skipped": 0} for label in EVALS.values()}
    for outcome in ("passed", "failed", "skipped"):
        for report in terminalreporter.stats.get(outcome, []):
            nodeid = getattr(report, "nodeid", "")
            for prefix, label in EVALS.items():
                if prefix in nodeid:
                    counts[label][outcome] += 1

    if not any(sum(c.values()) for c in counts.values()):
        return  # no evals ran (e.g. only unit tests selected)

    terminalreporter.section("evaluation harness")
    for label, c in counts.items():
        ran = c["passed"] + c["failed"]
        if ran:
            line = f"{label + ':':<22} {c['passed']}/{ran} ({c['passed'] / ran:.0%})"
        else:
            line = f"{label + ':':<22} not run"
        if c["skipped"]:
            line += f"   [{c['skipped']} skipped: FOUNDRY_API_KEY not set]"
        terminalreporter.write_line(line)
