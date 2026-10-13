#!/usr/bin/env python3
"""Portable experiment plans, sequential execution, and Slurm jobs (stdlib only)."""

from __future__ import annotations

import argparse
from collections import deque
import copy
import datetime as dt
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys

MARKER = "@@OVERSEER "
RESOURCE_FLAGS = {
    "partition": "partition",
    "account": "account",
    "qos": "qos",
    "constraint": "constraint",
    "gpus": "gpus",
    "cpus_per_task": "cpus-per-task",
    "memory": "mem",
    "time": "time",
    "nodes": "nodes",
    "ntasks": "ntasks",
}


def emit(event, **values):
    print(
        MARKER + json.dumps({"event": event, **values}, ensure_ascii=False), flush=True
    )


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def merge(base, child):
    result = copy.deepcopy(base)
    for key, value in child.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def load_config(path, project=None):
    path = Path(path).resolve()
    config = read(path)
    if config.get("version") != 1:
        raise ValueError("Experiment version must be 1")
    root = (
        Path(project).resolve()
        if project
        else (path.parent / config.get("project", "..")).resolve()
    )
    return config, root


def choices(config, dimension):
    values = config.get(dimension, {}).get("choices", [])
    entries = [
        {"id": str(v), "value": v} if not isinstance(v, dict) else copy.deepcopy(v)
        for v in values
    ]
    ids = [v["id"] for v in entries]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate {dimension} IDs")
    return entries


def selected(config, dimension):
    lookup = {v["id"]: v for v in choices(config, dimension)}
    result = []
    for name in config.get(dimension, {}).get("selected", []):
        key = str(name)
        if key not in lookup:
            raise ValueError(f"Unknown {dimension} selection: {key}")
        if key not in [v["id"] for v in result]:
            result.append(lookup[key])
    return result


def server_profiles(config):
    lookup = {v["id"]: v for v in choices(config, "servers")}

    def resolve(name, parents=()):
        if name in parents:
            raise ValueError("Server inheritance cycle: " + name)
        own = lookup[name]
        base = resolve(own["extends"], (*parents, name)) if own.get("extends") else {}
        profile = merge(base, own)
        profile["id"] = name
        profile["parents"] = [name, *base.get("parents", [])]
        return profile

    return [resolve(v["id"]) for v in selected(config, "servers")]


def expand(value, variables):
    text = str(value)

    def replace(match):
        key = match.group(1)
        if key not in variables:
            raise ValueError("Unknown experiment placeholder: " + key)
        return str(variables[key])

    return re.sub(r"\$\{([\w_]+)\}", replace, text)


def path_on_server(path, root, profile):
    return (
        str(Path(path))
        if Path(path).is_absolute()
        else str(Path(profile.get("project_dir", str(root))) / path)
    )


def validate_profile(profile, threads):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", profile["id"]):
        raise ValueError("Server IDs must be safe file names")
    if profile.get("enabled", True) is False:
        raise ValueError("Server profile is disabled: " + profile["id"])
    if profile.get("kind", "local") not in {"local", "ssh"}:
        raise ValueError("Server kind must be local or ssh")
    if profile.get("kind") == "ssh":
        host = profile.get("host", "")
        if not host or host.startswith("-") or any(c.isspace() for c in host):
            raise ValueError("Set a valid SSH host for " + profile["id"])
        if not profile.get("project_dir"):
            raise ValueError("Set the remote project_dir for " + profile["id"])
        if not Path(profile["project_dir"]).is_absolute():
            raise ValueError("The remote project_dir must be an absolute server path")
    resources = profile.get("slurm", {})
    if int(resources.get("ntasks", 1)) != 1:
        raise ValueError(
            "Sequential experiments use ntasks=1; configure a program launcher for MPI"
        )
    unknown = set(resources) - set(RESOURCE_FLAGS)
    if unknown:
        raise ValueError("Unknown Slurm resource fields: " + ", ".join(sorted(unknown)))
    for value in resources.values():
        if "\n" in str(value) or "\r" in str(value):
            raise ValueError("Slurm resource values must occupy one line")
    if resources.get("cpus_per_task") and int(resources["cpus_per_task"]) < max(
        threads, default=1
    ):
        raise ValueError("cpus_per_task is smaller than the selected thread count")


def plan(config, root, output_dir, validate=True):
    profiles = server_profiles(config)
    datasets = selected(config, "datasets")
    thread_entries = selected(config, "threads")
    threads = [int(v.get("value", v["id"])) for v in thread_entries]
    methods = selected(config, "methods")
    if any(v < 1 for v in threads):
        raise ValueError("Thread counts must be positive")
    if validate and not all((profiles, datasets, threads, methods)):
        raise ValueError(
            "Select at least one server, dataset, thread count, and method"
        )
    runs = []
    for profile in profiles:
        if validate:
            validate_profile(profile, threads)
        project = (
            str(root)
            if profile.get("kind", "local") == "local"
            else profile.get("project_dir", "")
        )
        profile["project_dir"] = project
        server_output = (
            str(Path(output_dir).resolve())
            if profile.get("kind", "local") == "local"
            else path_on_server(
                config.get("output_dir", "results/overseer")
                + "/"
                + Path(output_dir).name,
                root,
                profile,
            )
        )
        for dataset, thread, method in itertools.product(datasets, threads, methods):
            number = len(runs) + 1
            dataset_path = dataset.get("path", "")
            for profile_name in profile.get("parents", [profile["id"]]):
                if profile_name in dataset.get("paths", {}):
                    dataset_path = dataset["paths"][profile_name]
                    break
            dataset_path = path_on_server(dataset_path, root, profile)
            folder = str(Path(server_output) / f"{number:04d}")
            variables = {
                "project": project,
                "dataset": dataset_path,
                "dataset_id": dataset["id"],
                "threads": thread,
                "method": method.get("value", method["id"]),
                "server": profile["id"],
                "run_id": f"{number:04d}",
                "result": folder + "/result.json",
                "run_dir": folder,
                "preset": config.get("preset", ""),
                "target": config.get("target", ""),
            }
            binary = method.get("binaries", {}).get(
                profile["id"],
                method.get("binary", profile.get("binary", config.get("binary", ""))),
            )
            if validate and not binary:
                raise ValueError("Choose an executable binary")
            binary = path_on_server(expand(binary, variables), root, profile)
            arguments = method.get(
                "arguments", profile.get("arguments", config.get("args", []))
            )
            args = [expand(v, variables) for v in [*arguments, *method.get("args", [])]]
            env = {
                **config.get("environment", {}),
                **profile.get("environment", {}),
                **method.get("environment", {}),
            }
            env = {str(k): expand(v, variables) for k, v in env.items()}
            for key in env:
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                    raise ValueError("Invalid environment variable name: " + key)
            runs.append(
                {
                    "id": number,
                    "server": profile["id"],
                    "dataset": dataset["id"],
                    "threads": thread,
                    "method": method["id"],
                    "argv": [binary, *args],
                    "cwd": project,
                    "environment": env,
                    "folder": folder,
                }
            )
    return {
        "version": 1,
        "name": config.get("name", "Experiment"),
        "profiles": profiles,
        "runs": runs,
        "total": len(runs),
        "stop_on_failure": config.get("stop_on_failure", False),
    }


def marker_line(event, **values):
    return "printf '%s\\n' " + shlex.quote(
        MARKER + json.dumps({"event": event, **values})
    )


def build_commands(config, profile):
    build = config.get("build", {})
    variables = {
        "project": profile["project_dir"],
        "preset": config.get("preset", ""),
        "target": config.get("target", ""),
        "build_dir": build.get("directory", "build"),
        "build_jobs": build.get("jobs", 4),
    }
    return [[expand(v, variables) for v in argv] for argv in build.get("commands", [])]


def script(
    config, manifest, profile, build_only=False, batch=False, configure_only=False
):
    runs = [r for r in manifest["runs"] if r["server"] == profile["id"]]
    lines = ["#!/usr/bin/env bash"]
    if batch:
        resources = dict(profile.get("slurm", {}))
        resources.setdefault(
            "cpus_per_task", max((r["threads"] for r in runs), default=1)
        )
        resources.setdefault("ntasks", 1)
        for key, value in resources.items():
            if value != "":
                lines.append(
                    "#SBATCH --" + RESOURCE_FLAGS[key] + "=" + shlex.quote(str(value))
                )
        lines.append(
            "#SBATCH --job-name="
            + re.sub(r"[^\w-]", "_", manifest["name"] + "-" + profile["id"])
        )
    lines.extend(
        ["set -uo pipefail", "cd " + shlex.quote(profile["project_dir"]) + " || exit 1"]
    )
    for setup in profile.get("setup", []):
        lines.append("{\n" + setup + "\n} || exit 1")
    for key, value in profile.get("environment", {}).items():
        if "${" not in str(value):
            lines.append("export " + key + "=" + shlex.quote(str(value)))
    if build_only or config.get("build", {}).get("before_run", False):
        directory = config.get("build", {}).get("directory")
        if directory:
            query = str(Path(directory) / ".cmake/api/v1/query")
            lines.extend(
                [
                    "mkdir -p " + shlex.quote(query) + " || exit 1",
                    ": > " + shlex.quote(query + "/codemodel-v2"),
                ]
            )
        commands = build_commands(config, profile)
        if not commands:
            raise ValueError("Set build.commands before configuring or building")
        for argv in commands[:1] if configure_only else commands:
            lines.append(marker_line("build", server=profile["id"], argv=argv))
            lines.append("if " + shlex.join(argv) + "; then :; else")
            lines.append(marker_line("build_failed", server=profile["id"]))
            lines.extend(["exit 1", "fi"])
    if build_only:
        return "\n".join([*lines, marker_line("built", server=profile["id"]), ""])
    lines.append("failures=0")
    for run in runs:
        values = {k: run[k] for k in ("id", "server", "dataset", "threads", "method")}
        lines.extend(
            [
                "mkdir -p " + shlex.quote(run["folder"]) + " || exit 1",
                marker_line("start", total=manifest["total"], **values),
            ]
        )
        env = [k + "=" + str(v) for k, v in run["environment"].items()]
        lines.append(
            shlex.join(["env", *env, *run["argv"]])
            + " 2>&1 | tee "
            + shlex.quote(run["folder"] + "/output.log")
        )
        lines.append("code=${PIPESTATUS[0]}")
        # Exit code is dynamic; the rest is serialized and shell-quoted once.
        prefix = (
            MARKER + json.dumps({"event": "finish", **values})[:-1] + ', "exit_code": '
        )
        lines.append(
            "printf '%s%s%s\\n' " + shlex.quote(prefix) + ' "$code" ' + shlex.quote("}")
        )
        lines.append(
            'if [ "$code" -ne 0 ]; then failures=$((failures+1)); '
            + ('exit "$code"; ' if manifest["stop_on_failure"] else "")
            + "fi"
        )
    lines.append('[ "$failures" -eq 0 ]')
    return "\n".join([*lines, ""])


def ssh(profile, command, tty=False):
    return [
        "ssh",
        "-tt" if tty else "-T",
        "-o",
        "BatchMode=yes",
        *profile.get("ssh_options", []),
        profile["host"],
        command,
    ]


def allocation_args(profile, runs):
    resources = dict(profile.get("slurm", {}))
    resources.setdefault("cpus_per_task", max((r["threads"] for r in runs), default=1))
    resources.setdefault("ntasks", 1)
    return [
        "--" + RESOURCE_FLAGS[k] + "=" + str(v) for k, v in resources.items() if v != ""
    ]


def execute(argv, on_event, log):
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    def interrupt(signum, _frame):
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        raise KeyboardInterrupt

    previous = {
        sig: signal.signal(sig, interrupt)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        with Path(log).open("w") as output:
            for line in process.stdout:
                print(line, end="", flush=True)
                output.write(line)
                clean = line.strip()
                if clean.startswith(MARKER):
                    event = json.loads(clean[len(MARKER) :])
                    on_event(event)
        return process.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def export(config, root, folder):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    saved = copy.deepcopy(config)
    saved["project"] = os.path.relpath(root, folder)
    write(folder / "experiment.json", saved)
    manifest = plan(config, root, folder)
    write(folder / "manifest.json", manifest)
    if Path(__file__).resolve() != (folder / "runner.py").resolve():
        shutil.copyfile(__file__, folder / "runner.py")
    for profile in manifest["profiles"]:
        text = script(config, manifest, profile, batch=True)
        (folder / (profile["id"] + ".sbatch")).write_text(text)
    return manifest


def run(config, root, folder, build_only=False, configure_only=False):
    folder.mkdir(parents=True, exist_ok=True)
    manifest = plan(config, root, folder, validate=not build_only)
    if build_only:
        threads = [int(v.get("value", v["id"])) for v in selected(config, "threads")]
        for profile in manifest["profiles"]:
            validate_profile(profile, threads)
    write(folder / "manifest.json", manifest)
    results = []

    def record(event):
        if event.get("event") == "finish":
            results.append(event)
            write(
                folder / "results.json",
                {
                    "total": manifest["total"],
                    "completed": len(results),
                    "runs": results,
                },
            )

    code = 0
    for profile in manifest["profiles"]:
        text = script(
            config,
            manifest,
            profile,
            build_only=build_only,
            configure_only=configure_only,
        )
        remote = shlex.join(["bash", "-lc", text])
        argv = ["bash", "-c", text]
        if "slurm" in profile:
            emit("allocation", server=profile["id"])
            argv = [
                "srun",
                *allocation_args(profile, manifest["runs"]),
                "bash",
                "-lc",
                text,
            ]
            remote = shlex.join(argv)
        if profile.get("kind") == "ssh":
            argv = ssh(profile, remote, tty=True)
        emit("server", server=profile["id"])
        status = execute(argv, record, folder / (profile["id"] + ".log"))
        code = max(code, int(status != 0))
        if status and manifest["stop_on_failure"]:
            break
    emit(
        "complete",
        total=manifest["total"],
        completed=len(results),
        failed=sum(r["exit_code"] != 0 for r in results),
        exit_code=code,
        folder=str(folder),
    )
    return code


def submit(config, root, folder):
    manifest = export(config, root, folder)
    for profile in manifest["profiles"]:
        if "slurm" not in profile:
            raise ValueError(
                "Add Slurm resources to profile " + profile["id"] + " before submitting"
            )
    jobs = []
    for profile in manifest["profiles"]:
        if "slurm" not in profile:
            raise ValueError(
                "Add Slurm resources to profile " + profile["id"] + " before submitting"
            )
        text = (folder / (profile["id"] + ".sbatch")).read_text()
        argv = ["sbatch", "--parsable"]
        if profile.get("kind") == "ssh":
            argv = ssh(profile, shlex.join(argv))
        result = subprocess.run(
            argv, input=text, text=True, capture_output=True, check=True
        )
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise ValueError("Unexpected sbatch job ID: " + result.stdout)
        job = {"server": profile["id"], "job_id": job_id, "profile": profile}
        jobs.append(job)
        write(folder / "jobs.json", jobs)
        emit("submitted", server=profile["id"], job_id=job_id, folder=str(folder))
    return 0


def job_action(folder, action):
    jobs = read(folder / "jobs.json")
    for job in jobs:
        profile, job_id = job["profile"], job["job_id"]
        argv = (
            ["scancel", job_id]
            if action == "cancel"
            else ["squeue", "--noheader", "--jobs", job_id, "--format", "%T"]
        )
        if profile.get("kind") == "ssh":
            argv = ssh(profile, shlex.join(argv))
        result = subprocess.run(argv, text=True, capture_output=True, check=True)
        status = "CANCELED" if action == "cancel" else result.stdout.strip()
        if action == "poll" and not status:
            argv = [
                "sacct",
                "--noheader",
                "--parsable2",
                "--jobs",
                job_id,
                "--format",
                "JobIDRaw,State",
            ]
            if profile.get("kind") == "ssh":
                argv = ssh(profile, shlex.join(argv))
            result = subprocess.run(argv, text=True, capture_output=True, check=True)
            status = next(
                (
                    line.split("|")[1]
                    for line in result.stdout.splitlines()
                    if line.split("|")[0] == job_id
                ),
                "UNKNOWN",
            )
        job["status"] = status
        emit("job", server=job["server"], job_id=job_id, status=status)
    write(folder / "jobs.json", jobs)
    return 0


def dispatch(args):
    config, root = load_config(args.config, args.project)
    if args.action == "format":
        write(args.config, config)
        return 0
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else root / config.get("output_dir", "results/overseer") / stamp
    )
    if args.action == "log":
        if args.run_id is None or not args.output_dir:
            raise ValueError("Use --run-id and --output-dir to view a run log")
        manifest = read(folder / "manifest.json")
        trial = next((r for r in manifest["runs"] if r["id"] == args.run_id), None)
        if not trial:
            raise ValueError("Unknown run ID")
        profile = next(p for p in manifest["profiles"] if p["id"] == trial["server"])
        log = str(Path(trial["folder"]) / "output.log")
        if profile.get("kind") == "ssh":
            subprocess.run(
                ssh(profile, shlex.join(["tail", "-n", "300", log])), check=True
            )
        else:
            with Path(log).open() as source:
                print("".join(deque(source, maxlen=300)), end="")
        return 0
    if args.action == "targets":
        directory = root / config.get("build", {}).get("directory", "build")
        reply = directory / ".cmake/api/v1/reply"
        indexes = sorted(reply.glob("index-*.json"))
        targets = {}
        if indexes:
            for obj in read(indexes[-1]).get("objects", []):
                if obj["kind"] == "codemodel":
                    for configuration in read(reply / obj["jsonFile"]).get(
                        "configurations", []
                    ):
                        for target in configuration.get("targets", []):
                            data = read(reply / target["jsonFile"])
                            if data.get("type") == "EXECUTABLE" and data.get(
                                "artifacts"
                            ):
                                path = directory / data["artifacts"][0]["path"]
                                targets[target["name"]] = {
                                    "id": target["name"],
                                    "binary": os.path.relpath(path, root),
                                }
        print(json.dumps(sorted(targets.values(), key=lambda v: v["id"])))
        return 0
    if args.action == "describe":
        manifest = plan(config, root, folder, validate=False)
        print(json.dumps({"project": str(root), "config": config, "plan": manifest}))
        return 0
    if args.action in {"poll", "cancel"}:
        if not args.output_dir:
            raise ValueError("Use --output-dir for the submitted experiment")
        return job_action(folder, args.action)
    if args.action == "export":
        manifest = export(config, root, folder)
        emit("exported", folder=str(folder), total=manifest["total"])
        return 0
    if args.action == "submit":
        return submit(config, root, folder)
    return run(
        config,
        root,
        folder,
        build_only=args.action in {"build", "configure"},
        configure_only=args.action == "configure",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "format",
            "describe",
            "targets",
            "configure",
            "run",
            "build",
            "export",
            "submit",
            "poll",
            "cancel",
            "log",
        ],
    )
    parser.add_argument("config")
    parser.add_argument("--project")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-id", type=int)
    return dispatch(parser.parse_args(argv))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        emit("canceled")
        sys.exit(130)
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError) as error:
        emit("error", message=str(error))
        sys.exit(2)
