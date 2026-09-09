"""Native workspace routing and build layout; external runtimes are mocked."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.openagentsafety import build_images, run_infer


def test_apptainer_never_builds_docker(monkeypatch, tmp_path):
    sif = tmp_path / "agent.sif"
    sif.touch()
    monkeypatch.setenv("EVAL_AGENT_SERVER_SIF", str(sif))
    build = Mock(side_effect=AssertionError("Docker called"))
    backend = Mock(return_value="workspace")
    monkeypatch.setattr(run_infer, "build_workspace_image", build)
    monkeypatch.setattr(run_infer, "ApptainerWorkspace", backend)
    assert run_infer.create_workspace("apptainer", []) == "workspace"
    assert backend.call_args.kwargs["sif_file"] == str(sif)
    build.assert_not_called()


@pytest.mark.parametrize("backend", ["remote", "apptainer"])
def test_unsupported_or_missing_image_fails_before_start(monkeypatch, backend):
    monkeypatch.delenv("EVAL_AGENT_SERVER_SIF", raising=False)
    with pytest.raises(ValueError):
        run_infer.create_workspace(backend, [])


def test_docker_preserves_backend_configuration(monkeypatch):
    monkeypatch.setattr(run_infer, "build_workspace_image", lambda: "image:pin")
    backend = Mock()
    monkeypatch.setattr(run_infer, "DockerWorkspace", backend)
    run_infer.create_workspace("docker", ["DEBUG"])
    backend.assert_called_once_with(
        server_image="image:pin",
        platform="linux/amd64",
        extra_ports=True,
        forward_env=["DEBUG"],
    )


def test_build_uses_checkout_as_context(monkeypatch):
    monkeypatch.setattr(build_images, "get_image_name", lambda: "image:pin")
    monkeypatch.setattr(build_images, "get_vendor_sdk_commit", lambda: "pin")
    monkeypatch.setattr(build_images, "check_image_exists", lambda _: True)
    build = Mock(return_value=SimpleNamespace(error=None))
    monkeypatch.setattr(build_images, "run_docker_build_layer", build)
    assert build_images.build_workspace_image(force_rebuild=True) == "image:pin"
    root = Path(build_images.__file__).resolve().parents[2]
    assert build.call_args.kwargs["context"] == root
    assert (
        "COPY vendor/software-agent-sdk "
        in (root / "benchmarks/openagentsafety/Dockerfile").read_text()
    )


@pytest.mark.parametrize("kind", ["workspace", "utils"])
def test_missing_required_asset_stops_setup(kind):
    workspace = Mock()
    workspace.execute_command.return_value = SimpleNamespace(
        exit_code=22, stderr="HTTP 404"
    )
    with pytest.raises(RuntimeError, match="required asset"):
        run_infer.download_files_for_task(
            workspace,
            {
                f"has_{kind}": True,
                f"{kind}_files": [f"https://example.com/{kind}/missing.txt"],
            },
        )
