# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import garak._config
import garak._plugins

import garak.probes.base
import garak.probes.fileformats
import garak.attempt


def test_hf_files_load():
    p = garak.probes.fileformats.HF_Files()
    assert isinstance(p, garak.probes.base.Probe)


def test_is_local_path():
    p = garak.probes.fileformats.HF_Files()
    assert p._is_local_path("/usr/local/models/my-model") is True
    assert p._is_local_path("./models/my-model") is True
    assert p._is_local_path("../models/my-model") is True
    assert p._is_local_path("/home/user/checkpoints/llama") is True
    assert p._is_local_path("gpt2") is False
    assert p._is_local_path("meta-llama/Llama-2-7b") is False
    assert p._is_local_path("org/model-name") is False


def test_gather_local_files():
    p = garak.probes.fileformats.HF_Files()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "config.json").write_text("{}")
        (Path(tmpdir) / "model.safetensors").write_bytes(b"\x00")
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        (subdir / "tokenizer.json").write_text("{}")

        filenames = p._gather_local_files(tmpdir)
        assert len(filenames) == 3
        for f in filenames:
            assert os.path.isfile(f)

        basenames = sorted(os.path.basename(f) for f in filenames)
        assert basenames == ["config.json", "model.safetensors", "tokenizer.json"]


def test_gather_local_files_nonexistent_dir():
    p = garak.probes.fileformats.HF_Files()
    filenames = p._gather_local_files("/nonexistent/path/that/does/not/exist")
    assert filenames == []


def test_probe_local_path():
    p = garak.probes.fileformats.HF_Files()

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "config.json").write_text("{}")
        (Path(tmpdir) / "model.bin").write_bytes(b"\x00")

        generator = MagicMock()
        generator.__class__.__module__ = "garak.generators.huggingface"
        generator.__class__.__name__ = "Model"
        generator.name = tmpdir

        results = p.probe(generator)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], garak.attempt.Attempt)
        assert len(results[0].outputs) == 2


# files could be their own thing if Turns start taking named/typed entries
def test_hf_files_hf_repo():
    p = garak._plugins.load_plugin("probes.fileformats.HF_Files")
    garak._config.plugins.generators["huggingface"] = {
        "Model": {"name": "gpt2", "hf_args": {"device": "cpu"}},
    }
    g = garak._plugins.load_plugin(
        "generators.huggingface.Model", config_root=garak._config
    )
    r = p.probe(g)
    assert isinstance(r, list), ".probe should return a list"
    assert len(r) == 1, "HF_Files.probe() should return one attempt"
    assert isinstance(
        r[0], garak.attempt.Attempt
    ), "HF_Files.probe() must return an Attempt"
    assert isinstance(r[0].outputs, list), "File list scan should return a list"
    assert len(r[0].outputs) > 0, "File list scan should return list of filenames"
    for filename in r[0].outputs:
        assert isinstance(
            filename.text, str
        ), "File list scan should return list of Turns with .text being string filenames"
        assert os.path.isfile(
            filename.text
        ), "List of HF_Files paths should all be real files"
