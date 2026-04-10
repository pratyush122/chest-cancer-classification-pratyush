import base64
import pytest

from cnnClassifier.utils.common import ImageDecodeError, decodeImage, read_yaml


def test_decode_image_writes_bytes(tmp_path):
    output_path = tmp_path / "decoded.bin"
    payload = base64.b64encode(b"abc123").decode("ascii")

    decodeImage(payload, output_path)

    assert output_path.read_bytes() == b"abc123"


def test_decode_image_rejects_invalid_base64(tmp_path):
    with pytest.raises(ImageDecodeError):
        decodeImage("nope %%%", tmp_path / "decoded.bin")


def test_decode_image_enforces_size_limit(tmp_path):
    payload = base64.b64encode(b"abc123").decode("ascii")

    with pytest.raises(ImageDecodeError):
        decodeImage(payload, tmp_path / "decoded.bin", max_bytes=3)


def test_read_yaml_returns_config_box(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("author: Pratyush Mishra\nservice:\n  port: 8080\n", encoding="utf-8")

    config = read_yaml(yaml_path)

    assert config.author == "Pratyush Mishra"
    assert config.service.port == 8080
