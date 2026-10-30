from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED = {"schema_version", "voidgym_version", "python", "repo", "bridge_dll", "modding_api", "supported_games", "map_art"}


class ManifestError(ValueError):
    """The pinned installer manifest is incomplete or unsafe."""


def _is_https_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.hostname) and not parsed.username and not parsed.password


def _digest(value: object, label: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not _SHA256.fullmatch(value.lower()):
        raise ManifestError(f"{label} precisa ser um SHA-256 hexadecimal de 64 caracteres.")


def _validate_python(value: object) -> None:
    expected_url = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
    if not isinstance(value, dict) or value.get("version") != "3.12.10" or value.get("url") != expected_url:
        raise ManifestError("Manifesto precisa fixar o instalador oficial do Python 3.12.10.")
    _digest(value.get("sha256"), "python.sha256")


def _validate_repo(value: object) -> None:
    if not isinstance(value, dict) or not _is_https_url(value.get("url")) or not isinstance(value.get("ref"), str) or not value["ref"]:
        raise ManifestError("repo precisa definir URL HTTPS e ref fixos.")


def _validate_bridge(value: object, *, require_installable: bool) -> None:
    if not isinstance(value, dict):
        raise ManifestError("bridge_dll precisa ser um objeto.")
    if value.get("available") is True:
        if not _is_https_url(value.get("url")):
            raise ManifestError("bridge_dll.url precisa ser HTTPS.")
        _digest(value.get("sha256"), "bridge_dll.sha256")
    elif require_installable:
        raise ManifestError("A DLL do bridge ainda não foi publicada com URL e SHA-256 fixos.")


def _validate_api(value: object) -> None:
    if not isinstance(value, dict) or value.get("version") != "77" or value.get("game_version") != "1.5.78.11833":
        raise ManifestError("Modding API incompatível com o perfil GOG suportado.")
    if not _is_https_url(value.get("url")):
        raise ManifestError("modding_api.url precisa ser HTTPS.")
    _digest(value.get("sha256"), "modding_api.sha256")


def _validate_games(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise ManifestError("supported_games precisa conter ao menos um perfil.")
    for index, game in enumerate(value):
        if not isinstance(game, dict) or game.get("store") != "gog" or game.get("version") != "1.5.78.11833":
            raise ManifestError(f"Perfil de jogo {index} inválido ou não suportado.")
        _digest(game.get("exe_sha256"), f"supported_games[{index}].exe_sha256")
        _digest(game.get("globalgamemanagers_sha256"), f"supported_games[{index}].globalgamemanagers_sha256")


def _validate_map_art(value: object) -> None:
    if not isinstance(value, dict) or value.get("distribution_approved") not in (True, False):
        raise ManifestError("map_art precisa declarar explicitamente o estado de consentimento.")
    if type(value["distribution_approved"]) is not bool:
        raise ManifestError("map_art.distribution_approved precisa ser booleano.")
    if value["distribution_approved"]:
        if not _is_https_url(value.get("url")):
            raise ManifestError("map_art.url precisa ser HTTPS após consentimento.")
        _digest(value.get("sha256"), "map_art.sha256")
    if not isinstance(value.get("credit"), str) or not value["credit"].strip():
        raise ManifestError("O crédito da arte do mapa é obrigatório.")


def validate_manifest(data: object, *, require_installable: bool = False) -> dict[str, Any]:
    if not isinstance(data, dict) or not _REQUIRED.issubset(data):
        missing = sorted(_REQUIRED - set(data)) if isinstance(data, dict) else sorted(_REQUIRED)
        raise ManifestError("Manifesto inválido; campos ausentes: " + ", ".join(missing))
    if data.get("schema_version") != 1:
        raise ManifestError("Versão de esquema do manifesto não suportada.")
    _validate_python(data.get("python"))
    _validate_repo(data.get("repo"))
    _validate_bridge(data.get("bridge_dll"), require_installable=require_installable)
    _validate_api(data.get("modding_api"))
    _validate_games(data.get("supported_games"))
    _validate_map_art(data.get("map_art"))
    return data


def load_manifest(path: Path, *, require_installable: bool = False) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"Não foi possível ler o manifesto: {error}") from error
    return validate_manifest(data, require_installable=require_installable)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
