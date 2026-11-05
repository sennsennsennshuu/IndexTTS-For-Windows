"""项目内音色库：保存参考录音副本，不生成或修改语音模型权重。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from uuid import uuid4

from .paths import VOICE_LIBRARY_DIR


class VoiceLibraryError(ValueError):
    """可直接显示给用户的音色库错误；损坏清单不能被当作空库覆盖。"""


@dataclass(frozen=True)
class VoicePreset:
    """只存库内相对路径；名称不参与文件命名，避免中文及特殊字符路径冲突。"""
    id: str
    name: str
    audio_file: str
    original_filename: str
    created_at: str
    sha256: str
    size_bytes: int
    sample_rate: int
    frames: int
    channels: int

    @property
    def duration(self):
        return self.frames / self.sample_rate


@dataclass(frozen=True)
class VoiceLibraryState:
    presets: tuple[VoicePreset, ...] = ()
    selected_id: str | None = None


class VoiceLibrary:
    """音色库读写边界。每次修改前重读清单，所有删除只处理受管的单个副本。"""
    SUPPORTED = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    MAX_BYTES = 100 * 1024 * 1024

    def __init__(self, root=VOICE_LIBRARY_DIR):
        self.root = Path(root)
        self.index_path = self.root / "library.json"

    @staticmethod
    def _name_key(name):
        return unicodedata.normalize("NFKC", name).casefold()

    @classmethod
    def clean_name(cls, name):
        """显示名称允许中文和标点，但限制长度并拒绝不可见控制字符。"""
        if not isinstance(name, str) or any(unicodedata.category(c).startswith("C") for c in name):
            raise VoiceLibraryError("音色名称不能包含控制字符。")
        name = " ".join(name.split())
        if not name or len(name) > 48:
            raise VoiceLibraryError("请填写 1–48 个字符的音色名称。")
        return name

    def _check_root(self):
        """数据目录必须位于项目内，拒绝链接目录将删除操作引向项目外部。"""
        root = self.root.resolve()
        project = Path.cwd().resolve()
        if root == project or not root.is_relative_to(project):
            raise VoiceLibraryError("音色库必须位于项目内的独立目录。")
        if self.root.is_symlink() or (self.root / "audio").is_symlink():
            raise VoiceLibraryError("音色库不支持链接目录，请使用项目内的实际文件夹。")
        if self.index_path.is_symlink():
            raise VoiceLibraryError("音色清单不能使用符号链接。")

    def audio_path(self, preset):
        """路径由 UUID 和已知扩展名共同约束，不信任清单中的任意路径。"""
        self._check_root()
        suffix = Path(preset.audio_file).suffix
        if (not re.fullmatch(r"[0-9a-f]{32}", preset.id)
                or suffix not in self.SUPPORTED
                or preset.audio_file != f"audio/{preset.id}{suffix}"):
            raise VoiceLibraryError("音色清单包含不合法的录音路径。")
        path = self.root / preset.audio_file
        if path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve()):
            raise VoiceLibraryError("音色录音路径超出库目录，已停止操作。")
        return path

    def load(self):
        """空库无需创建目录；损坏或未知版本明确报错，保留原始清单。"""
        self._check_root()
        if not self.index_path.exists():
            return VoiceLibraryState()
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != 1:
                raise ValueError("不支持的清单版本")
            if not isinstance(payload.get("presets"), list):
                raise ValueError("音色条目不是列表")
            presets = tuple(VoicePreset(**item) for item in payload["presets"])
            ids, names = set(), set()
            for preset in presets:
                self.audio_path(preset)
                name = self.clean_name(preset.name)
                key = self._name_key(name)
                if preset.id in ids or key in names:
                    raise ValueError("存在重复名称或标识")
                ids.add(preset.id)
                names.add(key)
                if (not re.fullmatch(r"[0-9a-f]{64}", preset.sha256)
                        or not isinstance(preset.original_filename, str)
                        or not isinstance(preset.created_at, str)
                        or any(type(value) is not int or value <= 0 for value in (
                            preset.size_bytes, preset.sample_rate, preset.frames, preset.channels))):
                    raise ValueError("录音元数据不完整")
            selected = payload.get("selected_id")
            if selected is not None and selected not in ids:
                raise ValueError("选中音色不存在")
            return VoiceLibraryState(presets, selected)
        except (TypeError, ValueError, KeyError, OSError) as exc:
            raise VoiceLibraryError(f"音色库清单无法读取，已保留原文件：{exc}") from exc

    def _write(self, state):
        """同目录临时文件加原子替换，写入失败时旧清单仍然完整。"""
        self._check_root()
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "selected_id": state.selected_id,
                   "presets": [asdict(item) for item in state.presets]}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                    prefix=".library-", suffix=".tmp", dir=self.root, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.index_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def save(self, name, source):
        """先验证复制完成的录音，再提交清单；任何失败均撤销本次新副本。"""
        name = self.clean_name(name)
        state = self.load()
        if any(self._name_key(item.name) == self._name_key(name) for item in state.presets):
            raise VoiceLibraryError("已有同名音色，请换一个名称。")
        source = Path(source)
        if not source.is_file() or source.suffix.lower() not in self.SUPPORTED:
            raise VoiceLibraryError("请先选择有效的 WAV、MP3、FLAC、OGG 或 M4A 参考录音。")
        before = source.stat()
        if before.st_size <= 0 or before.st_size > self.MAX_BYTES:
            raise VoiceLibraryError("参考录音不能为空，且不能超过 100 MB。")
        preset_id = uuid4().hex
        relative = f"audio/{preset_id}{source.suffix.lower()}"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            digest = hashlib.sha256()
            with source.open("rb") as incoming, temporary.open("xb") as outgoing:
                for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                    digest.update(chunk)
                    outgoing.write(chunk)
                    if outgoing.tell() > self.MAX_BYTES:
                        raise VoiceLibraryError("参考录音在保存过程中超出大小限制。")
                outgoing.flush()
                os.fsync(outgoing.fileno())
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise VoiceLibraryError("参考录音在保存过程中发生变化，请重新保存。")
            import numpy as np
            import soundfile as sf
            # 验证完整解码，拒绝损坏、非有限数值和全静音参考，避免库内留存不可用音色。
            try:
                with sf.SoundFile(temporary) as audio:
                    rate, frames, channels = audio.samplerate, audio.frames, audio.channels
                    decoded, peak = 0, 0.0
                    for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
                        if not np.isfinite(block).all():
                            raise VoiceLibraryError("参考录音包含异常采样值，请更换录音。")
                        decoded += len(block)
                        peak = max(peak, float(np.max(np.abs(block))))
                if frames <= 0 or decoded != frames or peak < 1e-6:
                    raise VoiceLibraryError("参考录音为空、已损坏或全静音，请更换录音。")
            except (RuntimeError, sf.LibsndfileError) as exc:
                raise VoiceLibraryError("无法解码参考录音，请先转为 WAV 或 FLAC 后保存。") from exc
            preset = VoicePreset(preset_id, name, relative, source.name,
                                 datetime.now(timezone.utc).isoformat(), digest.hexdigest(),
                                 temporary.stat().st_size, rate, frames, channels)
            os.replace(temporary, destination)
            try:
                self._write(VoiceLibraryState((*state.presets, preset), preset.id))
            except Exception:
                destination.unlink(missing_ok=True)
                raise
            return preset
        finally:
            temporary.unlink(missing_ok=True)

    def select(self, preset_id):
        """保存选择状态；失效录音不允许作为可用音色恢复。"""
        state = self.load()
        preset = next((item for item in state.presets if item.id == preset_id), None)
        if preset_id is not None:
            if preset is None:
                raise VoiceLibraryError("音色已不存在，请刷新后重试。")
            path = self.audio_path(preset)
            if not path.is_file() or path.stat().st_size == 0:
                raise VoiceLibraryError("音色录音已丢失，请删除此条目后重新保存。")
        if state.selected_id != preset_id:
            self._write(VoiceLibraryState(state.presets, preset_id))
        return preset

    def delete(self, preset_id):
        """先暂存单个副本，清单提交成功后删除；提交失败则恢复原副本。"""
        state = self.load()
        preset = next((item for item in state.presets if item.id == preset_id), None)
        if preset is None:
            raise VoiceLibraryError("音色已不存在，请刷新后重试。")
        path = self.audio_path(preset)
        staged = path.with_name(f".delete-{uuid4().hex}{path.suffix}")
        moved = False
        try:
            if path.exists():
                path.rename(staged)
                moved = True
            remaining = tuple(item for item in state.presets if item.id != preset_id)
            selected = None if state.selected_id == preset_id else state.selected_id
            self._write(VoiceLibraryState(remaining, selected))
        except Exception:
            if moved:
                staged.rename(path)
            raise
        if moved:
            try:
                staged.unlink()
            except OSError:
                # 文件占用等异常必须可见，同时尽力恢复条目，不能悄悄遗留用户录音。
                logging.getLogger(__name__).exception("音色副本清理失败，正在恢复原条目")
                staged.rename(path)
                self._write(state)
                raise
        return preset
