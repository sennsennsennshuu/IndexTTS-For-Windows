"""依赖迁移辅助工具：复制指定环境中的 IndexTTS 依赖闭包；日常启动无需运行。"""
from pathlib import Path
from importlib.metadata import distributions
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import shutil
import json
import argparse

parser = argparse.ArgumentParser(description="从指定 site-packages 备份迁移运行依赖")
parser.add_argument("--source-site-packages", required=True, type=Path, help="相对于项目的源 site-packages 目录")
source = parser.parse_args().source_site_packages
if not source.is_dir():
    parser.error("源目录不存在")
target = Path('even/Lib/site-packages')
roots = '''PyQt6 torch torchaudio transformers librosa numpy omegaconf safetensors sentencepiece einops tqdm modelscope huggingface_hub cn2an jieba wetext g2p-en munch PyYAML soundfile scipy accelerate descript-audiotools json5 matplotlib pandas pip setuptools packaging'''.split()
available = {canonicalize_name(d.metadata['Name']): d for d in distributions(path=[str(source)])}
selected = {}
missing_dependencies = set()
pending = [(canonicalize_name(n), set()) for n in roots]
seen = set()
while pending:
    name, extras = pending.pop()
    key = (name, tuple(sorted(extras)))
    if key in seen:
        continue
    seen.add(key)
    if name not in available:
        missing_dependencies.add(name)
        continue
    dist = available[name]
    selected[name] = dist
    for raw in dist.requires or []:
        req = Requirement(raw)
        if req.marker is None or any(req.marker.evaluate({'extra': extra}) for extra in extras | {''}):
            pending.append((canonicalize_name(req.name), set(req.extras)))

report = {'missing_dependencies': sorted(missing_dependencies), 'distributions': {}, 'missing_files': [], 'files': 0, 'bytes': 0}
for name, dist in sorted(selected.items()):
    print(f'复制 {name}=={dist.version}', flush=True)
    report['distributions'][name] = dist.version
    for item in dist.files or []:
        # RECORD 中的 Scripts 条目含有源解释器路径，禁止照搬；启动统一使用 python -m。
        if '..' in item.parts or '__pycache__' in item.parts or item.suffix == '.pyc':
            continue
        src = source / item
        dst = target / item
        if not src.is_file():
            report['missing_files'].append(str(item))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        report['files'] += 1
        report['bytes'] += src.stat().st_size
Path('docs/environment_migration.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print(f"完成: {len(selected)} 个包，{report['files']} 个文件，{report['bytes']/1024**3:.2f} GiB", flush=True)
