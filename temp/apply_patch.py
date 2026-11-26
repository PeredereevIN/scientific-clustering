from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

PATCH_ROOT = Path(__file__).resolve().parent


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        raise RuntimeError(f"Не найден участок для патча: {label} ({path.relative_to(ROOT)})")
    if count > 1:
        raise RuntimeError(f"Неоднозначный участок патча: {label} — найдено {count} совпадений")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def backup(path: Path) -> None:
    relative = path.relative_to(ROOT)
    destination = BACKUP_ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def apply(root: Path) -> None:
    global ROOT, BACKUP_ROOT
    ROOT = root.resolve()
    required = [ROOT / "pyproject.toml", ROOT / "requirements.txt", ROOT / "src/sciintel/cli.py",
                ROOT / "src/sciintel/ingest.py", ROOT / "src/sciintel/utils.py", ROOT / "dashboard/app.py"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError("Это не корень scientific-clustering. Не найдены: " + ", ".join(missing))
    if "name = \"scientific-clustering\"" not in (ROOT / "pyproject.toml").read_text(encoding="utf-8"):
        raise RuntimeError("В выбранной папке обнаружен другой Python-проект")
    if (ROOT / "src/sciintel/linguistic").exists():
        raise RuntimeError("src/sciintel/linguistic уже существует. Патч не будет перезаписывать файлы.")
    if (ROOT / "tests/test_linguistic_analyzer.py").exists():
        raise RuntimeError("tests/test_linguistic_analyzer.py уже существует. Сначала проверьте его содержимое.")
    if (ROOT / "scripts/install_linguistic_models.ps1").exists():
        raise RuntimeError("scripts/install_linguistic_models.ps1 уже существует.")
    anchors = {
        ROOT / "src/sciintel/utils.py": ['SUPPORTED = {".pdf", ".md", ".markdown", ".txt"}'],
        ROOT / "src/sciintel/ingest.py": ["def read_document(", "\n\ndef ingest(", '            text_path.write_text(text, encoding="utf-8")', 'No PDF/MD/TXT documents found'],
        ROOT / "pyproject.toml": ["dependencies = []"],
        ROOT / "src/sciintel/cli.py": ['    graph.add_argument("--run-id", default=None)\n', '    elif args.command == "export-neo4j":\n'],
        ROOT / "dashboard/app.py": ['"Ключевые термины", "Семантический поиск"]', 'elif page == "Семантический поиск":\n'],
    }
    for file_path, expected in anchors.items():
        content = file_path.read_text(encoding="utf-8")
        for anchor in expected:
            if anchor not in content:
                raise RuntimeError(f"Проверка совместимости не пройдена: {file_path.relative_to(ROOT)} — {anchor!r}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BACKUP_ROOT = ROOT / ".patch-backups" / f"linguistic_{stamp}"
    touched = []
    for rel in ["pyproject.toml", "requirements.txt", "src/sciintel/utils.py", "src/sciintel/ingest.py",
                "src/sciintel/cli.py", "dashboard/app.py"]:
        path = ROOT / rel
        backup(path)
        touched.append(path)

    # Add the new package without replacing any existing project modules.
    target_package = ROOT / "src/sciintel/linguistic"
    if target_package.exists():
        raise RuntimeError("src/sciintel/linguistic уже существует. Патч не будет перезаписывать файлы.")
    shutil.copytree(PATCH_ROOT / "src/sciintel/linguistic", target_package)
    shutil.copy2(PATCH_ROOT / "tests/test_linguistic_analyzer.py", ROOT / "tests/test_linguistic_analyzer.py")
    scripts = ROOT / "scripts"
    scripts.mkdir(exist_ok=True)
    shutil.copy2(PATCH_ROOT / "scripts/install_linguistic_models.ps1", scripts / "install_linguistic_models.ps1")

    replace_once(ROOT / "src/sciintel/utils.py",
        'SUPPORTED = {".pdf", ".md", ".markdown", ".txt"}',
        'SUPPORTED = {".pdf", ".doc", ".docx", ".rtf", ".md", ".markdown", ".txt"}',
        "поддерживаемые форматы")

    ingest_path = ROOT / "src/sciintel/ingest.py"
    ingest_text = ingest_path.read_text(encoding="utf-8")
    start = ingest_text.find("def read_document(")
    end = ingest_text.find("\n\ndef ingest(", start)
    if start < 0 or end < 0:
        raise RuntimeError("Не удалось безопасно найти read_document() в ingest.py")
    replacement = '''def read_document(path: Path) -> tuple[str, str, int]:
    from .linguistic.converters import convert_to_markdown

    markdown, metadata = convert_to_markdown(path)
    first_heading = next((line[2:].strip() for line in markdown.splitlines()
                          if line.startswith("# ")), "")
    return clean_text(markdown), first_heading, int(metadata.get("page_count", 0))
'''
    ingest_path.write_text(ingest_text[:start] + replacement + ingest_text[end:], encoding="utf-8")
    replace_once(ingest_path,
        '            text_path.write_text(text, encoding="utf-8")',
        '            text_path.write_text(text, encoding="utf-8")\n            (out / "document.md").write_text(text, encoding="utf-8")',
        "сохранение Markdown sidecar")
    replace_once(ingest_path, 'No PDF/MD/TXT documents found',
                 'No supported PDF/DOC/DOCX/RTF/MD/TXT documents found', "сообщение ingest")

    pyproject = ROOT / "pyproject.toml"
    replace_once(pyproject, "dependencies = []", '''dependencies = [
    "spacy>=3.8,<4",
    "python-docx>=1.1,<2",
    "striprtf>=0.0.26,<1",
]''', "зависимости pyproject")
    requirements = ROOT / "requirements.txt"
    req = requirements.read_text(encoding="utf-8")
    additions = ["spacy>=3.8,<4", "python-docx>=1.1,<2", "striprtf>=0.0.26,<1"]
    if not all(item in req for item in additions):
        requirements.write_text(req.rstrip() + "\n\n# Linguistic analysis, morphology and document conversion\n" +
                                "\n".join(item for item in additions if item not in req) + "\n", encoding="utf-8")

    cli = ROOT / "src/sciintel/cli.py"
    replace_once(cli,
        '    ing = sub.add_parser("ingest", help="Обработать PDF, Markdown и TXT из папки inbox")',
        '    ing = sub.add_parser("ingest", help="Конвертировать PDF, DOC, DOCX, RTF, TXT и Markdown из папки inbox")',
        "CLI ingest help")
    replace_once(cli,
        '    graph.add_argument("--run-id", default=None)\n',
        '''    graph.add_argument("--run-id", default=None)

    ling = sub.add_parser("linguistic-index", help="Конвертировать документы в Markdown и построить морфологический индекс")
    ling.add_argument("--language", choices=["auto", "en", "ru"], default="auto")
    ling.add_argument("--force", action="store_true")
    ling.add_argument("--limit", type=int, default=None)

    ling_search = sub.add_parser("linguistic-search", help="Поиск по предложениям, словоформам, леммам и понятиям")
    ling_search.add_argument("query")
    ling_search.add_argument("--kind", choices=["sentences", "tokens", "concepts"], default="sentences")
    ling_search.add_argument("--limit", type=int, default=20)
''', "CLI parser")
    replace_once(cli,
        '    elif args.command == "export-neo4j":\n',
        '''    elif args.command == "linguistic-index":
        from .linguistic.pipeline import index_documents
        _json(index_documents(language=args.language, force=args.force, limit=args.limit))
    elif args.command == "linguistic-search":
        from .linguistic.indexer import search_concepts, search_sentences, search_tokens
        if args.kind == "concepts":
            results = search_concepts(args.query, args.limit)
        elif args.kind == "tokens":
            results = search_tokens(args.query, args.limit)
        else:
            results = search_sentences(args.query, args.limit)
        _json(results)
    elif args.command == "export-neo4j":
''', "CLI dispatch")

    dashboard = ROOT / "dashboard/app.py"
    replace_once(dashboard,
        '"Ключевые термины", "Семантический поиск"]',
        '"Ключевые термины", "Семантический поиск", "Лингвистический анализ"]',
        "навигация Streamlit")
    snippet = (PATCH_ROOT / "dashboard_page_snippet.py.txt").read_text(encoding="utf-8")
    replace_once(dashboard, 'elif page == "Семантический поиск":\n', snippet + 'elif page == "Семантический поиск":\n',
                 "страница лингвистического анализа")

    readme = ROOT / "README.md"
    backup(readme)
    readme_text = readme.read_text(encoding="utf-8")
    readme_text = readme_text.replace("Обработка PDF и текстовых файлов из data/inbox", "Обработка PDF, DOC, DOCX, RTF, TXT и Markdown из data/inbox")
    section = (PATCH_ROOT / "PATCH_README_RU.md").read_text(encoding="utf-8")
    readme.write_text(readme_text.rstrip() + "\n\n" + section, encoding="utf-8")

    print("Патч применён.")
    print(f"Резервные копии исходных файлов: {BACKUP_ROOT}")
    print("Следующий шаг: обновить зависимости и установить языковые модели spaCy (см. PATCH_README_RU.md).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Установщик патча лингвистического анализа scientific-clustering")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Корень существующего scientific-clustering")
    args = parser.parse_args()
    apply(args.project_root)
