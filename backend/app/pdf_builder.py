import shutil
import subprocess
from pathlib import Path

_WINDOWS_FALLBACK = r"C:\Program Files\LibreOffice\program\soffice.com"


def _find_soffice() -> str:
    for name in ("soffice", "soffice.com"):
        found = shutil.which(name)
        if found:
            return found
    if Path(_WINDOWS_FALLBACK).exists():
        return _WINDOWS_FALLBACK
    raise RuntimeError("LibreOffice (soffice) не найден. Установите его для конвертации в PDF.")


def build_protocol_pdf(docx_path: str, output_path: str):
    """Конвертирует уже собранный DOCX в PDF через LibreOffice headless."""
    soffice = _find_soffice()
    out_dir = Path(output_path).parent

    subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), docx_path],
        check=True,
        timeout=60,
    )

    generated = out_dir / (Path(docx_path).stem + ".pdf")
    if generated != Path(output_path):
        generated.rename(output_path)
