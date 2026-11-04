import re
from pathlib import Path

source_file = Path("lotm_full.txt")
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

print("Читаємо файл...")
content = source_file.read_text(encoding="utf-8")

print("Нарізаємо на розділи...")
chapters = re.split(r'(?=Chapter \d+)', content, flags=re.IGNORECASE)

saved_count = 0
for i, chap_text in enumerate(chapters):
    if len(chap_text.strip()) > 500:
        file_path = data_dir / f"chapter_{saved_count + 1}.txt"
        file_path.write_text(chap_text.strip(), encoding="utf-8")
        saved_count += 1

print(f"Готово! Згенеровано {saved_count} файлів-розділів у папці {data_dir}/.")