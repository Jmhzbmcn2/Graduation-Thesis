import os, json

SOURCE_DIR = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\Processing_Data\VietMed_Crawl_Data\vietmed_crawled"
OUTPUT = r"C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\medical_data_filelist.json"

files = sorted(f for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f)))[:1500]

data = [{"index": i+1, "filename": f} for i, f in enumerate(files)]

with open(OUTPUT, "w", encoding="utf-8") as out:
    json.dump(data, out, ensure_ascii=False, indent=2)

print(f"Done! {len(data)} files listed -> {OUTPUT}")
