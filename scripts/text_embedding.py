import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

CLEANED_PATH = "scopus_cleaned.csv"   # แก้ path ตามตำแหน่งไฟล์บน Colab
OUT_EMBED = "text_embeddings.npy"
OUT_META = "text_embeddings_meta.csv"

# 1. โหลดข้อมูล + เตรียมข้อความ (ให้ตรงกับสคริปต์ Feature Vector เพื่อรวมกันได้ทีหลัง)
df = pd.read_csv(CLEANED_PATH, low_memory=False).reset_index(drop=True)
df["row_id"] = df.index

def combine_text(row):
    parts = []
    if pd.notna(row.get("Title")) and str(row["Title"]).strip():
        parts.append(f"Title: {row['Title']}")
    if pd.notna(row.get("Abstract")) and str(row["Abstract"]).strip():
        parts.append(f"Abstract: {row['Abstract']}")
    if pd.notna(row.get("Author Keywords")) and str(row["Author Keywords"]).strip():
        parts.append(f"Author keywords: {row['Author Keywords']}")
    if pd.notna(row.get("Index Keywords")) and str(row["Index Keywords"]).strip():
        parts.append(f"Index keywords: {row['Index Keywords']}")
    return "\n".join(parts)

df["embedding_text"] = df.apply(combine_text, axis=1)
texts = df["embedding_text"].tolist()
print("จำนวนข้อความทั้งหมด:", len(texts))

# 2. โหลด SPECTER2 (เหมือนตัวอย่างของอาจารย์)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
model = model.to(device)
model.eval()

# 3. สร้าง embedding เป็น batch (SPECTER2 ใช้ [CLS] token, จำกัด 512 tokens)
def create_specter2_embeddings(texts, batch_size=16, max_length=512):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Creating SPECTER2 embeddings"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        all_embeddings.append(batch_embeddings)
    return np.vstack(all_embeddings)

embeddings = create_specter2_embeddings(texts, batch_size=16, max_length=512)
print("Embedding shape:", embeddings.shape)   # คาดว่าจะได้ (6000, 768)

# 4. บันทึกผลลัพธ์ (แยกไฟล์ embedding ตัวเลข กับ meta สำหรับเชื่อมกลับไปหาบทความต้นฉบับ)
np.save(OUT_EMBED, embeddings)
df[["row_id", "EID", "Title", "Subject"]].to_csv(OUT_META, index=False)
print(f"บันทึก {OUT_EMBED} และ {OUT_META} เรียบร้อย")
print("ดาวน์โหลดทั้ง 2 ไฟล์กลับมาใส่ Data/processed/ ในโปรเจกต์ แล้ว push ขึ้น GitHub")
