import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor, GPT2Tokenizer
from dataset import CWRUDataset
from model import SignalCaptioningModel

# --- AYARLAR ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"🚀 Cuda aktif! Cihaz: {torch.cuda.get_device_name(0)}")

EPOCHS = 20          # Uzun soluklu, hassas eğitim
BATCH_SIZE = 4
LEARNING_RATE = 1e-5 # DİKKAT: Hızı düşürdük (Fine-tuning için şart)
PATIENCE = 5         

print(f"İleri Seviye Eğitim (Fine-Tuning) {device} üzerinde başlıyor...")

# --- HAZIRLIK ---
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Dataset (RAM Cache)
print("Veri seti hazırlanıyor...")
dataset = CWRUDataset('data/raw', feature_extractor)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modeli Başlat
model = SignalCaptioningModel("MIT/ast-finetuned-audioset-10-10-0.4593", "gpt2").to(device)

# --- KRİTİK HAMLE 1: ESKİ MODELİ YÜKLE ---
best_model_path = "motor_model_best.pth"
if os.path.exists(best_model_path):
    print(f"📥 Önceki şampiyon model yükleniyor: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("✅ Kaldığımız yerden devam ediyoruz!")
else:
    print("⚠️ Kayıtlı model bulunamadı, sıfırdan başlanıyor (Önerilmez).")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- KRİTİK HAMLE 2: SCHEDULER (AKILLI HIZ KONTROLÜ) ---
# DÜZELTME: 'verbose=True' kaldırıldı (Yeni PyTorch sürümlerinde hata veriyor)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Takip Değişkenleri
best_loss = float('inf') 
patience_counter = 0     

model.train()

print("\n🔥 Hassas Eğitim Başlıyor! Hedef: Arıza Boyutlarını Öğretmek.\n")

for epoch in range(EPOCHS):
    total_loss = 0
    
    # MEVCUT HIZI BİZ ELLE YAZDIRALIM (Verbose yerine)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"--- Epoch {epoch+1}/{EPOCHS} Başladı (LR: {current_lr:.8f}) ---")
    
    for batch_idx, (signals, labels) in enumerate(train_loader):
        signals = signals.to(device)
        targets = tokenizer(labels, 
                            return_tensors="pt", 
                            padding="max_length", 
                            max_length=20, 
                            truncation=True).input_ids.to(device)
        
        optimizer.zero_grad()
        outputs = model(signals, labels=targets)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 20 == 0: 
            print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # Epoch sonu
    avg_loss = total_loss / len(train_loader)
    print(f"🏁 Epoch {epoch+1} Bitti. Ort. Kayıp: {avg_loss:.4f}")

    # --- SCHEDULER GÜNCELLEME ---
    # Loss durumuna göre hızı ayarla (Arka planda çalışır)
    scheduler.step(avg_loss)
    
    # Hız düştü mü diye kontrol edelim
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr < current_lr:
        print(f"📉 DİKKAT: Öğrenme hızı düşürüldü! ({current_lr:.8f} -> {new_lr:.8f})")

    # --- KAYIT MANTIĞI ---
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), "motor_model_best_tuned.pth") 
        print(f"💾 DAHA İYİSİ BULUNDU! Kaydedildi: motor_model_best_tuned.pth (Loss: {best_loss:.4f})")
    else:
        patience_counter += 1
        print(f"⏳ İyileşme yok. Sabır: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print(f"\n🛑 ERKEN DURDURMA! Model artık limitlerine ulaştı.")
            break

print("\n✅ Hassas Eğitim Tamamlandı!")