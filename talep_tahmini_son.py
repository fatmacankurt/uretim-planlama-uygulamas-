# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 14:05:53 2025

@author: Fatma Cankurt
"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re



# =============================
#  Genel Ayarlar 
df_all_long = None   
df = None
last_forecast_df = None  

# Görsel/Font ayarları
UI_FONT = ("Segoe UI", 11)
UI_FONT_BOLD = ("Segoe UI", 11, "bold")
RESULT_FONT = ("Segoe UI", 12)


WMA_WINDOW = 4

# Ağırlık modu: "linear", "equal", "custom"
WEIGHTS_MODE = "linear"
CUSTOM_WEIGHTS = None  # Liste/array veya None

AY_ISIMLERI = ["Ocak","Şubat","Mart","Nisan","Mayıs","Haziran",
               "Temmuz","Ağustos","Eylül","Ekim","Kasım","Aralık"]



def sonuc_kutusu_yaz(metin: str):
    """Tek satırlık sonuç kutusu yazdırma (overwrite)."""
    sonuc_kutusu.config(state="normal")
    sonuc_kutusu.delete("1.0", tk.END)
    sonuc_kutusu.insert(tk.END, metin + "\n")
    sonuc_kutusu.config(state="disabled")


def temizle_grafik():
    for w in grafik_frame.winfo_children():
        w.destroy()


#  Ağırlıklı Hareketli Ortalama (WMA)


def _wma_agirliklari(k: int):
    
    global WEIGHTS_MODE, CUSTOM_WEIGHTS
    if k <= 0:
        raise ValueError("k > 0 olmalı")

    if WEIGHTS_MODE == "custom":
        if CUSTOM_WEIGHTS is None or len(CUSTOM_WEIGHTS) != k:
            raise ValueError("Özel ağırlıklar k ile aynı uzunlukta olmalı.")
        w = np.array(CUSTOM_WEIGHTS, dtype=float)
    elif WEIGHTS_MODE == "equal":
        w = np.ones(k, dtype=float)
    else:
        # linear: 1..k (en yeni gözlem en yüksek ağırlık alır)
        w = np.arange(1, k+1, dtype=float)

    s = w.sum()
    if s == 0:
        raise ValueError("Ağırlıkların toplamı 0 olamaz.")
    return w / s


def wma_next_value(history, window=None):
   
    if window is None:
        window = WMA_WINDOW
    if len(history) == 0:
        return np.nan
    k = min(len(history), int(window))
    w = _wma_agirliklari(k)
    segment = np.array(history[-k:], dtype=float)
    # Ağırlıklar en yeniye büyük gelecek şekilde hizalı (linear: 1..k)
   
    return float(np.dot(segment, w))


def wma_forecast_series(history, steps, window=None):
    
    if window is None:
        window = WMA_WINDOW
    hist = list(history)
    preds = []
    for _ in range(steps):
        yhat = wma_next_value(hist, window)
        preds.append(yhat)
        hist.append(yhat)
    return np.array(preds, dtype=float)


def sonraki_aylar(son_yil, son_ay_no, k):
   
    yil = int(son_yil); ay = int(son_ay_no)
    out = []
    for _ in range(k):
        ay += 1
        if ay == 13:
            ay = 1
            yil += 1
        out.append((yil, ay))
    return out


BASE_COLS = ["Yıl", "Kayıt dönemi", "Mamul", "SA siparişi miktarı"]


def _extract_suffix(colname, base="Yıl"):
    m = re.fullmatch(rf"{re.escape(base)}(\.\d+)?", str(colname))
    return (m.group(1) or "") if m else None   # ← soneksiz hali "" olarak döndür


def _mevcut_blok_suffixleri(cols):
    suffixes = []
    for c in cols:
        suf = _extract_suffix(c, base="Yıl")
        if suf is not None and suf not in suffixes:
            suffixes.append(suf)
    # "" (boş) öne, sonra .1, .2, ...
    return sorted(suffixes, key=lambda s: (0 if s=="" else int(s[1:])))


def _blok_kolonlari_var_mi(cols, suf):
    return all(((b if suf=="" else f"{b}{suf}") in cols) for b in BASE_COLS)


def _num_coerce(x):
    """"1.234,56" gibi yerel biçimleri de sayıya çevir."""
    if isinstance(x, str):
        xs = x.replace(".", "").replace(",", ".")
        try:
            return float(xs)
        except Exception:
            return np.nan
    return pd.to_numeric(x, errors="coerce")


#  Eksik ayları doldurma yardımcı fonksiyonu 
def fill_missing_months(uzun_df: pd.DataFrame, fill_value=0):
    
    if uzun_df.empty:
        return uzun_df.copy()

    cols = ["MamulKodu", "Yil", "Ay_no", "Satis"]
    df = uzun_df[cols].copy()

    # 0-bazlı ay anahtarı: 2024-01 -> 2024*12 + 0
    df["month_key"] = df["Yil"] * 12 + (df["Ay_no"] - 1)

    out_parts = []
    for mamul, grp in df.groupby("MamulKodu", as_index=False):
        min_key = int(grp["month_key"].min())
        max_key = int(grp["month_key"].max())

        full_keys = pd.Series(range(min_key, max_key + 1), name="month_key")

        merged = full_keys.to_frame().merge(
            grp[["month_key", "Satis"]], on="month_key", how="left"
        )

        # Eksikleri doldur (0 = satış yok)
        merged["Satis"] = merged["Satis"].fillna(fill_value)

        # month_key -> (Yil, Ay_no) geri dönüş
        merged["Yil"] = (merged["month_key"] // 12).astype(int)
        merged["Ay_no"] = (merged["month_key"] % 12 + 1).astype(int)

        merged["MamulKodu"] = mamul
        out_parts.append(merged[["MamulKodu", "Yil", "Ay_no", "Satis"]])

    filled = pd.concat(out_parts, ignore_index=True)
    filled["AyAd"] = filled["Ay_no"].map(lambda x: AY_ISIMLERI[(int(x)-1) % 12])

    filled.sort_values(["MamulKodu", "Yil", "Ay_no"], inplace=True)
    filled = filled[["Yil", "Ay_no", "AyAd", "MamulKodu", "Satis"]].reset_index(drop=True)
    return filled



def _blok_df_cek(ham_df, suf):
    cols = {b: (b if suf=="" else f"{b}{suf}") for b in BASE_COLS}
    alt = ham_df[[cols[b] for b in BASE_COLS]].copy()
    alt.columns = BASE_COLS

    
    alt = alt.dropna(subset=["Mamul", "SA siparişi miktarı"])  
    alt["Yıl"] = pd.to_numeric(alt["Yıl"], errors="coerce")
    alt["Kayıt dönemi"] = pd.to_numeric(alt["Kayıt dönemi"], errors="coerce")
    # Ondalık/ayırıcı toleransı
    alt["SA siparişi miktarı"] = alt["SA siparişi miktarı"].apply(_num_coerce)

    alt = alt.dropna(subset=["Yıl", "Kayıt dönemi", "SA siparişi miktarı"])  

    alt["Ay_no"] = alt["Kayıt dönemi"].astype(int)
    alt["AyAd"] = alt["Ay_no"].map(lambda x: AY_ISIMLERI[(int(x)-1)%12])
    alt["MamulKodu"] = alt["Mamul"].astype(str)
    alt.rename(columns={"SA siparişi miktarı":"Satis", "Yıl":"Yil"}, inplace=True)
    alt = alt[["Yil", "Ay_no", "AyAd", "MamulKodu", "Satis"]]

    return alt


def excel_to_long(path):
    ham = pd.read_excel(path)
    suffixes = _mevcut_blok_suffixleri(ham.columns)
    parcalar = []
    for suf in suffixes:
        if _blok_kolonlari_var_mi(ham.columns, suf):
            parcalar.append(_blok_df_cek(ham, suf))
    if not parcalar:
        raise ValueError("Beklenen sütunlar bulunamadı: 'Yıl', 'Kayıt dönemi', 'Mamul', 'SA siparişi miktarı' (ve .1/.2 …)")

    uzun = pd.concat(parcalar, ignore_index=True)
   
    uzun = (
        uzun.groupby(["MamulKodu", "Yil", "Ay_no", "AyAd"], as_index=False)["Satis"].mean()
    )

    
    if (uzun["Satis"] < 0).any():
        messagebox.showwarning("Uyarı", "Negatif satış tespit edildi ve yok sayıldı.")
        uzun = uzun.loc[uzun["Satis"] >= 0].copy()

    uzun.sort_values(["MamulKodu", "Yil", "Ay_no"], inplace=True)
    uzun.reset_index(drop=True, inplace=True)

   
    uzun = fill_missing_months(uzun, fill_value=0)
   
    return uzun


def excel_sec():
    """Dosya Seç ve mamul listesini doldur."""
    global df_all_long, df, last_forecast_df
    yol = filedialog.askopenfilename(filetypes=[("Excel files","*.xlsx")])
    if not yol:
        return
    try:
        df_all_long = excel_to_long(yol)
        kodlar = sorted(df_all_long["MamulKodu"].unique().tolist())
        mamul_var.set("")
        combo_mamul["values"] = kodlar
        combo_mamul.state(["!disabled"])
        sonuc_kutusu_yaz(f"Dosya yüklendi. {len(kodlar)} mamul bulundu. Lütfen mamul seçiniz. (Eksik aylar 0 ile tamamlandı)")
        df = None
        last_forecast_df = None
        temizle_grafik()
    except Exception as e:
        messagebox.showerror("Hata", f"Okuma/Dönüştürme hatası:\n{e}")


def mamul_secildi(*args):
    global df, last_forecast_df
    if df_all_long is None:
        return
    secim = mamul_var.get()
    if not secim:
        return

    alt = df_all_long[df_all_long["MamulKodu"] == secim].copy()
    if alt.empty:
        messagebox.showerror("Hata", "Seçilen mamul için veri bulunamadı.")
        return

    df = alt
    last_forecast_df = None
    sonuc_kutusu_yaz(f"Seçilen mamul: {secim} | WMA penceresi (k) = {WMA_WINDOW} | Ağırlık modu = {weights_var.get()}")
    df_sorted = df.sort_values(["Yil","Ay_no"])  # güvenlik için
    ciz_grafik_future(df_sorted, [], [])  # geçmişi çiz, gelecek boş


def ay_tahmin_dialog():
    """Belirli Ay Tahmini"""
    global df
    if df is None:
        messagebox.showerror("Hata", "Önce dosyayı yükleyip bir mamul seçiniz.")
        return

    df_sorted = df.sort_values(["Yil","Ay_no"])
    last_yil = int(df_sorted["Yil"].iloc[-1])

    top = tk.Toplevel(pencere)
    top.title("Belirli Ay Tahmini")
    top.resizable(False, False)

    ttk.Label(top, text="Hedef Ay:").grid(row=0, column=0, padx=8, pady=8, sticky="e")
    ttk.Label(top, text="Hedef Yıl:").grid(row=1, column=0, padx=8, pady=8, sticky="e")

    ay_var_loc = tk.StringVar(value=AY_ISIMLERI[0])
    yil_var_loc = tk.StringVar(value=str(last_yil))

    combo_ay = ttk.Combobox(top, textvariable=ay_var_loc, state="readonly", values=AY_ISIMLERI, width=18)
    combo_ay.grid(row=0, column=1, padx=8, pady=8)

 
    yil_list = [str(y) for y in range(last_yil, last_yil+4)]
    combo_yil = ttk.Combobox(top, textvariable=yil_var_loc, state="readonly", values=yil_list, width=18)
    combo_yil.grid(row=1, column=1, padx=8, pady=8)

    def on_ok():
        try:
            hedef_ay_no = AY_ISIMLERI.index(ay_var_loc.get()) + 1
            hedef_yil = int(yil_var_loc.get())

          
            last_yil_ = int(df_sorted["Yil"].iloc[-1])
            last_ay = int(df_sorted["Ay_no"].iloc[-1])
            steps = (hedef_yil - last_yil_) * 12 + (hedef_ay_no - last_ay)
            while steps <= 0:  
                steps += 12

            y_hist = df_sorted["Satis"].values.astype(float)
            preds = wma_forecast_series(y_hist, steps=steps, window=WMA_WINDOW)
            y_pred = float(preds[-1])

            sonuc_kutusu_yaz(f"{ay_var_loc.get()} {hedef_yil} için tahmin: {y_pred:.2f} (k={WMA_WINDOW}, ağırlık modu={weights_var.get()})")

            future_pairs = sonraki_aylar(last_yil_, last_ay, steps)
            ciz_grafik_future(df_sorted, future_pairs, preds.tolist())
            top.destroy()
        except Exception as e:
            messagebox.showerror("Hata", f"Tahmin oluşturulamadı:\n{e}")

    btn_ok = ttk.Button(top, text="Tahmin Oluştur", command=on_ok)
    btn_ok.grid(row=2, column=0, columnspan=2, padx=8, pady=(4,10))


def n_ay_tahmin():
    global df
    if df is None:
        messagebox.showerror("Hata", "Önce dosyayı yükleyip bir mamul seçiniz.")
        return
    n = simpledialog.askinteger("Tahmin", "Kaç ay ileri?", minvalue=1)
    if not n:
        return

    df_sorted = df.sort_values(["Yil","Ay_no"])
    last_yil = int(df_sorted["Yil"].iloc[-1])
    last_ay = int(df_sorted["Ay_no"].iloc[-1])

    y_hist = df_sorted["Satis"].values.astype(float)
    preds = wma_forecast_series(y_hist, steps=n, window=WMA_WINDOW)

    future_pairs = sonraki_aylar(last_yil, last_ay, n)
    satirlar = [f"{AY_ISIMLERI[ay-1]} {yil}: {v:.2f}" for (yil, ay), v in zip(future_pairs, preds)]
    sonuc_kutusu_yaz("\n".join(satirlar))

    ciz_grafik_future(df_sorted, future_pairs, preds.tolist())



def _inceltme_indeksleri(labels, max_labels=28):
    
    L = len(labels)
    if L <= max_labels:
        return list(range(L))
    step = int(np.ceil(L / max_labels))
    return list(range(0, L, step))


def ciz_grafik_future(df_sorted, future_pairs, y_future):
    global last_forecast_df
    temizle_grafik()
    if df_sorted is None or df_sorted.empty:
        return

    fig, ax = plt.subplots(figsize=(9.6, 4.8))

    past_labels = [f"{AY_ISIMLERI[int(ay)-1]} {int(yil)}"
                   for yil, ay in zip(df_sorted["Yil"], df_sorted["Ay_no"])]
    x_past = list(range(1, len(df_sorted)+1))
    ax.plot(x_past,
            df_sorted["Satis"].values,
            "ro-", label="Gerçek Satışlar",
            linewidth=1.8, markersize=4.5)

    if future_pairs:
        start = len(df_sorted)
        t_future = list(range(start+1, start+1+len(future_pairs)))
        ax.plot(t_future, y_future, "go--",
                label="Tahmin (WMA)",
                linewidth=1.8, markersize=4.5)
        future_labels = [f"{AY_ISIMLERI[ay-1]} {yil}" for (yil, ay) in future_pairs]
        xticks = x_past + t_future
        xlabels = past_labels + future_labels

       
        past_df = df_sorted.copy()
        past_df = past_df.assign(Tip="Gerçek")
        fut_df = pd.DataFrame({
            "Yil": [y for (y, a) in future_pairs],
            "Ay_no": [a for (y, a) in future_pairs],
            "AyAd": [AY_ISIMLERI[a-1] for (y, a) in future_pairs],
            "MamulKodu": df_sorted["MamulKodu"].iloc[0] if not df_sorted.empty else "",
            "Satis": y_future,
            "Tip": "Tahmin"
        })
        last_forecast_df = pd.concat([past_df, fut_df], ignore_index=True)
    else:
        xticks = x_past
        xlabels = past_labels
        past_df = df_sorted.copy()
        past_df = past_df.assign(Tip="Gerçek")
        last_forecast_df = past_df

    
    show_idx = _inceltme_indeksleri(xlabels, max_labels=28)
    show_xticks = [xticks[i] for i in show_idx]
    show_xlabels = [xlabels[i] for i in show_idx]

    ax.set_xticks(show_xticks)
    ax.set_xticklabels(show_xlabels, rotation=45, fontsize=8)
    ax.set_ylabel("Satış", fontsize=10)
    ax.set_title("Satış Tahminleri (WMA)", fontsize=13, pad=8)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.2, linewidth=0.6)
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=grafik_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # Belleği boşalt (canvas kopyasını tutuyor)
    plt.close(fig)



def ciz_grafik(t_future, y_future):
    if df is None or df.empty:
        return
    df_sorted = df.sort_values(["Yil","Ay_no"])
    ciz_grafik_future(df_sorted, [], [])



def disari_aktar_grafik():
    
    if df is None or df.empty:
        messagebox.showerror("Hata", "Önce bir mamul seçip grafik oluşturunuz.")
        return
    yol = filedialog.asksaveasfilename(defaultextension=".png",
                                       filetypes=[("PNG","*.png"), ("SVG","*.svg"), ("PDF","*.pdf")])
    if not yol:
        return

   
    if last_forecast_df is not None and not last_forecast_df.empty:
        df_plot = last_forecast_df.copy()
        labels = [f"{AY_ISIMLERI[int(a)-1]} {int(y)}" for y, a in zip(df_plot["Yil"], df_plot["Ay_no"])]
        x = list(range(1, len(labels)+1))

        fig, ax = plt.subplots(figsize=(9.6, 4.8))
        
        msk_g = df_plot["Tip"] == "Gerçek"
        x_g = [i+1 for i, v in enumerate(msk_g) if v]
        y_g = df_plot.loc[msk_g, "Satis"].values
        ax.plot(x_g, y_g, "ro-", label="Gerçek Satışlar", linewidth=1.8, markersize=4.5)
        
        msk_t = df_plot["Tip"] == "Tahmin"
        if msk_t.any():
            x_t = [i+1 for i, v in enumerate(msk_t) if v]
            y_t = df_plot.loc[msk_t, "Satis"].values
            ax.plot(x_t, y_t, "go--", label="Tahmin (WMA)", linewidth=1.8, markersize=4.5)

        idx = _inceltme_indeksleri(labels, max_labels=28)
        ax.set_xticks([x[i] for i in idx])
        ax.set_xticklabels([labels[i] for i in idx], rotation=45, fontsize=8)
    else:
        df_sorted = df.sort_values(["Yil","Ay_no"])
        labels = [f"{AY_ISIMLERI[int(ay)-1]} {int(yil)}" for yil, ay in zip(df_sorted["Yil"], df_sorted["Ay_no"])]
        x = list(range(1, len(labels)+1))
        fig, ax = plt.subplots(figsize=(9.6, 4.8))
        ax.plot(x, df_sorted["Satis"].values, "ro-", label="Gerçek Satışlar", linewidth=1.8, markersize=4.5)
        idx = _inceltme_indeksleri(labels, max_labels=28)
        ax.set_xticks([x[i] for i in idx])
        ax.set_xticklabels([labels[i] for i in idx], rotation=45, fontsize=8)

    ax.set_ylabel("Satış", fontsize=10)
    ax.set_title("Satış Tahminleri (WMA)", fontsize=13, pad=8)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(yol, dpi=180, bbox_inches="tight")
    plt.close(fig)
    messagebox.showinfo("Bilgi", f"Grafik kaydedildi:\n{yol}")


def disari_aktar_tahmin_excel():
    global last_forecast_df
    if last_forecast_df is None or last_forecast_df.empty:
        messagebox.showerror("Hata", "Önce N Ay veya Belirli Ay tahmini üretiniz.")
        return
    yol = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                       filetypes=[("Excel","*.xlsx")])
    if not yol:
        return
    try:
        # Sütun sırası
        cols = ["MamulKodu", "Yil", "Ay_no", "AyAd", "Satis", "Tip"]
        df_out = last_forecast_df[cols].copy()
        with pd.ExcelWriter(yol) as writer:  # varsayılan engine (openpyxl)
            df_out.to_excel(writer, index=False, sheet_name="tahmin")
        messagebox.showinfo("Bilgi", f"Tahmin tablosu kaydedildi:\n{yol}")
    except Exception as e:
        messagebox.showerror("Hata", f"Kaydetme hatası:\n{e}")


pencere = tk.Tk()
pencere.title("Talep Tahmini — Çok Mamullü Excel (WMA)")
pencere.geometry("1000x800")

style = ttk.Style()
style.configure(".", font=UI_FONT)

# Menü çubuğu
menubar = tk.Menu(pencere)
menu_dosya = tk.Menu(menubar, tearoff=0)
menu_dosya.add_command(label="Excel Aç", command=excel_sec)
menu_dosya.add_separator()
menu_dosya.add_command(label="Grafiği Kaydet…", command=disari_aktar_grafik)
menu_dosya.add_command(label="Tahmin Tablosunu Kaydet…", command=disari_aktar_tahmin_excel)
menu_dosya.add_separator()
menu_dosya.add_command(label="Çıkış", command=pencere.destroy)
menubar.add_cascade(label="Dosya", menu=menu_dosya)
pencere.config(menu=menubar)

# Üst buton çubuğu
frame_btn = ttk.Frame(pencere)
frame_btn.pack(pady=10)

ttk.Button(frame_btn, text="Dosya Seç", command=excel_sec).grid(row=0, column=0, padx=6)
ttk.Button(frame_btn, text="Belirli Ay Tahmini", command=ay_tahmin_dialog).grid(row=0, column=1, padx=6)
ttk.Button(frame_btn, text="N Ay Tahmini", command=n_ay_tahmin).grid(row=0, column=2, padx=6)

# k (WMA penceresi) seçimi + Ağırlık modu
k_frame = ttk.Frame(pencere)
k_frame.pack(pady=(2, 6))
ttk.Label(k_frame, text="WMA penceresi (k):").grid(row=0, column=0, padx=(0,6))

k_var = tk.StringVar(value=str(WMA_WINDOW))
combo_k = ttk.Combobox(k_frame, textvariable=k_var, width=6, state="readonly",
                       values=["3","4","6","12"])
combo_k.grid(row=0, column=1, padx=(0,12))

ttk.Label(k_frame, text="Ağırlık modu:").grid(row=0, column=2, padx=(0,6))
weights_var = tk.StringVar(value="Lineer (1..k)")
combo_w = ttk.Combobox(
    k_frame,
    textvariable=weights_var,
    width=18,
    state="readonly",
    values=["Lineer (1..k)", "Eşit (SMA)", "Özel gir…"]
)
combo_w.grid(row=0, column=3)

def _refresh_plot_and_status():
    if df is not None and not df.empty:
        df_sorted = df.sort_values(["Yil","Ay_no"])
        ciz_grafik_future(df_sorted, [], [])
    sonuc_kutusu_yaz(f"WMA penceresi (k) {WMA_WINDOW} | Ağırlık modu = {weights_var.get()}")

def k_degisti(*args):
    global WMA_WINDOW, CUSTOM_WEIGHTS
    try:
        yeni_k = int(k_var.get())
        if yeni_k <= 0:
            raise ValueError
        WMA_WINDOW = yeni_k
        
        if WEIGHTS_MODE == "custom" and (CUSTOM_WEIGHTS is None or len(CUSTOM_WEIGHTS) != WMA_WINDOW):
            CUSTOM_WEIGHTS = None
            weights_var.set("Lineer (1..k)")
            _weights_mode_changed()
        _refresh_plot_and_status()
    except Exception:
        messagebox.showerror("Hata", "Geçersiz k değeri.")

def _weights_mode_changed(*args):
    global WEIGHTS_MODE, CUSTOM_WEIGHTS
    secim = weights_var.get()
    if secim.startswith("Lineer"):
        WEIGHTS_MODE = "linear"
        CUSTOM_WEIGHTS = None
        _refresh_plot_and_status()
    elif secim.startswith("Eşit"):
        WEIGHTS_MODE = "equal"
        CUSTOM_WEIGHTS = None
        _refresh_plot_and_status()
    else:
        # Özel gir…
        txt = simpledialog.askstring(
            "Özel Ağırlıklar",
            f"k={WMA_WINDOW} için ağırlıkları virgülle giriniz (ör. 1,2,3,4). Toplam otomatik normalize edilir."
        )
        if not txt:
            # İptal: önceki moda dön
            weights_var.set("Lineer (1..k)")
            WEIGHTS_MODE = "linear"
            CUSTOM_WEIGHTS = None
            return
        try:
            parcalar = [p.strip() for p in txt.split(",")]
            arr = np.array([float(p) for p in parcalar], dtype=float)
            if len(arr) != WMA_WINDOW:
                raise ValueError(f"Uzunluk {len(arr)} ≠ k={WMA_WINDOW}")
            if np.all(arr == 0):
                raise ValueError("Ağırlıkların tamamı 0 olamaz.")
            CUSTOM_WEIGHTS = arr.tolist()
            WEIGHTS_MODE = "custom"
            
            _refresh_plot_and_status()
        except Exception as e:
            messagebox.showerror("Hata", f"Ağırlıklar okunamadı:\n{e}")
            weights_var.set("Lineer (1..k)")
            WEIGHTS_MODE = "linear"
            CUSTOM_WEIGHTS = None
            _refresh_plot_and_status()

k_var.trace_add("write", k_degisti)
combo_w.bind("<<ComboboxSelected>>", _weights_mode_changed)

# Mamul seçimi
mamul_var = tk.StringVar()
mamul_var.trace_add("write", mamul_secildi)

mamul_frame = ttk.Frame(pencere)
mamul_frame.pack(pady=(2, 8))
ttk.Label(mamul_frame, text="Mamul Kodu:").grid(row=0, column=0, padx=(0,6))
combo_mamul = ttk.Combobox(mamul_frame, textvariable=mamul_var, state="disabled", width=36)
combo_mamul.grid(row=0, column=1)


sonuc_kutusu = tk.Text(pencere, height=5, width=120, state="disabled", bg="#f5f5f5", font=RESULT_FONT)
sonuc_kutusu.pack(padx=12, pady=8, fill="x")


grafik_frame = ttk.Frame(pencere)
grafik_frame.pack(fill="both", expand=True, padx=10, pady=6)

pencere.mainloop()
