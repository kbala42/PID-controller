import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# -----------------------------
# Streamlit temel ayar
# -----------------------------
st.set_page_config(page_title="PID Kontrol Labı", page_icon="⚙️")

st.title("⚙️ PID Kontrol Playground – Oda Isıtma Sistemi")
st.write(
    """
Bu laboratuvarda basit bir **oda ısıtma sistemini** PID kontrol ile yöneteceksin.

- Hedef sıcaklığı (setpoint) belirle  
- P / I / D katsayılarını ayarla  
- Oda sıcaklığının zamana bağlı grafiğini incele  
- Farklı ayarların overshoot (hedefi aşma), hız ve dalgalanma üzerindeki etkisini gözlemle
"""
)

st.markdown("---")


# -----------------------------
# Sistem parametreleri
# -----------------------------
st.subheader("1️⃣ Sistem Parametrelerini Seç")

col_sys1, col_sys2, col_sys3 = st.columns(3)

with col_sys1:
    T_ambient = st.slider(
        "Ortam sıcaklığı (°C)",
        min_value=0.0,
        max_value=30.0,
        value=20.0,
        step=1.0,
    )
with col_sys2:
    T_set = st.slider(
        "Hedef sıcaklık / Setpoint (°C)",
        min_value=15.0,
        max_value=30.0,
        value=22.0,
        step=0.5,
    )
with col_sys3:
    tau = st.slider(
        "Sistemin zaman sabiti τ (s)",
        min_value=10.0,
        max_value=200.0,
        value=60.0,
        step=10.0,
        help="Sistem ne kadar yavaş/ataletli. τ büyüdükçe oda ısınması yavaşlar.",
    )

k_heat = st.slider(
    "Isıtıcı kazancı (k_heat)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1,
    help="Isıtıcının etkisi. Ne kadar büyükse, aynı güçte oda daha hızlı ısınır.",
)

st.write(
    f"Seçilen sistem: ortam sıcaklığı **{T_ambient:.1f}°C**, "
    f"hedef **{T_set:.1f}°C**, zaman sabiti **τ = {tau:.0f} s**, "
    f"ısıtıcı kazancı **k_heat = {k_heat:.2f}**"
)


# -----------------------------
# PID parametreleri
# -----------------------------
st.subheader("2️⃣ PID Parametrelerini Ayarla")

col_pid1, col_pid2, col_pid3 = st.columns(3)

with col_pid1:
    Kp = st.slider(
        "Kp (P kazancı)",
        min_value=0.0,
        max_value=5.0,
        value=1.5,
        step=0.1,
    )
with col_pid2:
    Ki = st.slider(
        "Ki (I kazancı)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
    )
with col_pid3:
    Kd = st.slider(
        "Kd (D kazancı)",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
    )

st.write(
    f"PID parametreleri: **Kp = {Kp:.2f}**, **Ki = {Ki:.2f}**, **Kd = {Kd:.2f}**"
)

st.caption(
    "İpucu: Önce sadece P ile başlayıp (Ki=0, Kd=0) davranışı gözle, "
    "sonra I ve D bileşenlerini yavaş yavaş ekle."
)


# -----------------------------
# Simülasyon ayarları
# -----------------------------
st.subheader("3️⃣ Simülasyon Ayarları")

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    T_initial = st.slider(
        "Başlangıç sıcaklığı T₀ (°C)",
        min_value=0.0,
        max_value=30.0,
        value=18.0,
        step=0.5,
    )
with col_sim2:
    t_max = st.slider(
        "Toplam simülasyon süresi (s)",
        min_value=60.0,
        max_value=600.0,
        value=300.0,
        step=30.0,
    )

dt = st.slider(
    "Zaman adımı Δt (s)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
)

n_steps = int(t_max / dt) + 1
st.write(
    f"Simülasyon süresi **{t_max:.0f} s**, zaman adımı **Δt = {dt:.1f} s**, "
    f"toplam adım sayısı: **{n_steps}**"
)


# -----------------------------
# PID simülasyon fonksiyonu
# -----------------------------
def simulate_pid_room(
    T_ambient,
    T_set,
    T_initial,
    tau,
    k_heat,
    Kp,
    Ki,
    Kd,
    dt,
    n_steps,
):
    """
    Basit oda ısıtma modelinde PID kontrol simülasyonu.
    model:
        dT/dt = -(T - T_ambient)/tau + k_heat * (u/100)
    PID:
        u = Kp*e + Ki*∫e dt + Kd*de/dt
    """
    t = np.zeros(n_steps)
    T = np.zeros(n_steps)
    u = np.zeros(n_steps)
    e = np.zeros(n_steps)

    T[0] = T_initial
    e[0] = T_set - T[0]

    integral = 0.0
    prev_error = e[0]

    for k in range(n_steps - 1):
        # Hata
        error = T_set - T[k]
        e[k] = error

        # Integral ve türev
        integral += error * dt
        derivative = (error - prev_error) / dt

        # PID denetleyici
        u_raw = Kp * error + Ki * integral + Kd * derivative

        # Kontrol sinyalini sınırla (0–100%)
        u[k] = np.clip(u_raw, 0.0, 100.0)

        # Oda sıcaklık modelini güncelle
        dTdt = -(T[k] - T_ambient) / tau + k_heat * (u[k] / 100.0)
        T[k + 1] = T[k] + dTdt * dt

        # Zamanı güncelle
        t[k + 1] = t[k] + dt

        # Sonraki adım için
        prev_error = error

    # Son adımın hatasını doldur
    e[-1] = T_set - T[-1]
    # Son kontrol sinyalini tekrarla
    u[-1] = u[-2]

    return t, T, u, e


# Simülasyonu çalıştır
t, T, u, e = simulate_pid_room(
    T_ambient,
    T_set,
    T_initial,
    tau,
    k_heat,
    Kp,
    Ki,
    Kd,
    dt,
    n_steps,
)


# -----------------------------
# Grafikleri çiz
# -----------------------------
st.markdown("---")
st.subheader("4️⃣ Sıcaklık ve Kontrol Sinyali Grafikleri")

fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(t, T, label="Oda sıcaklığı T(t)")
ax1.axhline(T_set, color="gray", linestyle="--", label="Setpoint (hedef)")
ax1.set_xlabel("t (s)")
ax1.set_ylabel("Sıcaklık (°C)")
ax1.set_title("PID Kontrol Altında Oda Sıcaklığının Zamanla Değişimi")
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax1.legend()

st.pyplot(fig1)

st.subheader("Kontrol Sinyali (Isıtıcı Gücü)")

fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.plot(t, u, label="u(t) – Isıtıcı gücü (%)")
ax2.set_xlabel("t (s)")
ax2.set_ylabel("u(t) (%)")
ax2.set_ylim(-5, 105)
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax2.legend()

st.pyplot(fig2)


# -----------------------------
# İlk adımlar için tablo
# -----------------------------
st.subheader("5️⃣ İlk Adımların Sayısal Tablosu")

max_rows = min(20, n_steps)
df = pd.DataFrame(
    {
        "t (s)": t[:max_rows],
        "T(t) (°C)": T[:max_rows],
        "u(t) (%)": u[:max_rows],
        "Hata e(t)": e[:max_rows],
    }
)

st.dataframe(
    df.style.format(
        {
            "t (s)": "{:.1f}",
            "T(t) (°C)": "{:.2f}",
            "u(t) (%)": "{:.2f}",
            "Hata e(t)": "{:.2f}",
        }
    )
)


# -----------------------------
# Açıklama / Öğretmen kutusu
# -----------------------------
st.markdown("---")
st.info(
    "P bileşeni anlık hataya, I bileşeni geçmiş hata birikimine, "
    "D bileşeni ise hatanın değişim hızına bakarak kontrol sinyalini üretir. "
    "Amaç: Sıcaklığı hedefe hızlı ama kararlı bir şekilde ulaştırmak."
)

with st.expander("👩‍🏫 Öğretmen Kutusu – P / I / D Bileşenlerinin Rolü"):
    st.write(
        r"""
**P (Proportional):**

- Denetleyici çıkışının hatayla orantılı kısmı: \\(P = K_p e(t)\\)  
- Hata büyükken güçlü tepki, hata küçükken zayıf tepki verir.  
- Sadece P kullanılırsa, sistem çoğu zaman **hızlı** ama bazen **kalıcı hatalı** olabilir.

---

**I (Integral):**

- Geçmiş hataların toplamını dikkate alır:  
  \\(I = K_i \int e(t) \, dt\\)  
- Hata uzun süre küçük de olsa sıfırlamaya çalışır.  
- Steady-state error (kalıcı hata) azaltılır; fakat I çok büyükse sistem sallanıp **overshoot** yapabilir.

---

**D (Derivative):**

- Hatanın değişim hızına bakar:  
  \\(D = K_d \frac{d e(t)}{dt}\\)  
- Hata hızla değişiyorsa, gelecekte ne olacağını öngörüp fren görevi görür.  
- D bileşeni overshoot'u azaltmaya ve sistemi sakinleştirmeye yardımcı olur, ancak gürültüye hassastır.

---

Bu labda öğrenciler:

1. Sadece **P** ile başlayıp tepkiyi gözlemler,  
2. **I** ekleyerek kalıcı hatayı azaltır ama overshoot'u fark eder,  
3. **D** ekleyerek daha yumuşak, daha kontrollü bir tepki elde etmeye çalışır.

Böylece PID denetimin üç bileşeninin rolünü üretim hatları, robot kolları,
oda ısıtma sistemleri gibi gerçek dünyadaki uygulamalara bağlayabiliriz.
"""
    )

st.caption(
    "Bu modül, lise düzeyinde otomasyon ve kontrol kavramlarına sezgisel bir giriş sağlamak için tasarlanmıştır."
)
