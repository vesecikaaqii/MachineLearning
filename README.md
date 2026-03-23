<table border="0">
 <tr>
    <td><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/1200px-University_of_Prishtina_logo.svg.png" width="150" /></td>
    <td>
      <p><strong>Universiteti i Prishtinës</strong></p>
      <p>Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</p>
      <p>Inxhinieri Kompjuterike dhe Softuerike - Master</p>
      <p>Profesor: Prof. Lule Ahmedi</p>
      <p>Asistent: Prof. Mergim Hoti</p>
    </td>
 </tr>
</table>

---

# 🌦️ Machine Learning Project  
## Training Weather Forecasting Models in Kosovo

👨‍💻 **Punuan:** Vesë Cikaqi, Uranik Hodaj, Dafina Keqmezi  
📅 **Viti Akademik:** 2025/2026  

---

# 📊 REZULTATET E PROJEKTIT

## 📁 Përmbledhje e Dataset-it

| Parametri                     | Vlera Reale |
|-----------------------------|------------|
| Numri i qyteteve            | 27         |
| Parashikime për qytet       | 40         |
| Të dhëna aktuale / qytet    | 1          |
| Rreshta total (për run)     | ~1107      |
| Numri i kolonave            | 20         |
| Intervali kohor             | 3 orë      |
| Periudha e mbulimit         | 5 ditë     |

---

## 🌍 Qytetet e Analizuara (Shembull)

| Qyteti     | Rajoni     |
|-----------|-----------|
| Pristina  | Prishtinë |
| Prizren   | Prizren   |
| Peja      | Pejë      |
| Gjakova   | Gjakovë   |
| Mitrovica | Mitrovicë |
| Ferizaj   | Ferizaj   |
| Gjilan    | Gjilan    |

---

## 📈 Statistika Reale nga Dataset-i (Shembull)

| Qyteti   | Temperatura Min (°C) | Temperatura Max (°C) | Lagështia (%) | Gjendja |
|----------|---------------------|---------------------|--------------|--------|
| Pristina | 5.60                | 13.58               | 44 – 90      | Clouds / Clear |
| Prizren  | ~6.00               | ~15.00              | 40 – 85      | Clear / Clouds |
| Peja     | ~5.00               | ~14.00              | 50 – 88      | Clouds |
| Gjakova  | ~6.50               | ~15.50              | 45 – 80      | Clear |
| Mitrovica| ~5.50               | ~13.00              | 50 – 85      | Clouds |

---

## ⏱️ Struktura Kohore e të Dhënave

| Data        | Intervalet |
|------------|-----------|
| 2026-03-23 | çdo 3 orë |
| 2026-03-24 | çdo 3 orë |
| 2026-03-25 | çdo 3 orë |
| 2026-03-26 | çdo 3 orë |
| 2026-03-27 | çdo 3 orë |

---

## 📊 Shembull i të Dhënave Reale

| City     | Type     | Datetime            | Temp (°C) | Humidity (%) | Weather |
|----------|----------|---------------------|-----------|-------------|--------|
| Pristina | forecast | 2026-03-23 19:00    | 9.65      | 60          | Clouds |
| Pristina | forecast | 2026-03-24 01:00    | 5.60      | 90          | Clouds |
| Pristina | forecast | 2026-03-25 13:00    | 13.58     | 44          | Clear  |

---

## 🌡️ Analiza e Temperaturave

| Kategoria            | Rezultati |
|---------------------|----------|
| Qytetet më të nxehta| Prishtina, Prizreni |
| Qytetet më të ftohta| Dragash, Deçan |
| Diferenca mesatare  | ~7°C – 10°C |
| Fenomen i vërejtur  | Urban Heat Island |

---

## 💨 Analiza e Erës dhe Presionit

| Parametri       | Intervali |
|----------------|----------|
| Wind Speed     | 0.5 – 6 m/s |
| Pressure       | 1010 – 1018 hPa |
| Wind Direction | 0° – 360° |

---

## 💧 Analiza e Lagështisë

| Parametri        | Vlera |
|------------------|------|
| Lagështia min    | ~40% |
| Lagështia max    | ~90% |
| Mesatarja        | ~65% |

---

## 🌧️ Probabiliteti i Reshjeve

| Parametri | Vlera |
|----------|------|
| POP min  | 0.0  |
| POP max  | 1.0  |
| Mesatarja| ~0.3 |

---

## 🤖 Modelet e Machine Learning

| Modeli             | Statusi | Qëllimi |
|--------------------|--------|--------|
| Linear Regression  | Planned | Parashikimi i temperaturës |
| Random Forest      | Planned | Klasifikimi i motit |
| LSTM               | Planned | Parashikim kohor |
| XGBoost            | Planned | Temperatura ekstreme |
| K-Means            | Planned | Grupimi i qyteteve |

