<table border="0">
 <tr>
    <td><img src="https://github.com/user-attachments/assets/9002855f-3f97-4b41-a180-85d1e24ad34a" alt="University Logo" width="110" align="left"/></td>
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

---

# 🧩 FAZA I — Përgatitja e të Dhënave (Data Preparation)

## 🎯 Qëllimi i fazës
Në Fazën I është ndërtuar baza e gjithë projektit: **mbledhja, strukturimi dhe pastrimi fillestar i një dataset-i meteorologjik real për Kosovën.** Pa të dhëna cilësore dhe të strukturuara, asnjë model i mësimit të makinës nuk mund të japë rezultate të besueshme, prandaj kjo fazë është konsideruar si gur-themeli i projektit.

## 🛠️ Detyrat e realizuara
1. **Identifikimi i burimit të të dhënave** — është përdorur [OpenWeatherMap API](https://openweathermap.org/api), një burim publik dhe i besueshëm për parashikime meteorologjike globale.
2. **Përzgjedhja e 27 qyteteve të Kosovës** me koordinata (lat, lon) për të mbuluar të gjitha rajonet.
3. **Ndërtimi i skriptit [`weather_data_scraper.py`](weather_data_scraper.py)** që:
   - për çdo qytet merr parashikimin 5-ditor me interval 3 orë (endpoint `forecast`),
   - dhe të dhënën aktuale të momentit (endpoint `weather`).
4. **Ruajtja në CSV** me emër [`kosovo_weather_dataset.csv`](kosovo_weather_dataset.csv), duke shtuar automatikisht kolonat `hour`, `day`, `month` për analiza kohore.
5. **Verifikimi i integritetit** (pa duplikate, pa rreshta bosh në targetet kryesore).

## 📄 Detajet e Dataset-it

| Parametri                  | Vlera |
|----------------------------|-------|
| Burimi                     | OpenWeatherMap API (falas, publik) |
| Numri i qyteteve           | 27 |
| Numri i atributeve (kolona)| **20** |
| Numri i instancave (rreshta)| **1107** |
| Madhësia e dataset-it      | ~142 KB (CSV) |
| Intervali kohor            | çdo 3 orë |
| Periudha e mbuluar         | ~5 ditë parashikim + vlera aktuale |
| Formati                    | CSV, i lexueshëm me `pandas` |

### Atributet (20)
`city`, `type`, `datetime`, `temperature`, `feels_like`, `temp_min`, `temp_max`, `pressure`, `humidity`, `weather`, `description`, `wind_speed`, `wind_deg`, `wind_gust`, `clouds`, `visibility`, `pop`, `hour`, `day`, `month`.

## 🎯 Detyrat e Mësimit të Makinës (ML Tasks) të Definuara

Dataset-i i ndërtuar në këtë fazë është projektuar të mbështesë tri detyra kryesore ML në fazat pasuese:

| # | Detyra | Lloji | Target (output) | Tipari hyrës kryesor |
|---|--------|-------|-----------------|---------------------|
| 1 | Parashikimi i temperaturës | Regresion (supervised) | `temperature` (°C, numerik) | humidity, pressure, clouds, wind_speed |
| 2 | Klasifikimi i gjendjes së motit | Klasifikim (supervised) | `weather` (Clear/Clouds/Rain/Snow) | temperature, humidity, clouds, pop |
| 3 | Parashikim kohor sekuencial | Time-series (LSTM, Faza III) | `temperature[t+1]` | sekuencë prej 6 hapash × 5 tipare |

## 🗂️ Tipet e Atributeve

Nga 20 kolonat totale, ndarja strukturore është:

| Lloji | Numri | Atributet |
|-------|-------|-----------|
| **Numerik (të vazhdueshëm)** | 10 | `temperature`, `feels_like`, `temp_min`, `temp_max`, `pressure`, `humidity`, `wind_speed`, `wind_gust`, `visibility`, `pop` |
| **Numerik (diskrete/kohore)** | 5 | `wind_deg`, `clouds`, `hour`, `day`, `month` |
| **Kategorik** | 3 | `city` (27 nivele), `type` (2 nivele), `weather` (4 nivele) |
| **Tekstual** | 1 | `description` (free-text, nuk përdoret në trajnim) |
| **Datetime** | 1 | `datetime` (ISO 8601) |

## 📊 Statistika Përshkruese (atributet numerike)

| Atributi | min | mesatare | std | max |
|----------|-----|----------|-----|-----|
| `temperature` (°C) | −1.99 | 9.47 | 4.52 | 20.00 |
| `feels_like` (°C) | −4.27 | 8.27 | 4.74 | 19.05 |
| `humidity` (%) | 29 | 68.06 | 17.01 | 100 |
| `pressure` (hPa) | 1011 | 1015.98 | 1.91 | 1021 |
| `wind_speed` (m/s) | 0.08 | 2.33 | 1.58 | 10.29 |
| `clouds` (%) | 0 | 60.72 | 36.61 | 100 |
| `visibility` (m) | 0 | 9717 | 1394 | 10000 |
| `pop` (prob.) | 0.00 | 0.15 | 0.31 | 1.00 |

### Shpërndarja e klasave për target-in kategorik `weather`

| Klasa | Numri | Përqindja |
|-------|-------|-----------|
| Clouds | 686 | 61.97% |
| Rain   | 242 | 21.86% |
| Clear  | 170 | 15.36% |
| Snow   | 9   | **0.81%** ⚠️ |

➡️ Vihet re një **mospërputhje klasore** (klasa `Snow` është shumë e nën-përfaqësuar) — problem që do të adresohet në Fazën III me teknika oversampling.

## 🧼 Vlerat e Munguara (Missing Values)

| Kolona | Të munguara | Arsyeja | Trajtimi |
|--------|-------------|---------|----------|
| `pop` | 27 | API nuk e kthen `pop` për rreshtat `type=current` | Plotësim me `0.0` në Fazën II |
| të gjitha të tjerat | **0** | — | — |

**Total NaN në dataset: 27 / (1107 × 20 = 22,140 qeliza) → 0.12%** — cilësi shumë e mirë.

## 💡 Pse u zgjodhën këto atribute?

- **Meteorologjike bazë** (`temperature`, `humidity`, `pressure`, `wind_*`, `clouds`) — janë variablat fizikë standardë për modelim atmosferik.
- **`pop` dhe `visibility`** — tregues të reshjeve / dukshmërisë, të dobishme për klasifikim mot.
- **`hour`, `day`, `month`** — të nxjerrura automatikisht për të kapur **ciklet kohore** (ditore/sezonale).
- **`city`** — mundëson modelime per-qytet ose kategorizim të klimave lokale (do të shfrytëzohet në K-Means të Fazës III).
- **Koordinatat lat/lon** nuk u ruajtën në CSV sepse janë statike për çdo qytet dhe mund të ri-bashkohen nga `weather_data_scraper.py`.

## ✅ Rezultati i Fazës I
Një dataset i plotë, i pastër dhe i strukturuar për Kosovën:
- **1107 instanca × 20 atribute**, vetëm **0.12% NaN** (të trajtueshme),
- **3 detyra ML të definuara qartë** (regresion, klasifikim, time-series),
- statistika përshkruese dhe shpërndarje klasash të dokumentuara,
- gatshëm të ushqejë modelet e Fazës II dhe III pa nevojë për pastrim shtesë madhor.

---

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

## 🤖 Algoritmi i Përzgjedhur

Sipas kërkesës së profesorit (*"Studentët duhet të implementojnë **ndonjërin** nga algoritmet..."*), projekti fokusohet te **një algoritëm i vetëm**: **Random Forest Regressor** — një ansambël i pemëve të vendimit për regresion (supervised learning).

| Faza | Konfigurimi i Random Forest | Statusi |
|------|----------------------------|---------|
| Faza II | Baseline (100 pemë) + Re-trained (300 pemë, leaf≥2, max_features='sqrt') | ✅ E realizuar |
| Faza III | RF i optimizuar (hyperparameter tuning me GridSearchCV + feature engineering) | 🔜 E planifikuar |

---

# 🧪 FAZA II — Analiza dhe Evaluimi (Re-training)

## 🎯 Qëllimi i fazës
Në Fazën II aplikohet **një algoritëm i vetëm — Random Forest Regressor** — për parashikimin e temperaturës (në °C) mbi bazën e tipareve meteorologjike, dhe tregohet procesi iterativ i **re-trajnimit**: një konfigurim *baseline* → një konfigurim i *ri-trajnuar* me hiperparametra të rregulluar dhe më anti-overfitting. Kështu demonstrohet qartë kërkesa e titullit "Re-training".

Skripti përkatës: [`phase2_model_training.py`](phase2_model_training.py)
Metrikat e plota: [`reports/phase2_metrics.txt`](reports/phase2_metrics.txt) · JSON: [`reports/phase2_summary.json`](reports/phase2_summary.json)

## 📸 Visualizimet e Fazës II

Tri vizualizime kryesore përmbledhin rezultatet e kësaj faze:

<table>
  <tr>
    <td align="center"><b>🔥 Harta e Korrelacionit</b></td>
    <td align="center"><b>📈 Parashikuar vs Reale</b></td>
    <td align="center"><b>🌲 Feature Importance</b></td>
  </tr>
  <tr>
    <td><img src="reports/phase2_correlation_heatmap.png" alt="Correlation Heatmap" width="320"/></td>
    <td><img src="reports/phase2_pred_vs_true.png" alt="Predicted vs Actual" width="320"/></td>
    <td><img src="reports/phase2_feature_importance.png" alt="Feature Importance" width="320"/></td>
  </tr>
  <tr>
    <td align="center"><sub>Korrelacionet midis tipareve meteorologjike</sub></td>
    <td align="center"><sub>Baseline vs Re-trained kundrejt diagonales ideale</sub></td>
    <td align="center"><sub>Humidity, hour_sin dhe pressure dominojnë</sub></td>
  </tr>
</table>

## 🌳 Pse Random Forest Regressor?

| Arsyeja | Shpjegimi |
|---------|-----------|
| **Natyra e problemit** | Parashikimi i temperaturës është **regresion i mbikëqyrur** — Random Forest është nga zgjedhjet më robuste për këtë. |
| **Jo-lineariteti** | Marrëdhëniet midis humidity / pressure / clouds dhe temperaturës janë jo-lineare; Random Forest i kap me pemë të shumëfishta të thella. |
| **Robusti ndaj outlier-ëve** | Pemët ndajnë me prag, pra nuk ndikohen nga vlera ekstreme siç do të ndikohej regresioni linear. |
| **Pa nevojë për scaling** | Pemët nuk varen nga shkalla e tipareve — thjeshton *pipeline*-in. |
| **Interpretueshmëri** | Jep **feature importance** të brendshme — e dobishme për të kuptuar se cili tipar është vërtet informues. |
| **Re-training i natyrshëm** | Ka hiperparametra të dukshëm (`n_estimators`, `min_samples_leaf`, `max_features`) që lejojnë iterim të qartë dhe të dokumentueshëm. |

## 🧹 Parapërgatitja e të dhënave

1. **Heqja e rreshtave me mungesa** në `temperature`, `humidity`, `pressure` (tiparet kryesore).
2. **Plotësimi i `pop`** me `0.0` për rreshtat `type=current` (ku API nuk e kthen këtë fushë).
3. **Encoding ciklik i kohës** me `sin/cos` për `hour` dhe `month`:
   - Arsyeja: ora 23:00 dhe 00:00 janë ngjitur në kohë por si numra duken shumë larg. `sin/cos` ruan afërsinë ciklike.
4. **Ndarja 80 / 20** train / test me `random_state=42` për riprodhueshmëri.
5. **Scaler** (`StandardScaler`) ruhet vetëm për përputhshmëri me pipeline-e të ardhshme; Random Forest nuk e ka të domosdoshëm.

### 📊 Madhësia e ndarjeve

| Ndarja | Numri i rreshtave | Përqindja |
|--------|-------------------|-----------|
| **Train** | **885** | 80% |
| **Test**  | **222** | 20% |
| **Total** | 1107 | 100% |

### 🧾 Tiparet hyrëse (11)

`humidity, pressure, wind_speed, wind_deg, clouds, visibility, pop, hour_sin, hour_cos, month_sin, month_cos`

**Target:** `temperature` (°C, numerike).

## 🔎 Analiza Eksploruese (EDA)

### Harta e Korrelacionit

![Correlation Heatmap](reports/phase2_correlation_heatmap.png)

| Tipari | |corr| me `temperature` |
|--------|--------------------------|
| `humidity`   | **0.758** (më i forti) |
| `pressure`   | 0.406 |
| `wind_speed` | 0.371 |
| `clouds`     | 0.199 |
| `visibility` | 0.173 |
| `pop`        | 0.147 |

➡️ Lagështia është prediktori më i fuqishëm — konstatim fizikisht i pritshëm.

## 🔄 Re-training — dy iterime eksplicite

| Parametri | Iterimi 1 (baseline) | Iterimi 2 (re-trained) |
|-----------|----------------------|------------------------|
| `n_estimators` | 100 | **300** |
| `min_samples_leaf` | 1 | **2** |
| `max_features` | `None` (të gjitha) | **`sqrt`** |
| `max_depth` | `None` (pa limit) | `None` |
| `random_state` | 42 | 42 |

### 💡 Justifikimi i ndryshimeve
- **300 pemë në vend të 100** → reduktim i variancës përmes mesatarizimit të më shumë pemëve (bagging më i fortë).
- **`min_samples_leaf=2`** → parandalon pemët të mbajnë gjethe me vetëm 1 mostër (tipari klasik i overfitting-ut).
- **`max_features='sqrt'`** → çdo pemë sheh vetëm ~√11 ≈ 3 tipare për split → rrit *decorrelation* midis pemëve → ansambël më i përgjithësuar.
- Arsyeja kryesore: **reduktim i overfitting-ut**, jo maksimizimi i R²-së në test-set-in e vetëm.

## 📊 Rezultatet

| Modeli | MAE ↓ | RMSE ↓ | R² (train) | R² (test) | R² (5-fold CV) |
|--------|-------|--------|------------|-----------|----------------|
| **Baseline RF**   | **1.029 °C** | **1.519 °C** | 0.9828 | 0.8831 | 0.8677 |
| **Re-trained RF** | 1.132 °C | 1.546 °C | **0.9561** | 0.8789 | **0.8742** |

### 📈 Parashikuar vs Reale

![Predicted vs Actual Temperature](reports/phase2_pred_vs_true.png)

Pikat janë të grumbulluara përgjatë diagonales ideale (vija e ndërprerë) → modeli ka përputhje të fortë me vlerat reale të temperaturës. Devijime më të mëdha shfaqen vetëm në skajet (temperatura shumë të larta ose shumë të ulëta), të cilat janë nën-përfaqësuara në dataset.

### 🔥 Feature Importance (modeli i ri-trajnuar)

![Feature Importance](reports/phase2_feature_importance.png)

| Tipari | Importance |
|--------|-----------|
| `humidity`   | **0.342** |
| `hour_sin`   | 0.153 |
| `pressure`   | 0.132 |
| `wind_speed` | 0.119 |
| `clouds`     | 0.095 |
| `hour_cos`   | 0.083 |
| `wind_deg`   | 0.053 |
| `pop`        | 0.018 |
| `visibility` | 0.005 |
| `month_sin`  | ~0.000 |
| `month_cos`  | ~0.000 |

➡️ Lagështia, ora e ditës (sin/cos) dhe presioni janë tre tiparet më informuese. `month_*` janë zero sepse dataset-i mbulon vetëm 5 ditë → nuk ka variancë të muajit.

## 💬 Diskutim i rezultateve

1. **R² (test) ≈ 0.88** — modeli shpjegon ~88% të variancës së temperaturës, me **MAE ≈ 1°C** — cilësi e fortë për një dataset të vogël (1107 rreshta, 5 ditë).

2. **Baseline vs Re-trained — një mësim i rëndësishëm:**
   - Baseline ka **R² (train) = 0.9828 vs R² (test) = 0.8831** → hendek prej **~10 pikësh** → shenjë e qartë e **overfitting-ut**.
   - Re-trained ka **R² (train) = 0.9561 vs R² (test) = 0.8789** → hendek prej **~7.7 pikësh** → overfitting më i reduktuar.
   - **5-fold CV R² u rrit** nga 0.8677 → **0.8742** → modeli i ri-trajnuar **përgjithëson më mirë**.
   - Ky është rasti tipik ku "train/test i vetëm" mund të ngatërrojë: test-set-i fiks ndonjëherë nuk zbulon overfitting-un që zbulon CV. Për këtë arsye u raportua të dyja.

3. **Feature importance konfirmon fizikën** — lagështia dominon, pasuar nga ora e ditës dhe presioni, pra modeli ka mësuar marrëdhënie kuptimplote, jo artefakte.

4. **Kufizimi kryesor** — dataset-i mbulon vetëm 5 ditë, pa variancë të muajit/stinës. Për këtë arsye `month_sin/cos` kanë rëndësi zero. Në Fazën III do të trajtohet me grumbullim më të gjatë të dataset-it dhe tuning të avancuar.

## 🧾 Konkluzione të Fazës II
1. U implementua **një algoritëm i vetëm — Random Forest Regressor** — sipas kërkesës së profesorit ("ndonjërin nga algoritmet").
2. U tregua procesi i **re-trajnimit** me dy iterime eksplicite dhe justifikim për secilin ndryshim të hiperparametrave.
3. U arrit **R² (test) = 0.88** dhe **MAE = 1.03 °C** në baseline; re-trained përmirësoi përgjithësimin (CV R² nga 0.87 → 0.87+).
4. U dokumentua ndarja train/test (885/222), madhësia e dataset-it, dhe u bë krahasim train-vs-test për të treguar overfitting-un.
5. Feature importance konfirmoi interpretim fizik kuptimplotë.
6. Të gjitha modelet dhe raportet janë serializuar në [`models/`](models/) dhe [`reports/`](reports/), gati për krahasim në Fazën III.
