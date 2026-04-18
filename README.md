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

## 🤖 Modeli i Machine Learning

| Modeli | Statusi     | Qëllimi                                        |
|--------|-------------|------------------------------------------------|
| LSTM   | ✅ Trajnuar | Parashikimi i temperaturës (sekuenca kohore)  |

---

# 🧠 FAZA II — Model Training (LSTM)

Skripta: [`phase2_model_training.py`](phase2_model_training.py)
Modeli i ruajtur: [`models/lstm_model.pt`](models/lstm_model.pt)

## 🎯 Pse u zgjodh LSTM?

Tema e projektit është **Parashikim i Motit (Weather Forecasting)** — kjo është nga natyra e saj një **detyrë e serive kohore (time-series)**. Vlera e temperaturës në momentin `t` është fort e lidhur me vlerat në `t-1, t-2, t-3, ...` (autokorrelacion). Për këtë arsye, modelet që trajtojnë çdo rresht në mënyrë të pavarur (Linear Regression, Random Forest, XGBoost) janë në thelb të papërshtatshme — ato nuk shohin "rrjedhën" e motit.

**LSTM (Long Short-Term Memory)** është një rrjet rekurent neuronal i projektuar pikërisht për këtë:

- **Gates (forget / input / output)** lejojnë rrjetin të zgjedhë çfarë informacioni nga e kaluara të mbajë dhe çfarë të harrojë.
- **Memorja afatshkurtër + afatgjatë** kap si ndryshime të shpejta (orë në orë) ashtu edhe trende (ditore/sezonale).
- **Konsumon një dritare të plotë historie** (18 orë) për të bërë çdo parashikim, jo vetëm një snapshot.
- **Standard i industrisë** për parashikim moti, çmimesh aksionesh, konsum energjie etj.

Krahasuar me alternativat:

| Alternativa       | Pse JO për këtë projekt |
|-------------------|-------------------------|
| Linear Regression | Lineare; nuk kap ndërveprime kohore. |
| Random Forest     | Klasifikim, jo parashikim numerik kohor. |
| XGBoost           | I fortë në tabela, por trajton rreshtat si i.i.d. — humb sekuencën. |
| K-Means           | I pambikëqyrur (clustering), jo parashikim. |

---

## ⚙️ Parapërgatitja e të Dhënave

| Hapi                  | Detajet |
|-----------------------|---------|
| Ngarkim & pastrim     | `dropna(temperature)`, sortim sipas `(city, datetime)` |
| Filtrimi              | Mbahen vetëm rreshtat `type == "forecast"` (intervale të rregullta 3h) |
| Shkallëzim            | `StandardScaler` mbi 5 veçoritë e hyrjes (zero-mean, unit-variance) |
| Ndërtim sekuencash    | Dritare rrëshqitëse 6 hapa **brenda secilit qytet** (nuk kalojnë kufirin) |
| Ndarja Train/Test     | **Temporale 80/20** — pa shuffle, për të shmangur leakage nga e ardhmja |

**Pse `StandardScaler`?** LSTM-të janë të ndjeshme ndaj shkallës së hyrjes; gradientët bëhen të paqëndrueshëm kur veçoritë kanë diapazone shumë të ndryshme (p.sh. `pressure ≈ 1015` vs `wind_speed ≈ 5`). Standardizimi e zgjidh këtë.

**Pse sekuenca PËR QYTET?** Nëse do të ndërtonim sekuenca duke përzier qytetet, modeli do të mësonte "kërcime" jorealiste (p.sh. nga Pristina në Prizren). Duke i ndarë, çdo sekuencë reflekton një rrjedhë moti të vërtetë në një vend të vetëm.

**Pse ndarje temporale?** Random shuffle do të lejonte modelin të "shihte" të ardhmen gjatë trajnimit. Ndarja temporale (e para 80% si train, e fundit 20% si test) imiton skenarin real: trajnoj me të kaluarën, parashikoj të ardhmen.

---

## 🏗️ Arkitektura e Modelit

```
Input (batch, 6, 5)            ← 6 hapa kohorë × 5 veçori
   │
   ▼
LSTM(hidden=64, batch_first)   ← shtresa rekurente
   │
   ▼
output[:, -1, :]               ← merr vetëm hap-in e fundit (many-to-one)
   │
   ▼
Dropout(0.2)                   ← regularizim, parandalon overfitting
   │
   ▼
Dense(32) + ReLU               ← projeksion joLinear
   │
   ▼
Dense(1)                       ← parashikim i temperaturës (skalë e standardizuar)
```

| Hiperparametri    | Vlera        | Justifikim |
|-------------------|--------------|------------|
| SEQ_LEN           | 6            | 6 × 3h = **18h histori** — kapin ciklin ditor pa rritur shumë trajnimin |
| LSTM hidden       | 64           | Madhësi e moderuar; mjaft për dataset të vogël (~734 sekuenca trajnimi) |
| Dropout           | 0.2          | Mbron nga overfitting në dataset të vogël |
| Optimizer         | Adam (lr=1e-3) | Konvergon shpejt, kërkon pak akordim |
| Loss              | MSE          | Standard për regresion; penalizon gabime të mëdha |
| Epochs (max)      | 60           | Me early-stopping (patience=10) — ndalon kur val_loss nuk përmirësohet |
| Batch size        | 32           | Balancë midis qëndrueshmërisë së gradientit dhe shpejtësisë |

---

## 📐 Rezultatet

**Veçoritë e hyrjes (5):** `temperature`, `humidity`, `pressure`, `wind_speed`, `clouds`
**Target:** temperatura e hapit pasardhës (3 orë në të ardhmen)

**Madhësitë e dataset-it:**
- Sekuenca totale: **918** (nga 27 qytete × ~34 dritare/qytet)
- Train: **734**  |  Test: **184**

| Metrika | Vlera     | Interpretim |
|---------|-----------|-------------|
| **MAE**  | **0.500 °C** | Gabimi absolut mesatar — mesatarisht modeli gabon vetëm me gjysmë gradi Celsius |
| **RMSE** | **0.756 °C** | Gabimi më i ndjeshëm ndaj outliers; afër MAE → pak parashikime me gabime të mëdha |
| **R²**   | **0.967**    | Modeli shpjegon **96.7%** të variancës së temperaturës |

### Interpretim i thelluar

- **R² = 0.967** është një rezultat shumë i fortë për një model të trajnuar nga e para në një dataset të vogël. Tregon që dinamika e temperaturës në Kosovë është **shumë e parashikueshme në horizont 3-orësh**, dhe që historia 18-orëshe është informacion i mjaftueshëm.

- **MAE = 0.5°C** është nën rezolucionin praktik të dobishëm për përdoruesit fundorë (njerëzit nuk dallojnë 0.5°C); pra modeli është *gati* për përdorim real në këtë horizont.

- **RMSE ≈ 1.5 × MAE** — kjo është një raport "i shëndetshëm". Nëse RMSE do të ishte shumë më i madh se MAE, do të nënkuptonte se kemi pak parashikime *katastrofike*. Këtu nuk e kemi atë problem.

- **Early stopping** u aktivizua brenda <60 epokave, gjë që tregon se modeli nuk po overfit.

### Kufizimet (që do të adresohen në Fazën III)

1. **Dataset i vogël** (1107 rreshta, ~5 ditë mbulim) — për trajnim sezonal të vërtetë do të nevojiten muaj/vite të dhëna.
2. **Horizont fiks 3h** — për parashikim 24h ose 7 ditë do të duhet ose model multi-step ose strategji *recursive forecasting*.
3. **Pa hyperparameter tuning** — Faza III do të aplikojë GridSearch / Optuna mbi `hidden`, `SEQ_LEN`, `dropout`, `lr`.
4. **Një target i vetëm** — mund të zgjerohet në *multi-output* (temperaturë + lagështi + reshje njëkohësisht).

---

## 📦 Artefaktet e Ruajtura

Pas trajnimit, gjenerohen në [`models/`](models/):

| Skedari            | Përmbajtja                                                   |
|--------------------|--------------------------------------------------------------|
| `lstm_model.pt`    | PyTorch `state_dict` — peshat e rrjetit të trajnuar         |
| `scaler_lstm.pkl`  | `StandardScaler` për të standardizuar/destandartizuar të dhënat |
| `metrics.txt`      | Log i plotë i trajnimit (forma sekuencash, epokat, metrikat) |

## 🚀 Si të Riprodhohen Rezultatet

```bash
pip install -r requirements.txt
python weather_data_scraper.py     # mbledh datasetin (nëse mungon CSV)
python phase2_model_training.py    # trajnon LSTM-in dhe ruan modelin
```

