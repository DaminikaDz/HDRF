# HDRF v3 — prototyp i przygotowanie eksperymentu E0

HDRF (*Hyperparameter-Diverse Random Forest*) to własna implementacja lasu losowego, w którym każde drzewo CART może mieć niezależnie wylosowane hiperparametry. Projekt służy do zbadania, czy heterogeniczność drzew wpływa na jakość predykcji, różnorodność modeli i koszt obliczeniowy.

**Status:** prototyp działa dla klasyfikacji i regresji; przygotowano pobieranie danych OpenML, ich odczyt oraz kontrolną walidację krzyżową. **Rozkłady losowania nie zostały jeszcze skalibrowane w E0.** Dotychczasowe wyniki to diagnostyka implementacji, nie końcowe rezultaty pracy.

## Struktura projektu

| Plik / katalog | Rola |
| --- | --- |
| `hdrf/sampling.py` | `SamplingConfig`, parametry domyślne i centralne oraz losowanie hiperparametrów drzew. |
| `hdrf/forest.py` | Budowa lasu: bootstrap, uczenie drzew CART, agregacja predykcji i rejestrowanie użytych konfiguracji. |
| `hdrf/__init__.py` | Publiczne importy, m.in. `HDRFClassifier`, `HDRFRegressor`, `SamplingConfig`. |
| `demo.py` | Demonstracja na danych syntetycznych: klasyfikacja i regresja, wstępne porównania modeli. |
| `controlled_probe.py` | Kontrolowana diagnostyka `max_features` przy wspólnych ziarnach CART i indeksach bootstrap. |
| `check_e0_baseline.py` | Kontrola trybów `default` i `center` przy losowaniu jednego hiperparametru. |
| `fetch_e0.py` | Pobieranie danych i podziałów walidacyjnych z zadań OpenML przewidzianych do E0. |
| `inspect_e0.py` | Kontrola metadanych zbiorów i integralności podziałów. |
| `data_loader.py` | Ładowanie lokalnie pobranych danych E0 oraz foldów (`load_e0_task`). |
| `real_data_smoke_test.py` | Szybkie sprawdzenie uczenia na jednym foldzie rzeczywistych danych. |
| `real_data_cv.py` | Kontrolne porównanie modeli we wszystkich 10 foldach dwóch pierwszych zbiorów. |
| `tests/`, `test_data_loader.py` | Testy jednostkowe i testy ładowania danych. |
| `results/cv_smoke/` | Wyniki kontrolnej CV, po uruchomieniu `real_data_cv.py`. |

## Instalacja i uruchomienie

Komendy wykonujemy w **PowerShell**, z katalogu `D:\PWW\inzynierka\implementacja`.

```powershell
# Jednorazowe utworzenie środowiska, jeśli jeszcze go nie ma:
py -3.12 -m venv .venv

# Instalacja zależności do środowiska projektu:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Uruchamianie skryptów bez potrzeby aktywacji .venv:
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe demo.py
.\.venv\Scripts\python.exe controlled_probe.py
.\.venv\Scripts\python.exe check_e0_baseline.py
```

Jeżeli środowisko `.venv` jest już aktywne (w terminalu widać `(.venv)`), zamiast pełnej ścieżki do interpretera można używać po prostu `python`. Jeśli na komputerze nie działa polecenie `py`, utwórz środowisko komendą `python -m venv .venv`, pod warunkiem że `python` wskazuje na zainstalowaną wersję 3.12.

## Jak działa HDRF

Każde drzewo otrzymuje własną konfigurację hiperparametrów. Parametr `randomize` określa, **które** z nich mają być losowane. Pozostałe przyjmują wartości stałe zależne od trybu `fixed_params_mode`.

| Hiperparametr | Co kontroluje |
| --- | --- |
| `max_depth` | Maksymalna głębokość drzewa (`None` oznacza brak narzuconego limitu). |
| `min_samples_split` | Minimalna liczba obserwacji konieczna do podziału węzła. |
| `min_samples_leaf` | Minimalna liczba obserwacji w liściu. |
| `criterion` | Funkcja oceny jakości podziałów. |
| `max_features` | Liczba / udział cech rozpatrywanych przy podziale. |
| `max_samples` | Wewnętrzna nazwa stosowanego u nas **bootstrap ratio**. |

### Dwa tryby wartości niezmienianych (`fixed_params_mode`)

- **`"default"`** — parametry, których nie losujemy, mają wartości odpowiadające domyślnym ustawieniom `RandomForest` w scikit-learn. Przykładowo: `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `max_samples=1.0`. Ten tryb jest potrzebny do **E0**, gdy badamy wpływ losowania pojedynczego hiperparametru przy domyślnych wartościach pozostałych.
- **`"center"`** — parametry, których nie losujemy, przyjmują robocze punkty centralne z `SamplingConfig` (np. `min_samples_split=6`, `min_samples_leaf=3`, `max_samples=0.85`). Jest przydatny do kontrolowanych eksperymentów i przyszłych ablacji, ale **te punkty nie są jeszcze wynikami kalibracji E0**.

**Ważne:** `fixed_params_mode="default"` nie oznacza, że nasz las jest implementacyjnie identyczny z `sklearn.ensemble.RandomForestClassifier`. Wewnętrzny mechanizm losowania i trenowania może się różnić; ustawienia hiperparametrów i zgodność wyników to dwie osobne sprawy.

### Przykłady użycia

```python
from dataclasses import replace
from hdrf import HDRFClassifier, HDRFRegressor, SamplingConfig

base = SamplingConfig()

# 1. Las homogeniczny o roboczych parametrach centralnych.
homogeneous_center = HDRFClassifier(
    n_estimators=100,
    randomize=(),
    random_state=42,
    sampling_config=base,
)

# 2. Przykład E0: losujemy TYLKO max_features;
#    reszta parametrów ma domyślne wartości RF.
only_features_e0 = HDRFClassifier(
    n_estimators=100,
    randomize=("max_features",),
    random_state=42,
    sampling_config=replace(base, fixed_params_mode="default"),
)

# 3. Kontrolne porównanie z punktem centralnym:
#    tę samą liczbę drzew i seed zachowujemy w obu wariantach.
fixed_features = HDRFClassifier(
    n_estimators=100,
    randomize=(),
    random_state=42,
    sampling_config=replace(base, center_classification_features=0.75),
)

random_features = HDRFClassifier(
    n_estimators=100,
    randomize=("max_features",),
    random_state=42,
    sampling_config=base,
)

# 4. Kandydacka przestrzeń losowania bootstrap ratio nawet powyżej 1.
#    To przykład konfiguracji, a nie przyjęty wynik kalibracji E0.
bootstrap2 = HDRFClassifier(
    n_estimators=100,
    random_state=42,
    sampling_config=replace(base, bootstrap_bounds=(0.7, 2.0)),
)

# Analogiczne warianty są dostępne dla HDRFRegressor.
```

W przykładach utworzono modele, ale ich nie wytrenowano. Do treningu używamy `model.fit(X_train, y_train)`, a do predykcji `model.predict(X_test)`.

### Bootstrap ratio może być większe od 1

W naszym HDRF `max_samples=1.0` oznacza losowanie `n` indeksów **ze zwracaniem** z `n` dostępnych obserwacji, a nie gwarancję użycia wszystkich unikalnych wierszy. Wartość `1.5` oznacza `1.5n` losowań ze zwracaniem i może zwiększyć oczekiwaną liczbę unikalnych obserwacji. Nie oznacza jednak pobrania `150%` unikalnych danych. Taki sposób interpretacji `max_samples` różni się od ograniczeń tego parametru w scikit-learn.

### Powtarzalność i kontrolowane porównania

- `random_state=42` daje odtwarzalny eksperyment. Stosujemy osobne strumienie RNG dla bootstrapu, inicjalizacji drzew oraz losowania hiperparametrów.
- W porównaniach wariantów implementacji HDRF przy wspólnym seedzie indeksy bootstrap i ziarna CART pozostają zgodne, **o ile nie zmieniamy losowanego rozmiaru bootstrapu**.
- Po treningu można analizować m.in. `tree_params_`, `bootstrap_info_` i `tree_seeds_`.
- Gdy `random_state=None`, do ponownego uruchomienia można wykorzystać zapisane `seed_entropy_`.

## OpenML: przygotowanie danych do E0

Plan E0 zakłada **16 zbiorów rozwojowych: 8 klasyfikacyjnych i 8 regresyjnych**. Na razie przygotowano i sprawdzono dwa pierwsze:

| Zbiór | OpenML task ID | Zadanie | Obserwacje | Cechy |
| --- | ---: | --- | ---: | ---: |
| `credit` | `361055` | Klasyfikacja | 16 714 | 10 |
| `Ailerons` | `361077` | Regresja | 13 750 | 33 |

Oba sprawdzone zbiory mają wyłącznie cechy numeryczne i nie mają braków danych. **Nie oznacza to**, że pozostałe zbiory E0 również nie będą wymagać przygotowania lub kodowania cech kategorycznych.

Skrypty danych:

```powershell
# Uruchom po pobraniu danych wymaganych przez dany skrypt:
.\.venv\Scripts\python.exe inspect_e0.py
.\.venv\Scripts\python.exe -m pytest -q test_data_loader.py
.\.venv\Scripts\python.exe real_data_smoke_test.py
.\.venv\Scripts\python.exe real_data_cv.py
```

Za pobieranie danych i oryginalnych podziałów z OpenML odpowiada `fetch_e0.py`; uruchom go zgodnie z listą zadań i argumentami zdefiniowanymi w skrypcie. **Nie pobieraj od razu wszystkich zbiorów, jeśli chcesz najpierw sprawdzić przetwarzanie dwóch obecnych.**

`load_e0_task` z `data_loader.py` odczytuje **już pobrany zbiór i przygotowane podziały**. To funkcja projektu, a nie biblioteka zewnętrzna ani algorytm HDRF. Jej zadaniem jest udostępnienie tych samych obserwacji i foldów wszystkim porównywanym modelom.

### Kontrolna walidacja krzyżowa (`real_data_cv.py`)

Aktualny skrypt przeprowadza 10-fold CV dla dwóch zbiorów (`credit`, `Ailerons`) z `B=100` drzewami i porównuje:

1. `RF default` — bazowy `RandomForest` ze scikit-learn;
2. `Homogeneous center` — nasz HDRF z `randomize=()` i roboczymi parametrami centralnymi;
3. `HDRF full` — nasz HDRF z losowaniem wszystkich włączonych hiperparametrów.

**Uwaga metodologiczna:** ta kontrolna CV w obecnej postaci **nie jest jeszcze E0**. W szczególności nie realizuje systematycznego testowania pojedynczych hiperparametrów w trybie `fixed_params_mode="default"`, nie kalibruje rozkładów i korzysta z `B=100`, a nie z docelowego ustawienia E0.

Po zakończeniu skrypt zapisuje pliki:

```text
results/cv_smoke/361055_folds.csv
results/cv_smoke/361055_summary.csv
results/cv_smoke/361077_folds.csv
results/cv_smoke/361077_summary.csv
```

Wyniki dotychczasowej diagnostyki (średnia z 10 foldów):

| Zbiór, miara | RF default | Homogeneous center | HDRF full |
| --- | ---: | ---: | ---: |
| `credit`, accuracy ↑ | 0.77085 | 0.77594 | 0.77498 |
| `Ailerons`, RMSE ↓ | 0.00016169 | 0.00016200 | 0.00016189 |

Różnice są niewielkie; tych dwóch porównań nie traktujemy jako dowodu przewagi którejkolwiek metody.

## Co pozostaje do zrobienia

1. Zweryfikować wczytywanie i preprocessing pozostałych 14 zbiorów E0, w tym przypadki z brakami danych i cechami kategorycznymi.
2. Zaimplementować właściwy protokół **E0**: badanie rozkładów pojedynczych hiperparametrów, porównania wariantów i wybór docelowych przestrzeni losowania na zbiorach rozwojowych.
3. Po zakończeniu E0 **zamrozić konfigurację HDRF** przed walidacją na nowych zbiorach (E1–E3).
4. Uzupełnić analizę statystyczną, różnorodność drzew i koszt obliczeniowy zgodnie z protokołem pracy.
5. Rozważyć rozszerzenia techniczne: `n_jobs`, OOB score, `sample_weight`, obsługę macierzy sparse.

**Najważniejsza zasada eksperymentalna:** wyniki na zbiorach przeznaczonych do końcowej oceny nie mogą służyć do dobierania rozkładów HDRF. Kalibrację przeprowadzamy tylko w ramach E0.
