# HDRF v3 — prototyp i przygotowanie eksperymentu E0

HDRF (*Hyperparameter-Diverse Random Forest*) to własna implementacja lasu losowego, w którym każde drzewo CART może mieć niezależnie wylosowane hiperparametry. Projekt służy do zbadania, czy heterogeniczność drzew wpływa na jakość predykcji, różnorodność modeli i koszt obliczeniowy.

**Status:** prototyp działa dla klasyfikacji i regresji; przygotowano pobieranie danych OpenML, ich odczyt oraz kontrolną walidację krzyżową. **Rozkłady losowania nie zostały jeszcze skalibrowane w E0.** Dotychczasowe wyniki to diagnostyka implementacji, nie końcowe rezultaty pracy.

## Struktura projektu

| Plik / katalog | Rola |
| --- | --- |
| `hdrf/sampling.py` | `SamplingConfig` i rozkłady losowania hiperparametrów drzew. |
| `hdrf/forest.py` | Budowa lasu: bootstrap przez wagi, szacowanie D_full, równoległe uczenie drzew CART, agregacja predykcji. |
| `hdrf/center.py` | Empiryczny środek rozkładów (model RF ze środka). |
| `fetch_e0.py`, `inspect_e0.py`, `data_loader.py` | Pobieranie, kontrola i ładowanie zadań OpenML dla E0 wraz z oryginalnymi foldami. |
| `real_data_smoke_test.py` | Szybkie sprawdzenie uczenia na jednym foldzie. |
| `pilot_e0.py` | Pilotaż rozkładów na credit i Ailerons (10 foldów). Wyniki w `results/pilot_e0/`. |
| `tests/`, `test_data_loader.py` | Testy jednostkowe. |

## Instalacja i uruchomienie

Komendy wykonujemy w **PowerShell**, z katalogu głównego repozytorium. Wymagany Python 3.10 (scikit-learn 1.7.2 wymaga co najmniej 3.10).

```powershell
# Jednorazowe utworzenie środowiska, jeśli jeszcze go nie ma:
py -3.10 -m venv .venv

# Instalacja zależności do środowiska projektu:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Uruchamianie skryptów bez potrzeby aktywacji .venv:
.\.venv\Scripts\python.exe -m pytest -q
```

Jeżeli środowisko `.venv` jest już aktywne (w terminalu widać `(.venv)`), zamiast pełnej ścieżki do interpretera można używać po prostu `python`. Jeśli na komputerze nie działa polecenie `py`, utwórz środowisko komendą `python -m venv .venv`, pod warunkiem że `python` wskazuje na zainstalowaną wersję 3.10.

## Jak działa HDRF

Każde drzewo dostaje własną konfigurację hiperparametrów. Parametr `randomize` wybiera, które grupy losujemy (domyślnie wszystkie). Pozostałe mają wartości domyślne scikit-learn albo wartości z `fixed_params`.

| Grupa | Losowanie (stałe w `SamplingConfig`, robocze, do kalibracji w E0) |
| --- | --- |
| `complexity` | Tryb per drzewo. **depth** (prawd. `depth_mode_prob`): `max_depth` jednostajne na {⌈c·log₂n⌉, …, D_full}, gdzie D_full oznacza pełne drzewo; liść = 1. **leaf**: `min_samples_leaf` log-jednostajne na {1, …, `leaf_max`}; głębokość bez limitu. D_full to mediana głębokości kilku pełnych drzew pilotażowych. `min_samples_split` = 2 zawsze. |
| `criterion` | Jednostajnie z {gini, entropy} / {squared_error, friedman_mse}. |
| `max_features` | Mieszanka: z prawd. `features_default_prob` K = K₀, w przeciwnym razie ln K ~ U[ln(K₀/a), ln(K₀·b)], obcięte do [1, p]. K₀ = √p (klasyfikacja), p albo p/3 (regresja, `regression_k0`). Przy a = b mediana = K₀. |
| `max_samples` | BR (bootstrap ratio) jednostajny na `bootstrap_bounds`; bootstrap ze zwracaniem o rozmiarze BR·n (BR może być > 1). |

Bootstrap realizujemy jak scikit-learn: wagami obserwacji (liczba wylosowań), bez kopiowania wierszy. HDRF z `randomize=()` daje wtedy las równoważny `RandomForest` ze scikit-learn.

**RF ze środka** (`empirical_center`): K i BR to mediany z samplera (Monte Carlo), `criterion` domyślne. Złożoność: mediana liczby liści M drzew HDRF uczonych na tych samych danych, a potem jedna wartość `min_samples_leaf` (lub `max_depth`), dla której drzewa mają średnio liczbę liści najbliższą M.

```python
from hdrf import HDRFClassifier, SamplingConfig, empirical_center

hdrf = HDRFClassifier(n_estimators=500, random_state=42, n_jobs=-1).fit(X_train, y_train)
only_features = HDRFClassifier(500, randomize=("max_features",), random_state=42, n_jobs=-1)
center, info = empirical_center(X_train, y_train, "classification", reference=hdrf, limit="leaf")
rf_center = HDRFClassifier(500, randomize=(), fixed_params=center, random_state=42, n_jobs=-1)
```

Powtarzalność: osobne strumienie RNG dla bootstrapu, ziaren drzew, D_full i każdej grupy hiperparametrów, więc włączenie losowania jednej grupy nie zmienia losowań pozostałych. Wynik nie zależy od `n_jobs`. Po treningu dostępne są `tree_params_`, `bootstrap_info_`, `tree_seeds_`, `d_full_`.

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
.\.venv\Scripts\python.exe pilot_e0.py
```

Za pobieranie danych i oryginalnych podziałów z OpenML odpowiada `fetch_e0.py`; uruchom go zgodnie z listą zadań i argumentami zdefiniowanymi w skrypcie. **Nie pobieraj od razu wszystkich zbiorów, jeśli chcesz najpierw sprawdzić przetwarzanie dwóch obecnych.**

`load_e0_task` z `data_loader.py` odczytuje **już pobrany zbiór i przygotowane podziały**. To funkcja projektu, a nie biblioteka zewnętrzna ani algorytm HDRF. Jej zadaniem jest udostępnienie tych samych obserwacji i foldów wszystkim porównywanym modelom.

### Pilotaż rozkładów (`pilot_e0.py`)

Na credit i Ailerons, 10 foldów OpenML, B = 500: RF domyślny, losowanie każdej grupy osobno, pełny HDRF, tryby złożoności osobno, RF ze środka (leaf i depth), w regresji także K₀ = p/3. To nie jest E0: dwa zbiory i robocze stałe.

## Co pozostaje do zrobienia

1. Zweryfikować wczytywanie i preprocessing pozostałych 14 zbiorów E0, w tym przypadki z brakami danych i cechami kategorycznymi.
2. Zaimplementować właściwy protokół **E0**: badanie rozkładów pojedynczych hiperparametrów, porównania wariantów i wybór docelowych przestrzeni losowania na zbiorach rozwojowych.
3. Po zakończeniu E0 **zamrozić konfigurację HDRF** przed walidacją na nowych zbiorach (E1–E3).
4. Uzupełnić analizę statystyczną, różnorodność drzew i koszt obliczeniowy zgodnie z protokołem pracy.
5. Rozważyć rozszerzenia techniczne: OOB score (potrzebny w E2* i E3), `sample_weight`, obsługę macierzy sparse.

**Najważniejsza zasada eksperymentalna:** wyniki na zbiorach przeznaczonych do końcowej oceny nie mogą służyć do dobierania rozkładów HDRF. Kalibrację przeprowadzamy tylko w ramach E0.
