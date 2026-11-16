# Pilotaże etapu 1 (wybór sposobów losowania), archiwum

Skrypty, którymi wybraliśmy rozkłady (wariant A_p, teraz domyślny `SamplingConfig`).
Uruchamiały się z katalogu głównego repozytorium, ze starym API `SamplingConfig`
(K przez `features_default_prob`, `features_down`, `features_up`, `classification_k0`,
`regression_k0`). **Z obecnym kodem się nie uruchomią**. Kod, który je liczył, jest
w commicie `b11ed96`, a `pilot_k_warunkowe.py` (opcja `classification_k0="by_splitter"`)
w stanie roboczym tuż po nim. Wyniki (CSV per zbiór, `summary.csv`, `log.txt`) są lokalnie
w `results/<nazwa>/` (poza gitem).

Kolejność i wnioski:

| skrypt | zbiory | co sprawdza | wniosek |
|---|---|---|---|
| `pilot_e0.py` | credit, Ailerons | grupy osobno, tryby złożoności, środek leaf/depth, K₀ = p/3 | punkt wyjścia v3 |
| `pilot_szerokosc.py` | credit, Ailerons | szersze rozkłady | szerzej ≠ lepiej |
| `pilot_dev16.py` | 16 dev | RF, HDRF bazowy, środek (leaf) | regresja +0.5%, klasyfikacja ≈ RF |
| `pilot_pelniejsze.py` | 16 dev | depth `log_full`, leaf_max = 5, środek przez `max_leaf_nodes` | ≈ bazowy |
| `pilot_splitter.py` | 16 dev | wariant A (splitter π = 0.5), środek A, Extra Trees | regresja +1.2% (p = 0.04); klasyfikacja traci przy K₀ = √p |
| `pilot_k_p.py` | 16 dev | A_p (K₀ = p w klasyfikacji), π = 1 | **A_p wybrany**: log-loss +1.55% vs RF, π = 0.5 > π = 1 |
| `pilot_k_warunkowe.py` | 8 klas. | K₀ zależne od splittera (random: p, best: √p) | tańszy, ale gorszy od A_p (log-loss −0.2%, 4/8) |
