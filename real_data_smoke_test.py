
from data_loader import load_e0_task
from hdrf import HDRFClassifier
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from hdrf import HDRFRegressor


# 1. Wczytujemy rzeczywisty zbior OpenML
task = load_e0_task(361055)

# 2. Pobieramy pierwszy oryginalny fold
X_train, X_test, y_train, y_test = task.get_fold(0)

print("Trening:", X_train.shape)
print("Test:", X_test.shape)

# 3. Tworzymy trzy modele
models = {
    "RF default": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    ),

    "Homogeneous center": HDRFClassifier(
        n_estimators=100,
        randomize=(),
        random_state=42
    ),

    "HDRF full": HDRFClassifier(
        n_estimators=100,
        random_state=42
    )
}

# 4. Trenujemy i oceniamy
for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"{name}: accuracy={accuracy:.4f}")

print("\n=== REGRESJA: Ailerons ===")

task = load_e0_task(361077)

X_train, X_test, y_train, y_test = task.get_fold(0)

print("Trening:", X_train.shape)
print("Test:", X_test.shape)

models = {
    "RF default": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    ),

    "Homogeneous center": HDRFRegressor(
        n_estimators=100,
        randomize=(),
        random_state=42
    ),

    "HDRF full": HDRFRegressor(
        n_estimators=100,
        random_state=42
    )
}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    print(
        f"{name}: RMSE={rmse:.8f}, R2={r2:.4f}"
    )
