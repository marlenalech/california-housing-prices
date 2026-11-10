import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

pd.set_option('display.width', None)

df=pd.read_csv('housing (1).csv', sep=r'\s+', header=None)

pd.set_option('display.width', 1000)

nazwy_kolumn = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

df=pd.read_csv('housing (1).csv', sep=r'\s+', header=None, names=nazwy_kolumn)

print("--1. PIERWSZE 5 WIERSZY --")
print(df.head())

print("\n--- Informacje o kolmunach typach---")
print(df.info())

print("\n--- 3 podsumowanie statystyczne describe")
print(df.describe().T)

print("\n--- braki danych ---")
print(df.isnull().sum())

#macierz korelacji
macierz_korelacji = df.corr()
print("\n--- Wykres 1: Mapa ciepła ---")
plt.figure(figsize=(10,8))
sns.heatmap(
    macierz_korelacji,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1, vmax=1
)
plt.title("Mapa ciepła korelacji - California Housing", fontsize=14)
plt.tight_layout()
plt.show()
print("\n--- Wykres 2: Ranking najważniejszych cech wpływających na cenę ---")
korelacja_z_cena = macierz_korelacji['MEDV'].drop('MEDV').sort_values()
plt.figure(figsize=(9,5))
korelacja_z_cena.plot(kind='barh', color='teal')
plt.title("Korelacja cech z wartością domu (median_house_value)", fontsize=14)
plt.xlabel("Współczynnik korelacji Pearsona")
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
X = df.drop('MEDV', axis=1)
y = df['MEDV']
X = pd.get_dummies(X, drop_first=True)
#krok 20
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#krok 21 skalowanie
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

print("\n--- Podział i skalowanie zakonczone sukcesem! ---")
print(f"Liczba wierszy trenigowych : {len(X_train)}, testowych: {len(X_test)}")