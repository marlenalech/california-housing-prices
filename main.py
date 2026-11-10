import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df=pd.read_csv('housing (1).csv', sep=r'\s+', header=None)
print("--1. PIERWSZE 5 WIERSZY --")
print(df.head())

print("\n--- Informacje o kolmunach typach---")
print(df.info())
print("\n--- 3 podsumowanie statystyczne describe")
print(df.describe())