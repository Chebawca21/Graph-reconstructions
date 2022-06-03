# ZZSN Grupa 29<br>Rekonstrukcja grafów za pomocą modelów Graphite, GAE oraz VGAE

# Instalacja
Po sklonowaniu tworzenie środowiska conda

<p><code>$ conda env create -f environment.yml<br>
         $ conda activate zzsn-grupa29</code></p>

# Uruchomienie
Uruchomienie generacji grafów ze wszystkich 6 "rodzin"

<p><code>$ python data.py</code><p>

<br>

Uruchomienie uczenia wszystkich 4 modeli dla każdego zbioru grafów

<p><code>$ python test.py</code><p>
Zmiana parametrów każdego modelu jest możliwa poprzez zmianę parametrów funkcji