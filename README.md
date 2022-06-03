# ZZSN Grupa 29<br>Rekonstrukcja grafów za pomocą modelów Graphite, GAE oraz VGAE

# Instalacja
Po sklonowaniu tworzenie środowiska conda

```
$ conda env create -f environment.yml
$ conda activate zzsn-grupa29
```

# Uruchomienie
Uruchomienie generacji grafów ze wszystkich 6 "rodzin"

```
$ python data.py
```

<br>

Uruchomienie uczenia wszystkich 4 modeli dla każdego zbioru grafów

```
$ python test.py
```

<br>

Uruchomienie uczenia wszystkich 4 modeli dla wybranego zbioru danych
```
$ python test2.py
```

Zmiana parametrów każdego modelu jest możliwa poprzez zmianę parametrów funkcji
