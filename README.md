# Programmierkonzepte
Programmierkonzepte und Algorithmen WiSe20/21
Aufgabe 2, YCbCr, Dilatation und MPI

## Installation
1. Repository clonen
2. Anpassen des Bild-Pfades in Zeile 47

```
string path = @"C:\Users\xovoo\Desktop\_temp\data\animal\1.kitten_small.jpg";
```
3. Projekt kompilieren
4. Projekt mit MPI ausführen (n = Anzahl der Prozesse)
```
mpiexec -n 8 DilationMPIBroadcast.exe
```
