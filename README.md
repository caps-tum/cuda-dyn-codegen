# cuda-dyn-codegen
    --type arg                 "cpu", "gpu" or "dyn"
    --width arg (=4096)        width of the matrix
    --height arg (=4096)       height of the matrix
    --numIterations arg (=50)  number of iterations to calculate
    --csv arg (=output.csv)    name of the csv file
    --matrix arg               name of the matrix output file

    GPU Options:
    --kernel arg (=1)          version of the kernel to use

z. B. berechnet

    gpu-impl.exe --type gpu --kernel 2 --width 5000 --height 5000 --csv test.csv --matrix test.txt

50 Iterationen mit einer 5000x5000 Matrix mit dem 2. Kernel, speichert die Messdaten in
test.csv und die Matrix mit dem Ergebnis in test.txt.

    Gpu
    Width;Height;Stencils/Second (all);Stencils/Second (comput)
    5000;5000;4331542222;5806602510

* "all" umfasst das Allokieren von Speicher, das Umkopieren und die eigentliche Berechnung
* "comput" bezieht sich nur auf die Anzahl der Berechnungen

Kernel 6 und der zur Laufzeit kompilierte Kernel sind identisch, beide benutzen momentan einen Stencil
mit der Form:

      1
      1
    11111
      1
      1

Wobei Kernel 6 die Gewichtung der einzelnen Elemente aus einem Array ausliest und der dynamische Kernel diese zur
Laufzeit einkompiliert bekommt. Dementsprechend spart der dynamische Kernel bei einer Gewichtung von 1 für alle
Elemente 9 Multiplikationen, 9 Additionen und vor allem die zugehörigen Speicherzugriffe ein.
