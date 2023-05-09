# RYCHLÉ DISKRIMINATIVNÍ NEURONOVÉ SÍTĚ PRO OPRAVU TEXTU

Autor: Sebastián Chupáč
Login: xchupa03

Tento projekt obsahuje skripty, natrénované modely a trénovacie, validačné a testovacie súbory pre trénovanie a testovanie neurónových sietí ktoré detekujú a opravujú chyby v texte. Zoznam zdrojových súborov:

- **training_detection.py** trénovací skript pre modely na detekciu chýb.

- **training_correction.py** trénovací skript pre modely na pozične viazanú korekciu chýb.

- **training_correction_CTC.py** trénovací skript pre modely na korekciu chýb využívajúce CTC.

- **models.py** súbor obsahuje definície tried jednotlivých architektúr.

- **dataset.py**  implementuje triedu MyDataset, ktorá zo vstupného súboru textových úsekov vytvorí a uloží dáta vo formáte item.

- **dataset_pad.py**  implementuje triedu MyDataset, ktorá zo vstupného súboru textových úsekov vytvorí a uloží dáta vo formáte item, využíva padding a používa sa s CTC modelmi.

- **process_corups.py** tento skript obsahuje funkcie pre spracovanie korpusu a vytváranie súborov s textovými úsekmi. 

- **inference.py** skript na testovanie, ktorý do súboru zaznamená jednotlivé vstupy, groud truth avýstupy zneurónovej siete v textovom formáte, pre počítanie štatistík. 

- **eval.py**  skript na počítanie štatistík zvýstupu inference.py, prípadne štatistík trénovacích a testovacích súborov s úsekmi textu.

- **ansi_print.py**  obsahuje funkciu a\_print(text, truth, corr\_color, err\_color), ktorá využíva ansi kódovanie farieb vterminály pre farebný výpis textu do terminálu, využiteľné hlavne pri pozične viazaných riešeniach. 

- **cudann.py** kontrola kompatibility a funkčnosti CUDA

- **train-add-typos.py** súbor na experimentovanie s generovaním chýb

Zoznam potrebných Python3.10 knižníc:

- torch 1.13
- numpy 1.23.4
- Levenshtein 0.20.9

