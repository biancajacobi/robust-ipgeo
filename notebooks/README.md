# notebooks/

Interaktive **Darstellungs- und Explorationsschicht** — *nicht* der reproduzierbare
Kern des Projekts.

## Rollenteilung (bewusst)

- **Berechnung & reproduzierbare Pipeline** bleiben in `data/`, `estimators/`,
  `eval/` und `experiments/` (reine `.py`-Module, getestet, diffbar, deterministisch
  ausführbar). Das ist der bewertete „lauffähige Teil" und der forensisch saubere
  Pfad — Notebooks sind wegen frei wählbarer Zellreihenfolge dafür ungeeignet.
- **Notebooks importieren diese Module nur und visualisieren** die fertigen
  Ergebnisse aus `eval/out/`. Hier wird **keine** Schätz-/Auswertungslogik
  dupliziert; gehört etwas in die Pipeline, wandert es nach `eval/report.py`
  bzw. `experiments/`.

## Reproduktion

Erst die Pipeline laufen lassen (erzeugt die CSVs in `eval/out/`):

```bash
python experiments/exp_accuracy.py
python experiments/exp_contamination.py
python experiments/exp_samplesize.py
```

Dann `explore.ipynb` öffnen (Kernel = Projekt-`.venv`). „Restart & Run All"
muss jederzeit ohne manuelle Zwischenschritte durchlaufen.
