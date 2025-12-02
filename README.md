# MBO Prognose

Een voorspellingsmodel voor MBO-studenteninstroom, dat historische aanmeldingsgegevens gebruikt om toekomstige studentenaantallen per opleiding en leertraject te voorspellen.

## Projectstructuur

```
student-instroom-mbo/
├── configuration.yaml      # Configuratiebestand (startjaar, uitgesloten jaren)
├── input/                  # Map voor invoergegevens
│   ├── applications_enriched_with_context_mboa.csv  # Aanmeldingsgegevens
│   └── inschrijvingen_summary_mboa.csv              # Inschrijvingsoverzicht
├── output/                 # Map voor uitvoer van voorspellingen
├── scripts/
│   ├── models/            # Modelimplementaties
│   │   ├── individual_mbo.py      # Individueel voorspellingsmodel (Bayesian)
│   │   └── sarima_predictor.py    # SARIMA voorspellingslogica
│   ├── prediction_methods/ # Voorspellingsmethoden
│   └── utils/             # Hulpprogramma's
├── cli.py                 # Command-line interface parser
├── main.py                # Hoofdscript voor voorspellingen
└── clear_cache.py         # Script om cache te wissen
```

## Vereisten

- Python 3.12+
- UV package manager

## Installatie

1. Clone de repository:
```bash
git clone https://github.com/cedanl/student-instroom-mbo.git
cd student-instroom-mbo
```

2. Installeer `uv` (indien nog niet geïnstalleerd):
   - **Windows**: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - **macOS/Linux**: `curl -LsSf https://astral.sh/uv/install.sh | sh`

3. Maak en synchroniseer de virtuele omgeving:
   Dit commando maakt de virtuele omgeving aan (indien deze niet bestaat) en installeert/synchroniseert alle afhankelijkheden uit `pyproject.toml`.
```bash
uv sync
```

4. Activeer de virtuele omgeving:
   - **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

5. (Optioneel) Als je een oudere versie hebt of een synchronisatie wilt forceren:
```bash
uv sync --reinstall
```

## Configuratie

### Environment Variables
Maak een `.env` bestand in de root directory om de locaties van je gegevensbestanden te specificeren:

```env
ROOT_PATH="C:\\Path\\To\\Your\\Project\\Root"
```

### Configuration File
Het `configuration.yaml` bestand bevat belangrijke instellingen:
*   **individual_start_year**: Startjaar voor het trainen van het individuele model (standaard: 2023)
*   **covid_year**: Uitgesloten jaar vanwege afwijkende aanmeldingsdeadlines (standaard: 2020)

**Let op:** Filtering is verwijderd uit de configuratie omdat het model snel genoeg draait om alle opleidingen en leertrajecten in één keer te verwerken.

## Gebruik

1. Plaats je aanmeldingsgegevens CSV in de `input/` map.

2. Werk de configuratie bij in `configuration.yaml` indien nodig.

3. Draai het voorspellingsmodel:
```bash
# Draai voor het huidige jaar en week
uv run main.py

# Draai voor een specifiek jaar en week
uv run main.py -y 2024 -w 42

# Draai voor meerdere weken
uv run main.py -y 2024 -w 1 2 3

# Draai voor een bereik van weken
uv run main.py -y 2024 -w 10:20

# Schrijf resultaten naar bestand
uv run main.py -y 2024 -w 42 -wf

# Verbose output voor debugging
uv run main.py -y 2024 -w 42 -v
```

**Command Line Arguments:**
*   `-w`, `--weeks`: Een of meer weeknummers of bereiken (bijv. `5 6 7`, `10:15`, `39:38`)
*   `-y`, `--years`: Een of meer academische jaren of bereiken (bijv. `2023 2024`, `2022:2025`)
*   `-wf`, `--write-file`: Schrijf voorspellingen naar bestand
*   `-p`, `--print`: Print programma-uitvoer
*   `-v`, `--verbose`: Print gedetailleerde model-uitvoer

De resultaten worden opgeslagen in `output/output_mbo_[timestamp].xlsx`.

## Model Details

Het huidige model gebruikt **Bayesian voorspellingsmethoden**:

*   **Individual_ratio**: Bayesian ratio-gebaseerde voorspelling
*   **Individual_mean**: Bayesian cluster-gebaseerde voorspelling

**Let op:** SARIMA-functionaliteit is momenteel uitgeschakeld. Zie `README_SARIMA.md` voor details over de status en hoe deze in de toekomst opnieuw kan worden ingeschakeld.

## Uitbreiden van Voorspellingen

### Een Nieuw Model Toevoegen
Om de voorspellingen uit te breiden met een nieuwe module:

1.  Maak een nieuw script in `scripts/models/` (bijv. `nieuwe_module.py`).
2.  Implementeer een klasse of functie die een DataFrame met voorspellingen retourneert.
3.  Zorg dat de uitvoer overeenkomt met de structuur van de bestaande resultaten (kolommen voor `Schooljaar`, `Opleidingscode`, etc.).
4.  Update `main.py` om het nieuwe model aan te roepen.

### Ensemble Creatie
In de toekomst kan een ensemble-model worden gemaakt om voorspellingen van meerdere modules te combineren:

1.  Maak een nieuw script `scripts/models/ensemble.py`.
2.  Importeer de individuele modellen (zoals `scripts.models.individual_mbo`).
3.  Draai elk model afzonderlijk om hun voorspellingen te verkrijgen.
4.  Combineer de resultaten, bijvoorbeeld door het gemiddelde te nemen of een gewogen gemiddelde op basis van historische nauwkeurigheid.
5.  Update `main.py` om het ensemble-script aan te roepen in plaats van alleen het individuele model.

---

# English Documentation

A prediction model for MBO student enrollments, using historical enrollment data to forecast future student numbers per program and learning path.

## Requirements

- Python 3.12+
- UV package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cedanl/student-instroom-mbo.git
cd student-instroom-mbo
```

2. Install `uv` (if not already installed):
   - **Windows**: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - **macOS/Linux**: `curl -LsSf https://astral.sh/uv/install.sh | sh`

3. Create and sync the virtual environment:
   This command will create the virtual environment (if it doesn't exist) and install/sync all dependencies from `pyproject.toml`.
```bash
uv sync
```

4. Activate the virtual environment:
   - **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

5. (Optional) If you have an older version or need to force a sync:
```bash
uv sync --reinstall
```

## Configuration

### Environment Variables
Create a `.env` file in the root directory to specify the locations of your data files:

```env
ROOT_PATH="C:\\Path\\To\\Your\\Project\\Root"
```

### Configuration File
The `configuration.yaml` file contains important settings:
*   **individual_start_year**: Starting year for individual model training (default: 2023)
*   **covid_year**: Excluded year due to different application deadlines (default: 2020)

**Note:** Filtering has been removed from the configuration since the model runs quickly enough to process all programs and learning paths in one go.

## Usage

1. Place your enrollment data CSV in the `input/` directory.

2. Update the configuration in `configuration.yaml` if needed.

3. Run the prediction model:
```bash
# Run with current year and week
uv run main.py

# Run with specific year and week
uv run main.py -y 2024 -w 42

# Run with multiple weeks
uv run main.py -y 2024 -w 1 2 3

# Run with a range of weeks
uv run main.py -y 2024 -w 10:20

# Write results to file
uv run main.py -y 2024 -w 42 -wf

# Verbose output for debugging
uv run main.py -y 2024 -w 42 -v
```

**Command Line Arguments:**
*   `-w`, `--weeks`: One or more week numbers or ranges (e.g., `5 6 7`, `10:15`, `39:38`)
*   `-y`, `--years`: One or more academic years or ranges (e.g., `2023 2024`, `2022:2025`)
*   `-wf`, `--write-file`: Write predictions to file
*   `-p`, `--print`: Print program output
*   `-v`, `--verbose`: Print detailed model output

Results are saved to `output/output_mbo_[timestamp].xlsx`.

## Model Details

The current model uses **Bayesian prediction methods**:

*   **Individual_ratio**: Bayesian ratio-based prediction
*   **Individual_mean**: Bayesian cluster-based prediction

**Note:** SARIMA functionality is currently disabled. See `README_SARIMA.md` for details on the status and how to re-enable it in the future.

## Expanding Predictions

### Adding a New Model
To expand predictions with a new module:

1.  Create a new script in `scripts/models/` (e.g., `new_module.py`).
2.  Implement a class or function that returns a DataFrame with predictions.
3.  Ensure the output matches the structure of existing results (columns for `Schooljaar`, `Opleidingscode`, etc.).
4.  Update `main.py` to call the new model.

### Ensemble Creation
In the future, an ensemble model can be created to combine predictions from multiple modules:

1.  Create a new script `scripts/models/ensemble.py`.
2.  Import the individual models (like `scripts.models.individual_mbo`).
3.  Run each model separately to get their predictions.
4.  Combine the results, for example by taking the average or a weighted average based on historical accuracy.
5.  Update `main.py` to call the ensemble script instead of just the individual model.

## Troubleshooting

### Common Issues

#### KeyError: 'Opleidingscode' or similar column errors
**Problem**: CSV file is not being parsed correctly.

**Solution**: 
- Check that your CSV file uses the correct delimiter (`;`, `,`, or tab)
- The system auto-detects delimiters, but verify your file format
- Ensure column names match exactly (case-sensitive)

#### ModuleNotFoundError
**Problem**: Dependencies not installed or virtual environment not activated.

**Solution**:
```bash
# Sync dependencies
uv sync

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

#### No predictions generated
**Problem**: Insufficient historical data or data quality issues.

**Solution**:
- Ensure you have at least 2 years of historical data
- Check that `individual_start_year` in `configuration.yaml` is set correctly
- Verify your data files contain the required columns (see `input/README.md`)
- Run with `-v` flag for detailed output: `uv run main.py -y 2024 -w 8 -v`

#### Cache issues
**Problem**: Stale cached data causing incorrect predictions.

**Solution**:
```bash
# Clear all cache directories
python clear_cache.py
```

### Data Requirements

For best results:
- Minimum 2 years of historical enrollment data
- Complete weekly application status updates
- Consistent program codes across years
- Final enrollment counts in the summary file

### Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review `input/README.md` for data requirements
3. Run with verbose output: `uv run main.py -y 2024 -w 8 -v`
4. Check the GitHub issues page for similar problems

## License

See LICENSE file for details.
