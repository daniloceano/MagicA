<div align="center">
  <img src="docs/assets/images/magica_logo_blackbg.png" alt="MagicA Logo" width="300">
  
  # MagicA
  *Magic Adjustment - Advanced Statistical Data Fitting*
</div>

**MagicA** (Magic Adjustment) is a Python package for statistical data adjustment, with special focus on wind data, including advanced fitting techniques, goodness-of-fit tests, and visualization.

## Planned Features

- ✨ Distribution fitting (Weibull, Normal, Lognormal, etc.)
- 📊 Goodness-of-fit tests (Kolmogorov-Smirnov, Anderson-Darling, etc.)
- 🎯 Automatic best distribution selection
- 📈 Integrated visualization functions
- 🌪️ Specialized in wind data analysis
- 🔧 Advanced statistical fitting techniques

## Installation

For local development:

```bash
# Clone the repository
git clone https://github.com/daniloceano/MagicA.git
cd MagicA

# Install in development mode
pip install -e .
```

## Basic Usage (In Development)

```python
import magica as ma

# Load data
processor = ma.DataProcessor()
processor.load_data('wind_data.csv')

# Get basic statistics
stats = processor.get_basic_stats()
print(stats)

# Get data as numpy array
data_array = processor.get_data_array()
```

## Current Project Structure

```
magica/
├── __init__.py
├── core/
│   ├── __init__.py
│   └── data_processor.py  # Basic data loading and processing
└── utils/                 # Utilities (in development)
    └── __init__.py
```

## Development

This project is in early development. The approach is incremental, adding features as needed.

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests (when implemented)
pytest

# Run linting
flake8 magica/
black magica/
```

## License

MIT License

## Author

- **Danilo Couto de Souza**
- Email: danilo.oceano@gmail.com
- GitHub: [@daniloceano](https://github.com/daniloceano)
