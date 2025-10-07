<div align="center">
  <img src="docs/assets/images/magica_logo_blackbg.svg" alt="MagicA Logo" width="300">
  
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

## Documentation

📖 **Documentation**: [magica.readthedocs.io](https://magica.readthedocs.io/en/latest/) *(under construction)*

The documentation is currently being built and will include:

## Example Wind Data

This repository includes a real wind speed dataset from the INMET meteorological station in Rio Grande, Brazil:

- **File:** `data/INMET_RIO_GRANDE_wind.csv`
- **Source:** INMET (Instituto Nacional de Meteorologia)
- **Description:** Wind speed measurements for the Rio Grande station
- **Usage:**
    - Use in tutorials, development tests, and benchmarks
    - Example:
      ```python
      import pandas as pd
      wind_df = pd.read_csv('data/INMET_RIO_GRANDE_wind.csv')
      print(wind_df.head())
      ```

Please cite the data source if you use it in publications or reports.

## Development

This project is in early development. The approach is incremental, adding features as needed.

## License

MIT License

## Author

- **Danilo Couto de Souza**
- Email: danilo.oceano@gmail.com
- GitHub: [@daniloceano](https://github.com/daniloceano)
