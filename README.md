<div align="center">
  <img src="docs/assets/images/magica_logo_blackbg.svg" alt="MagicA Logo" width="300">
  
  # MagicA
  *Magic Adjustment - Advanced Statistical Data Fitting*
</div>

**MagicA** (Magic Adjustment) is a Python package for statistical data adjustment, with special focus on wind data and extreme value analysis, including advanced fitting techniques, goodness-of-fit tests, and visualization.

## Features

- 📊 **Distribution Fitting**: Weibull, Normal, Lognormal, Gumbel, GEV, and more
- ✅ **Goodness-of-Fit Tests**: Kolmogorov-Smirnov and Chi-square tests
- 🎯 **Extreme Value Analysis**: Peaks Over Threshold (POT), return values, and return periods
- 🌪️ **Directional Wind Analysis**: Specialized tools for directional extreme wind analysis
- 📈 **Visualization**: Built-in plotting functions including polar plots for directional data
- 🔧 **Synthetic Data Generation**: Generate realistic wind data with directional characteristics for testing and development

## Documentation

📖 **Full Documentation**: [magica.readthedocs.io](https://magica.readthedocs.io/en/latest/)

Complete documentation including tutorials, API reference, and examples.

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
