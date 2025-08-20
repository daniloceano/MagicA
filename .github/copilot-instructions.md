# MagicA Project Instructions

## Project Overview
MagicA (Magic Adjustment) is a Python package for statistical data adjustment, with special focus on wind data, including advanced fitting techniques, goodness-of-fit tests, and visualization.

## Developer Information
- **Developer**: Danilo Couto de Souza
- **Email**: danilo.oceano@gmail.com
- **GitHub**: daniloceano

## Behavior Guidelines for Copilot

### IMPORTANT: Conservative Development Approach
- **NEVER create multiple files or extensive code automatically**
- **ALWAYS ask before creating new modules, classes, or functions**
- **Work incrementally** - create only what is explicitly requested
- **Focus on single, specific tasks** rather than broad implementations
- The developer wants full control over what code is created

### Code Creation Rules
1. **Ask first**: Before creating any new file or significant code block, ask for permission
2. **Start simple**: Begin with minimal implementations that can be expanded later
3. **One thing at a time**: Focus on single functions or classes per request
4. **Avoid assumptions**: Don't assume project structure or requirements beyond what's stated

### Project Maintenance
- **ALWAYS update `__init__.py` files** when adding or removing modules
- **Update imports** in parent modules when structure changes
- **Keep documentation in sync** with code changes
- **Maintain clean project structure**

## Technical Stack
- **Core**: NumPy, SciPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Testing**: Pytest
- **Documentation**: Sphinx with NumPy style docstrings

## Code Style Guidelines
- Use type hints for all functions
- Follow PEP 8 naming conventions
- Use descriptive variable names for statistical parameters
- Include comprehensive docstrings with examples
- Prefer NumPy/SciPy functions for statistical operations
- Comment code in English when appropriate

## Current Project Structure
```
magica/
├── __init__.py
├── core/
│   ├── __init__.py
│   └── data_processor.py  # Simple data loading and processing
└── utils/                 # Utility functions (when needed)
```

## Development Philosophy
- **Iterative development**: Build features one by one
- **User-controlled**: Developer decides what gets implemented when
- **Clean and simple**: Avoid over-engineering
- **Well-documented**: Every function should have clear documentation
