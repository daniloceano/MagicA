"""
Statistical distribution fitting and adjustment for wind data
"""

import numpy as np
from scipy import stats
from typing import Union, Dict, Any, Optional, Tuple
import warnings

from .data_processor import DataProcessor


class MagicAdjuster:
    """
    Simple class for statistical distribution fitting using SciPy.
    
    This class takes processed data and fits statistical distributions,
    designed to be extended with goodness-of-fit tests and advanced features.
    """
    
    def __init__(self, data_processor: DataProcessor):
        """
        Initialize the adjuster with processed data.
        
        Parameters
        ----------
        data_processor : DataProcessor
            Processor instance with loaded data
        """
        if data_processor.data is None:
            raise ValueError("DataProcessor must have data loaded.")
        
        self.data_processor = data_processor
        self.data = data_processor.get_data_array()
        self.fitted_distribution = None
        self.fitted_params = None
        self.distribution_name = None
        
    def fit_distribution(self, distribution: Union[str, object], **kwargs) -> 'MagicAdjuster':
        """
        Fit a statistical distribution to the data.
        
        Parameters
        ----------
        distribution : str or scipy.stats distribution
            Distribution to fit. Can be:
            - String: 'weibull', 'gamma', 'lognorm', 'norm', etc.
            - SciPy distribution object: stats.weibull_min, stats.gamma, etc.
        **kwargs : dict
            Additional arguments passed to the distribution's fit method
            
        Returns
        -------
        MagicAdjuster
            Adjuster instance with fitted distribution
        """
        # Handle string distribution names
        if isinstance(distribution, str):
            distribution_map = {
                'alpha': stats.alpha,
                'anglit': stats.anglit,
                'arcsine': stats.arcsine,
                'beta': stats.beta,
                'betaprime': stats.betaprime,
                'binom': stats.binom,
                'boltzmann': stats.boltzmann,
                'burr': stats.burr,
                'burr12': stats.burr12,
                'cauchy': stats.cauchy,
                'chi': stats.chi,
                'chi2': stats.chi2,
                'cosine': stats.cosine,
                'crystalball': stats.crystalball,
                'dgamma': stats.dgamma,
                'dweibull': stats.dweibull,
                'erlang': stats.erlang,
                'expon': stats.expon,
                'exponnorm': stats.exponnorm,
                'exponweib': stats.exponweib,
                'exponpow': stats.exponpow,
                'f': stats.f,
                'fatiguelife': stats.fatiguelife,
                'fisk': stats.fisk,
                'foldcauchy': stats.foldcauchy,
                'foldnorm': stats.foldnorm,
                'frechet_l': stats.frechet_l,
                'frechet_r': stats.frechet_r,
                'gamma': stats.gamma,
                'gausshyper': stats.gausshyper,
                'genbeta': stats.genbeta,
                'genchi2': stats.genchi2,
                'genexpon': stats.genexpon,
                'genextreme': stats.genextreme,
                'genf': stats.genf,
                'genhyperbolic': stats.genhyperbolic,
                'geninvgauss': stats.geninvgauss,
                'genlogistic': stats.genlogistic,
                'genpareto': stats.genpareto,
                'gennorm': stats.gennorm,
                'genpareto': stats.genpareto,
                'genpowerlaw': stats.genpowerlaw,
                'genrayleigh': stats.genrayleigh,
                'genschechter': stats.genschechter,
                'genweibull': stats.genweibull,
                'gilbrat': stats.gilbrat,
                'gumbel_l': stats.gumbel_l,
                'gumbel_r': stats.gumbel_r,
                'halfcauchy': stats.halfcauchy,
                'halflogistic': stats.halflogistic,
                'halfnorm': stats.halfnorm,
                'halfgumbel_l': stats.halfgumbel_l,
                'halfgumbel_r': stats.halfgumbel_r,
                'hypsecant': stats.hypsecant,
                'invgamma': stats.invgamma,
                'invgauss': stats.invgauss,
                'invweibull': stats.invweibull,
                'johnsonsb': stats.johnsonsb,
                'johnsonsu': stats.johnsonsu,
                'kappa3': stats.kappa3,
                'kappa4': stats.kappa4,
                'ksone': stats.ksone,
                'kstwobign': stats.kstwobign,
                'laplace': stats.laplace,
                'loggamma': stats.loggamma,
                'loglaplace': stats.loglaplace,
                'lognorm': stats.lognorm,
                'logistic': stats.logistic,
                'lognorm': stats.lognorm,
                'loguniform': stats.loguniform,
                'lomax': stats.lomax,
                'maxwell': stats.maxwell,
                'mielke': stats.mielke,
                'moyal': stats.moyal,
                'nakagami': stats.nakagami,
                'ncf': stats.ncf,
                'nct': stats.nct,
                'norm': stats.norm,
                'norminvgauss': stats.norminvgauss,
                'pareto': stats.pareto,
                'pearson3': stats.pearson3,
                'powerlaw': stats.powerlaw,
                'powerlognorm': stats.powerlognorm,
                'powernorm': stats.powernorm,
                'rayleigh': stats.rayleigh,
                'reciprocal': stats.reciprocal,
                'rice': stats.rice,
                'recipinvgauss': stats.recipinvgauss,
                'semicircular': stats.semicircular,
                'skewcauchy': stats.skewcauchy,
                'skewnorm': stats.skewnorm,
                'students_t': stats.t,
                'tmax': stats.tmax,
                'tmin': stats.tmin,
                'trapz': stats.trapz,
                'triang': stats.triang,
                'truncexpon': stats.truncexpon,
                'truncinvgauss': stats.truncinvgauss,
                'truncnorm': stats.truncnorm,
                'uniform': stats.uniform,
                'vonmises': stats.vonmises,
                'vonmises_line': stats.vonmises_line,
                'wald': stats.wald,
                'weibull_max': stats.weibull_max,
                'weibull_min': stats.weibull_min,
                'wrapcauchy': stats.wrapcauchy
            }
            if distribution.lower() not in distribution_map:
                available = list(distribution_map.keys())
                raise ValueError(f"Unknown distribution '{distribution}'. Available: {available}")
            
            self.fitted_distribution = distribution_map[distribution.lower()]
            self.distribution_name = distribution.lower()
        else:
            # Assume it's a SciPy distribution object
            self.fitted_distribution = distribution
            self.distribution_name = getattr(distribution, 'name', str(distribution))
        
        # Fit the distribution
        try:
            self.fitted_params = self.fitted_distribution.fit(self.data, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to fit {self.distribution_name} distribution: {e}")
        
        return self
    
    def get_fitted_params(self) -> Tuple:
        """
        Get the fitted distribution parameters.
        
        Returns
        -------
        tuple
            Fitted parameters of the distribution
        """
        if self.fitted_params is None:
            raise ValueError("No distribution has been fitted yet.")
        
        return self.fitted_params
    
    def get_distribution_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted distribution.
        
        Returns
        -------
        dict
            Dictionary with distribution information
        """
        if self.fitted_distribution is None:
            raise ValueError("No distribution has been fitted yet.")
        
        return {
            'name': self.distribution_name,
            'parameters': self.fitted_params,
            'num_params': len(self.fitted_params),
            'data_size': len(self.data)
        }
    
    def __repr__(self) -> str:
        """String representation of the object."""
        if self.fitted_distribution is None:
            return f"MagicAdjuster(data_size={len(self.data)}, no distribution fitted)"
        return f"MagicAdjuster(data_size={len(self.data)}, distribution='{self.distribution_name}')"
