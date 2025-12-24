"""
Configuration management module for DPP-PDC.
Handles loading and validation of experiment configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration class for experiments with validation."""
    
    # Data settings
    temperatures: List[str] = field(default_factory=lambda: ["850K"])
    file_path_template: str = "datasets/Cu-Mg_Zn_{temperature}.csv"
    
    # Algorithm settings
    algorithms: List[str] = field(default_factory=lambda: ["RS", "PDC", "DPP-PDC"])
    batch_sizes: List[int] = field(default_factory=lambda: [5, 8, 10, 12])
    n_runs: int = 10
    n_initial_sampling: int = 10
    max_iterations: Optional[int] = None  # None = auto-calculate
    
    # Bayesian settings
    prior_mu: float = -4.0
    prior_sigma: float = 4.0
    mcmc_samples: int = 1000
    mcmc_tune: int = 1000
    mcmc_chains: int = 1
    target_accept: float = 0.9
    
    # Output settings
    output_dir: str = "results"
    save_plots: bool = True
    verbose: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        # Validate algorithms
        valid_algorithms = {"RS", "PDC", "DPP-PDC", "K-Medoids"}
        for algo in self.algorithms:
            if algo not in valid_algorithms:
                raise ValueError(
                    f"Invalid algorithm: {algo}. "
                    f"Valid options: {valid_algorithms}"
                )
        
        # Validate batch sizes
        for bs in self.batch_sizes:
            if bs < 1:
                raise ValueError(f"Batch size must be positive, got {bs}")
        
        # Validate n_runs
        if self.n_runs < 1:
            raise ValueError(f"n_runs must be positive, got {self.n_runs}")
        
        # Validate MCMC parameters
        if self.target_accept <= 0 or self.target_accept >= 1:
            raise ValueError(
                f"target_accept must be between 0 and 1, got {self.target_accept}"
            )
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "ExperimentConfig":
        """Create configuration from dictionary (e.g., loaded from TOML)."""
        # Flatten nested config structure
        flat_config = {}
        
        # Handle nested sections
        if "experiment" in config_dict:
            flat_config.update(config_dict["experiment"])
        if "data" in config_dict:
            flat_config.update(config_dict["data"])
        if "output" in config_dict:
            flat_config.update(config_dict["output"])
        if "bayesian" in config_dict:
            flat_config.update(config_dict["bayesian"])
        
        # Also handle flat config
        for key, value in config_dict.items():
            if not isinstance(value, dict):
                flat_config[key] = value
        
        # Create instance with available parameters
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in flat_config.items() if k in valid_fields}
        
        return cls(**filtered_config)
    
    @classmethod
    def from_toml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from TOML file."""
        import sys
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib
        
        with open(path, "rb") as f:
            config_dict = tomllib.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "experiment": {
                "temperatures": self.temperatures,
                "algorithms": self.algorithms,
                "batch_sizes": self.batch_sizes,
                "n_runs": self.n_runs,
                "n_initial_sampling": self.n_initial_sampling,
            },
            "data": {
                "file_path_template": self.file_path_template,
            },
            "output": {
                "output_dir": self.output_dir,
                "save_plots": self.save_plots,
                "verbose": self.verbose,
            },
            "bayesian": {
                "prior_mu": self.prior_mu,
                "prior_sigma": self.prior_sigma,
                "mcmc_samples": self.mcmc_samples,
                "mcmc_tune": self.mcmc_tune,
                "mcmc_chains": self.mcmc_chains,
                "target_accept": self.target_accept,
            },
        }
    
    def get_data_path(self, temperature: str) -> str:
        """Get data file path for a specific temperature."""
        return self.file_path_template.format(temperature=temperature)
    
    def set_quick_test(self) -> "ExperimentConfig":
        """Set configuration for quick testing."""
        self.temperatures = ["850K"]
        self.batch_sizes = [12]
        self.n_runs = 2
        return self
    
    def set_temperatures(self, temperatures: List[str]) -> "ExperimentConfig":
        """Set temperatures to test."""
        self.temperatures = temperatures
        return self
    
    def set_algorithms(self, algorithms: List[str]) -> "ExperimentConfig":
        """Set algorithms to test."""
        self.algorithms = algorithms
        self._validate()
        return self
    
    def set_batch_sizes(self, batch_sizes: List[int]) -> "ExperimentConfig":
        """Set batch sizes to test."""
        self.batch_sizes = batch_sizes
        self._validate()
        return self
    
    def set_file_path_template(self, template: str) -> "ExperimentConfig":
        """Set file path template."""
        self.file_path_template = template
        return self
    
    def summary(self) -> str:
        """Return a summary string of the configuration."""
        return (
            f"ExperimentConfig(\n"
            f"  temperatures={self.temperatures},\n"
            f"  algorithms={self.algorithms},\n"
            f"  batch_sizes={self.batch_sizes},\n"
            f"  n_runs={self.n_runs},\n"
            f"  output_dir='{self.output_dir}'\n"
            f")"
        )
