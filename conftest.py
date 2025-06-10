# conftest.py
import secrets
import numpy as np
import pytest

# ---------- options ---------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--seed",
        action="store",
        default=None,
        help="Global numpy random seed. "
             "If omitted, a fresh random seed is generated."
    )

# ---------- session setup ---------------------------------------------------
def pytest_configure(config):
    # Determine or create the seed
    cli_value = config.getoption("--seed")
    seed = int(cli_value) if cli_value is not None else secrets.randbelow(2**32)

    # Make it available everywhere
    config._numpy_seed = seed          # private stash for later hooks
    np.random.seed(seed)               # global effect

# ---------- report/header hooks --------------------------------------------
def pytest_report_header(config):
    """Line shown *before* collection starts."""
    return f"numpy.random.seed set to: {config._numpy_seed}"

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Extra lines shown *after* the usual short summary."""
    terminalreporter.section("Random Seed")
    terminalreporter.write_line(f"numpy.random.seed = {config._numpy_seed}")
