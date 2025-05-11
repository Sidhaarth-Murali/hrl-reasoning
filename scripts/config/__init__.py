# Configuration directory
# Contains configurations for training with both math and code environments
from . import math_configs
from . import code_configs

# Use configurations as:
# Run with math environment:
# python scripts/run.py --config-name archer_math --env-type math
# 
# Run with code environment:
# python scripts/run.py --config-name bi_level_score_code --env-type code 