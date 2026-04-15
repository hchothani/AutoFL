# test scripts

this directory contains test scripts to verify the functionality of the autofl system.

IMP NOTE: a lot of these test scripts are generated using claude-sonnect LLM.

they are rough and may contain errors. the reason for using claude is just to generate quick fast tests so can check if the system is working.

## running tests

run tests from the main project directory:

```bash
# test configuration system
python tests/test_config_system.py

# test CIFAR10 workloads
python tests/test_cifar10_training.py

# test CIFAR100 workloads
python tests/test_cifar100_training.py
```
