import importlib
import pkgutil

# Auto-import all modules in the enhancements package
for _, module_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_name}")
