import importlib.util
import sys
def load_source(module_name,py_path):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    my_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = my_module
    spec.loader.exec_module(my_module)
    print(my_module)
    return my_module


