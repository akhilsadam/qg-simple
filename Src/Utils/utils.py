import importlib
import math

def print_config(obj, indent=0):
    """Recursively prints all attributes of a class or object."""
    if not hasattr(obj, "__dict__") and not isinstance(obj, type):  # If it's a simple value, print it
        print(" " * indent + str(obj))
        return

    for attr_name in dir(obj):
        if attr_name.startswith("__"):  # Skip special attributes
            continue

        attr_value = getattr(obj, attr_name)

        if isinstance(attr_value, type):  # If it's a class, recurse into it
            print(" " * indent + f"{attr_name}:")
            print_config(attr_value, indent + 4)
        elif not callable(attr_value):  # Print regular attributes
            print(" " * indent + f"{attr_name} = {attr_value}")
