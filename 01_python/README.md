
```markdown
# Python Naming Conventions (PEP 8)

Following consistent naming conventions is crucial for the readability and maintainability of Python code. PEP 8 (Python Enhancement Proposal 8) is the official style guide for Python, and it provides naming guidelines for various components in your project.

## 📁 1. Folder Names

- Use lowercase letters.
- Separate words with underscores `_`.
- Keep names short, descriptive, and meaningful.
- Avoid spaces, special characters, or uppercase letters.

**Examples:**
```

project\_name/
data\_processing/
utils/
tests/

```

---

## 📄 2. File Names

- Use lowercase letters.
- Separate words with underscores `_`.
- Avoid spaces, special characters, or uppercase letters.

**Examples:**
```

data\_loader.py
utils.py
test\_models.py

````

---

## 🔡 3. Variable Names

- Use lowercase letters.
- Separate words with underscores `_`.
- Use descriptive names.
- Avoid single-letter names (unless in short loops or math contexts).

**Examples:**
```python
user_name = "John"
total_count = 100
is_valid = True
````

---

## 🔧 4. Function Names

* Use lowercase letters.
* Separate words with underscores `_`.
* Use verbs or verb phrases to describe actions.

**Examples:**

```python
def calculate_total_price(items):
    pass

def is_user_authenticated(user):
    pass
```

---

## 🧱 5. Class Names

* Use **CamelCase** (PascalCase).
* Start each word with a capital letter.
* Use nouns or noun phrases.

**Examples:**

```python
class DataProcessor:
    pass

class UserAuthentication:
    pass
```

---

## 🔒 6. Constants

* Use **UPPERCASE** letters.
* Separate words with underscores `_`.

**Examples:**

```python
MAX_CONNECTIONS = 100
DEFAULT_TIMEOUT = 30
```

---

## 📦 7. Module Names

* Use lowercase letters.
* Keep names short and descriptive.
* Avoid underscores unless needed for readability.

**Examples:**

```
math_utils.py
config.py
```

---

## 🔐 8. Private Variables and Functions

* Prefix with a **single underscore** `_` to indicate internal use.
* Prefix with **double underscores** `__` for name mangling (to avoid subclass conflicts).

**Examples:**

```python
_internal_variable = 42

def _helper_function():
    pass

class MyClass:
    def __init__(self):
        self.__private_var = 10
```

---

## 🧾 Summary Table

| Entity        | Convention                    | Example                   |
| ------------- | ----------------------------- | ------------------------- |
| Folder        | `lowercase_with_underscores`  | `data_processing/`        |
| File          | `lowercase_with_underscores`  | `data_loader.py`          |
| Variable      | `lowercase_with_underscores`  | `user_name`               |
| Function      | `lowercase_with_underscores`  | `calculate_total_price()` |
| Class         | `CamelCase`                   | `DataProcessor`           |
| Constant      | `UPPERCASE_WITH_UNDERSCORES`  | `MAX_CONNECTIONS`         |
| Module        | `lowercase`                   | `math_utils.py`           |
| Private       | `_single_leading_underscore`  | `_internal_variable`      |
| Name Mangling | `__double_leading_underscore` | `__private_var`           |

---

By adhering to these conventions, your Python code will be more **consistent**, **readable**, and **aligned with the broader Python community**.

```

---

Would you like me to save it as a downloadable `README.md` file?
```
