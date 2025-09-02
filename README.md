# Entity Matching

This package provides tools for solving the **entity matching problem**:  
given a set of entities (names, terms, or references), determine whether they refer to the same real-world object and map them to a canonical representation.

---

## Overview

For example:

```python
data_dict = {
    '澳門': 'Refers to Macao, the Special Administrative Region of China and former Portuguese colony. A beautiful place',
    '濠鏡澳': 'An ancient Chinese name for Macao, literally meaning "Oyster Mirror Bay," referring to the area\'s geographic features before it became known as Macao.',
    '香港': 'Refers to Hong Kong, the Special Administrative Region of China and former British colony.',
    '鏡海': 'An ancient poetic name for the waters around Macao, literally meaning "Mirror Sea."'
}
````

In this example, **`澳門`, `濠鏡澳`, and `鏡海`** all refer to the same entity: **Macau**.
The core method of this library, `stem()`, returns the canonical form (or *stem*) of an entity:

```python
stem('澳門')        # 澳門
stem('濠鏡澳')      # 澳門
stem('香港')        # 香港
stem('鏡海')        # 澳門
```

---

## Features

* **Entity resolution**: Identifies different surface forms referring to the same real-world entity.
* **Simple interface**: A straightforward workflow with `add_dict()`, `build()`, and `stem()`.
* **Asynchronous support**: Provides both synchronous (`build()`) and asynchronous (`abuild()`) processing.

---
## Installation

### 1. **Install directly from GitHub (recommended)**

If your code is stored in a public GitHub repo, you can install it with:

```bash
pip install git+https://github.com/anton/entity-matching.git
```

This will fetch the latest version directly from GitHub.

If you have tags or branches, you can specify them:

```bash
# Specific branch
pip install git+https://github.com/awepo-pro/entity_matching.git

# Specific release tag
pip install git+https://github.com/awepo-pro/entity_matching.git@v1.0.0
```

If you use `uv`, 

```bash
uv add git+https://github.com/awepo-pro/entity_matching
```

---

### 2. **Install locally from source**

If someone clones your repo, they can install it with:

```bash
git clone https://github.com/awepo-pro/entity_matching.git
cd entity-matching
pip install .
```

or (for development mode):

```bash
pip install -e .
```

---

## Usage Example
check my [blog](https://awepo-pro.github.io/posts/entity-matching/) for more details

---

## Bug discovered
NONE!

---

## License

MIT License
