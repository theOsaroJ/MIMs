# MIMs

A driver for Active Learning, Bayesian Optimization and Inducing Points.

**Repository:** https://github.com/theOsaroJ/MIMs  
**Source code:** https://github.com/theOsaroJ/MIMs/tree/main/src

**Data files:** https://github.com/theOsaroJ/MIMs/tree/main/data

---

## 📦 Requirements

- **Python 3.6+**  
- POSIX shell (bash, zsh, etc.)  
- `pip` (or `conda`) for installing dependencies  

---

## 🚀 Installation

```bash
# 1. Clone the repo
git clone https://github.com/theOsaroJ/MIMs.git
cd MIMs

# 2. (Optional) Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---
## 📝 Usage

From the project root, invoke the script with your chosen options (acquisition functions). For example:
```bash
python3 src/al_bo.py \
  --acquisition ei \
  --num_points_grid 5000 \
  --query_size 1 \
  > output.txt
```

For the inducing points, simply run:
```bash
python3 src/ip.py 
  > output.txt
```
