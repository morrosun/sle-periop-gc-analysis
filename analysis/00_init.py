import os
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ["out", "out/fig", "data", "scripts", "sql"]:
    p = os.path.join(base, sub)
    if not os.path.isdir(p):
        os.makedirs(p)
        print("created", p)
    else:
        print("exists ", p)
