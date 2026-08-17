import sys, xir
from collections import Counter
g = xir.Graph.deserialize(sys.argv[1])
subs = g.get_root_subgraph().toposort_child_subgraph()
c = Counter()
for sg in subs:
    try: d = sg.get_attr("device")
    except Exception: d = "?"
    c[d] += 1
print(f"{sys.argv[1]}: {len(subs)} subgraphs  {dict(c)}")
