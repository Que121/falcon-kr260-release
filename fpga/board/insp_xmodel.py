#!/usr/bin/env python3
import sys, xir
g = xir.Graph.deserialize(sys.argv[1])
subs = g.get_root_subgraph().toposort_child_subgraph()
print("subgraphs:", len(list(subs)))
for i, s in enumerate(g.get_root_subgraph().toposort_child_subgraph()):
    dev = s.get_attr("device") if s.has_attr("device") else "?"
    its = list(s.get_input_tensors()); ots = list(s.get_output_tensors())
    def fp(t): return t.get_attr("fix_point") if t.has_attr("fix_point") else None
    print("[%d] %-4s in=%s out=%s" % (
        i, dev,
        [(t.name[:40], [int(x) for x in t.dims], fp(t)) for t in its],
        [(t.name[:40], [int(x) for x in t.dims], fp(t)) for t in ots]))
