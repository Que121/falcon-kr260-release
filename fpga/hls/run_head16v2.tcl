open_project head16v2_proj
set_top scr_head16v2
add_files scr_head16v2.cpp
add_files -tb tb_head16v2.cpp
open_solution sol1 -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 3.3 -name default
puts "=== CSIM ==="
csim_design
puts "=== CSYNTH ==="
csynth_design
puts "=== EXPORT ==="
export_design -format ip_catalog
puts "ALL_DONE_V2"
exit
