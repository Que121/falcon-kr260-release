open_project head16_proj
set_top scr_head16
add_files scr_head16.cpp
add_files -tb tb_head16.cpp
open_solution sol1 -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 3.3 -name default
puts "=== CSIM START ==="
csim_design
puts "=== CSIM END ==="
exit
