open_project cs_proj
set_top scr_conv_stream
add_files scr_conv_stream.cpp
add_files -tb tb_conv_stream.cpp
open_solution sol1 -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 3.3 -name default
puts "=== CSIM ==="; csim_design
puts "=== CSYNTH ==="; csynth_design
puts "=== EXPORT ==="; export_design -format ip_catalog
puts "ALL_DONE_CS"; exit
