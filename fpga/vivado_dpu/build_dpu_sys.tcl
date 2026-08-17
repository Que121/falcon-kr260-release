# Phase-2: DPU (B4096) + our two custom IPs in ONE bitstream, Vivado 2022.2, KR260 (xck26).
# Strategy: pre-create the project with IP repos (DPU v4.1 + our 2 IPs re-exported in 2022.2), then SOURCE
# the LogicTronix KR260 B4096 base Tcl (its create_project is guarded `if {$list_projs eq ""}`, so it builds
# the DPU into OUR project), then APPEND our two IPs on a dedicated 200 MHz pl_clk2 (control via the free
# M_AXI_HPM0_FPD, data via the free S_AXI_HP2_FPD), then impl -> dpu.bit/.hwh/.xclbin.
# Paths are set via the three vars below (overridden by the launcher).
#
# DRAFT — to test + iterate once Vivado 2022.2 is installed and the IPs are re-exported in 2022.2.

if { ![info exists DPU_IP] }   { set DPU_IP   "$::env(HOME)/dpu_assets/dpu_ip/DPUCZDX8G_ip_repo_VAI_v3.0" }
if { ![info exists GATHER_IP] } { set GATHER_IP "$::env(HOME)/OccFPGA/fpga/hls22/ip_repo/gather" }
if { ![info exists RESIZE_IP] } { set RESIZE_IP "$::env(HOME)/OccFPGA/fpga/hls22/ip_repo/resize" }
if { ![info exists BASE_TCL] } { set BASE_TCL  "$::env(HOME)/dpu_assets/KR260-DPU-TRD-Vitis-AI-3.0/DPU-B4096-architecture/VIVADO-Design/kr260-dpu-trd-b4096-dec15.tcl" }
set BOARDPART {xilinx.com:kr260_som:part0:1.1}

create_project -force dpusys ./dpusys -part xck26-sfvc784-2LV-c
catch { set_property BOARD_PART $BOARDPART [current_project] }
set_property ip_repo_paths [list $DPU_IP $GATHER_IP $RESIZE_IP] [current_project]
update_ip_catalog -rebuild

# build the DPU base design (bd "top") into our project
source $BASE_TCL
current_bd_design [get_bd_designs top]

set ps  [get_bd_cells zynq_ultra_ps_e]
# enable the free control master (HPM0_FPD=GP0), free data slave (HP2=GP4, 128b), and a 200 MHz PL clock (pl_clk2)
set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__USE__S_AXI_GP4 {1} \
  CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
  CONFIG.PSU__FPGA_PL2_ENABLE {1} ] $ps
catch { set_property CONFIG.PSU__CRL_APB__PL2_REF_CTRL__FREQMHZ {200} $ps }

create_bd_cell -type ip -vlnv xilinx.com:hls:bev_gather:1.0 gather_0
create_bd_cell -type ip -vlnv xilinx.com:hls:resize_bilinear:1.0 resize_0
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 psr_ip
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smc_ctrl2
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] [get_bd_cells smc_ctrl2]
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smc_data2
set_property -dict [list CONFIG.NUM_SI {5} CONFIG.NUM_MI {1}] [get_bd_cells smc_data2]

set clk2 [get_bd_pins zynq_ultra_ps_e/pl_clk2]
connect_bd_net $clk2 [get_bd_pins psr_ip/slowest_sync_clk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e/pl_resetn0] [get_bd_pins psr_ip/ext_reset_in]
set rstn [get_bd_pins psr_ip/peripheral_aresetn]
foreach p {zynq_ultra_ps_e/maxihpm0_fpd_aclk zynq_ultra_ps_e/saxihp2_fpd_aclk smc_ctrl2/aclk smc_data2/aclk gather_0/ap_clk resize_0/ap_clk} { connect_bd_net $clk2 [get_bd_pins $p] }
foreach p {smc_ctrl2/aresetn smc_data2/aresetn gather_0/ap_rst_n resize_0/ap_rst_n} { connect_bd_net $rstn [get_bd_pins $p] }

# control: PS HPM0_FPD -> {gather, resize} s_axi_control
connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e/M_AXI_HPM0_FPD] [get_bd_intf_pins smc_ctrl2/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl2/M00_AXI] [get_bd_intf_pins gather_0/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl2/M01_AXI] [get_bd_intf_pins resize_0/s_axi_control]
# data: 5 m_axi -> PS HP2 (to DDR)
connect_bd_intf_net [get_bd_intf_pins gather_0/m_axi_gmem0] [get_bd_intf_pins smc_data2/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins gather_0/m_axi_gmem1] [get_bd_intf_pins smc_data2/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins gather_0/m_axi_gmem2] [get_bd_intf_pins smc_data2/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins resize_0/m_axi_gmem0] [get_bd_intf_pins smc_data2/S03_AXI]
connect_bd_intf_net [get_bd_intf_pins resize_0/m_axi_gmem1] [get_bd_intf_pins smc_data2/S04_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_data2/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e/S_AXI_HP2_FPD]

assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design
puts "DPU_SYS_BD_OK"
set bdf [get_files top.bd]
make_wrapper -files $bdf -top -import
set_property top top_wrapper [current_fileset]
generate_target all $bdf
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
puts "IMPL_STATUS: [get_property STATUS [get_runs impl_1]]"
open_run impl_1
report_utilization -file util_dpu_sys.rpt
report_timing_summary -file timing_dpu_sys.rpt
write_hw_platform -fixed -include_bit -force dpu_sys.xsa
puts "DPU_SYS_BITSTREAM_DONE"
