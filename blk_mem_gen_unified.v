`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:19:47
// Design Name: 
// Module Name: blk_mem_gen_unified
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
module blk_mem_gen_unified #(
    parameter integer BUF_ADDR_W = 16, 
    parameter integer WIDTH      = 128, 
    parameter integer DEPTH      = 65536 
)(
    input  wire                   clka,
    
    // [说明] 读端口控制
    input  wire                   ena,   // 读使能 (来自 PE，只有读的时候才为 1)
    input  wire [BUF_ADDR_W-1:0]  addra, // 读地址
    output wire [WIDTH-1:0]       douta, // 读数据

    // [说明] 写端口控制
    input  wire                   dma_wr_en,   // 写使能
    input  wire [BUF_ADDR_W-1:0]  dma_wr_addr, // 写地址
    input  wire [WIDTH-1:0]       dma_wr_data  // 写数据
);

    xpm_memory_sdpram #(
        .ADDR_WIDTH_A        (BUF_ADDR_W),
        .ADDR_WIDTH_B        (BUF_ADDR_W),
        .WRITE_DATA_WIDTH_A  (WIDTH),
        .READ_DATA_WIDTH_B   (WIDTH),
        .BYTE_WRITE_WIDTH_A  (WIDTH),    

        .MEMORY_SIZE         (WIDTH * DEPTH), 
        
        // [推荐]: "ultra" 使用 URAM (资源占用 30%)
        // 如果想让工具自动决定，可以改为 "auto"
        .MEMORY_PRIMITIVE    ("ultra"),       
        
        .READ_LATENCY_B      (2),
        .WRITE_MODE_B        ("read_first"),
        .USE_MEM_INIT        (0),
        .ECC_MODE            ("no_ecc")
    ) u_weight_buf (
        // ================== Port A (Write) ==================
        .clka   (clka),
        .ena    (dma_wr_en), 
        
        .wea    (dma_wr_en),
        .addra  (dma_wr_addr),
        .dina   (dma_wr_data),

        // ================== Port B (Read) ==================
        .clkb   (clka),
        .enb    (ena), 
        .addrb  (addra),
        .doutb  (douta),
        .rstb   (1'b0),
        .regceb (1'b1),
        .sleep  (1'b0)  
    );

endmodule