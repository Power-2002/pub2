`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:26:14
// Design Name: 
// Module Name: unified_feature_buffer
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
module unified_feature_buffer (
  input  wire         clk,
  input  wire         rst_n,
  input  wire         bank_wr_sel,
  input  wire         bank_rd_sel,
  input  wire         wr_en,
  input  wire [16:0]  wr_addr,
  input  wire [127:0] wr_data,
  input  wire         rd_en,
  input  wire [16:0]  rd_addr,
  output reg  [127:0] rd_data,
  output reg          rd_valid
);
  localparam integer DEPTH  = 8192;
  localparam integer WIDTH  = 128;
  localparam integer ADDR_W = 13;
  localparam integer BYTE_W = 128; 

  wire [127:0] dout_b0, dout_b1;
  wire [ADDR_W-1:0] wr_addr_s = wr_addr[ADDR_W-1:0];
wire [ADDR_W-1:0] rd_addr_s = rd_addr[ADDR_W-1:0];
  

xpm_memory_sdpram #(
  .ADDR_WIDTH_A       (ADDR_W),
  .ADDR_WIDTH_B       (ADDR_W),
  .MEMORY_SIZE        (WIDTH * DEPTH),
  .MEMORY_PRIMITIVE   ("block"),
  .WRITE_DATA_WIDTH_A (WIDTH),
  .READ_DATA_WIDTH_B  (WIDTH),
  .BYTE_WRITE_WIDTH_A (WIDTH),

  .READ_LATENCY_B     (2),
  .WRITE_MODE_B       ("read_first"),

  .USE_MEM_INIT       (0),
  .ECC_MODE           ("no_ecc"),
  .WAKEUP_TIME        ("disable_sleep"),
  .AUTO_SLEEP_TIME    (0),
  .MESSAGE_CONTROL    (0)
) u_bank0 (
  .clka   (clk),
  .ena    (wr_en && (bank_wr_sel==1'b0)),
  .wea    (wr_en && (bank_wr_sel==1'b0)),
  .addra  (wr_addr_s),
  .dina   (wr_data),

  .clkb   (clk),
  .enb    (rd_en && (bank_rd_sel==1'b0)),
  .addrb  (rd_addr_s),
  .doutb  (dout_b0),

  .rstb   (1'b0),
  .regceb (1'b1),
  .sleep  (1'b0)
  
);

xpm_memory_sdpram #(
  .ADDR_WIDTH_A       (ADDR_W),
  .ADDR_WIDTH_B       (ADDR_W),
  .MEMORY_SIZE        (WIDTH * DEPTH),
  .MEMORY_PRIMITIVE   ("block"),

  .WRITE_DATA_WIDTH_A (WIDTH),
  .READ_DATA_WIDTH_B  (WIDTH),
  .BYTE_WRITE_WIDTH_A (WIDTH),

  .READ_LATENCY_B     (2),
  .WRITE_MODE_B       ("read_first"),

  .USE_MEM_INIT       (0),
  .ECC_MODE           ("no_ecc"),
  .WAKEUP_TIME        ("disable_sleep"),
  .AUTO_SLEEP_TIME    (0),
  .MESSAGE_CONTROL    (0)
) u_bank1 (
  .clka   (clk),
  .ena    (wr_en && (bank_wr_sel==1'b1)),
  .wea    (wr_en && (bank_wr_sel==1'b1)),
  .addra  (wr_addr_s),
  .dina   (wr_data),

  .clkb   (clk),
  .enb    (rd_en && (bank_rd_sel==1'b1)),
  .addrb  (rd_addr_s),
  .doutb  (dout_b1),

  .rstb   (1'b0),
  .regceb (1'b1),
  .sleep  (1'b0)
);
  
  reg rd_en_d1, rd_en_d2;
  reg bank_rd_sel_d1, bank_rd_sel_d2;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_en_d1 <= 1'b0;
      rd_en_d2 <= 1'b0;
      rd_valid <= 1'b0;
      rd_data  <= 128'd0;
      bank_rd_sel_d1 <= 0;
      bank_rd_sel_d2 <= 0;
    end else begin
      rd_en_d1 <= rd_en;
      rd_en_d2 <= rd_en_d1;
      
      rd_valid <= rd_en_d2;
      
      bank_rd_sel_d1 <= bank_rd_sel;
      bank_rd_sel_d2 <= bank_rd_sel_d1;
      if (rd_en_d2) begin
        rd_data <= (bank_rd_sel_d2 == 1'b0) ? dout_b0 : dout_b1;
      end
    end
  end

endmodule