`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:38:35
// Design Name: 
// Module Name: prefetch_double_buffer
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

module prefetch_double_buffer #(
  parameter integer OUT_W  = 112,
  parameter integer OUT_H  = 112,
  parameter integer TILE_H = 6,
  parameter integer DATA_W = 8,
  parameter integer LANES  = 16
)(
  input  wire clk,
  input  wire rst_n,
  input  wire prefetch_start,
  input  wire [$clog2(OUT_H)-1:0] tile_row,

  // external memory interface (feature buffer read side)
  output reg  mem_en,
  output reg  [$clog2(OUT_W*OUT_H*(32/LANES))-1:0] mem_addr,
  input  wire [LANES*DATA_W-1:0] mem_dout,
  input  wire mem_valid,

  // consumer interface
  input  wire read_enable,
  input  wire [$clog2(OUT_W)-1:0] read_addr,
  output reg  [TILE_H*DATA_W*LANES-1:0] buffer_out,
  output reg  buffer_ready,
  output wire prefetch_busy,
  output wire prefetch_done
);

  localparam integer WORD_W   = DATA_W * LANES; // 128
  localparam integer PADDING  = 1;
  localparam integer PADDED_W = OUT_W + 2 * PADDING;

  localparam integer ADDR_W   = $clog2(OUT_W*OUT_H*(32/LANES));
  localparam integer BUF_AW   = $clog2(PADDED_W);

  // ---------------- control regs ----------------
  reg active_read_buf;
  reg active_write_buf;
  reg write_active;

  reg [BUF_AW-1:0]         write_col;
  reg [$clog2(TILE_H)-1:0] write_row;

  reg [$clog2(OUT_H)-1:0]  fetch_row;
  reg [$clog2(OUT_H)-1:0]  base_tile_row;

  reg prefetch_done_reg;
  reg just_completed_buf;
  reg switch_pending;
  reg switch_pending_d;

  assign prefetch_busy = write_active;
  assign prefetch_done = prefetch_done_reg;

  // ---------------- address helper (keep your original mapping) ----------------
  function [ADDR_W-1:0] calc_addr;
    input [$clog2(OUT_H)-1:0] row;
    input [BUF_AW-1:0]        col;
    reg [31:0] temp_addr;
    begin
      temp_addr = row * OUT_W + col;
      calc_addr = temp_addr[ADDR_W-1:0];
    end
  endfunction

  // ---------------- Row RAMs (A and B) ----------------
  // For each buffer, we have TILE_H memories:
  //   depth = PADDED_W
  //   width = WORD_W (128)
  wire [WORD_W-1:0] rowA_dout [0:TILE_H-1];
  wire [WORD_W-1:0] rowB_dout [0:TILE_H-1];

  // Address to RAMs: during write use write_col; during read use read_addr.
  wire [BUF_AW-1:0] ram_addr = write_active ? write_col : read_addr[BUF_AW-1:0];

  // Enable RAM: write uses mem_valid; read uses read_enable & buffer_ready
  wire ram_en_write = write_active && mem_valid;
  wire ram_en_read  = (read_enable && buffer_ready && (read_addr < PADDED_W));
  wire ram_ena      = write_active ? ram_en_write : ram_en_read;

  genvar r;
  generate
    for (r = 0; r < TILE_H; r = r + 1) begin : GEN_BUFA_ROWS
      xpm_memory_spram #(
        .ADDR_WIDTH_A        (BUF_AW),
        .MEMORY_SIZE         (WORD_W * PADDED_W),
        .MEMORY_PRIMITIVE    ("block"),
        .WRITE_DATA_WIDTH_A  (WORD_W),
        .READ_DATA_WIDTH_A   (WORD_W),
        .READ_LATENCY_A      (1),
        .USE_MEM_INIT        (0),
        .ECC_MODE            ("no_ecc"),
        .WAKEUP_TIME         ("disable_sleep"),
        .AUTO_SLEEP_TIME     (0),
        .MESSAGE_CONTROL     (0)
      ) u_bufA_row (
        .clka   (clk),
        .ena    (ram_ena),
        .wea    (ram_en_write && (write_row == r[$clog2(TILE_H)-1:0]) && !active_write_buf),
        .addra  (ram_addr),
        .dina   (mem_dout),
        .douta  (rowA_dout[r]),
        .regcea (1'b1),
        .rsta   (1'b0),
        .sleep  (1'b0)
      );
    end
  endgenerate

  generate
    for (r = 0; r < TILE_H; r = r + 1) begin : GEN_BUFB_ROWS
      xpm_memory_spram #(
        .ADDR_WIDTH_A        (BUF_AW),
        .MEMORY_SIZE         (WORD_W * PADDED_W),
        .MEMORY_PRIMITIVE    ("block"),
        .WRITE_DATA_WIDTH_A  (WORD_W),
        .READ_DATA_WIDTH_A   (WORD_W),
        .READ_LATENCY_A      (1),
        .USE_MEM_INIT        (0),
        .ECC_MODE            ("no_ecc"),
        .WAKEUP_TIME         ("disable_sleep"),
        .AUTO_SLEEP_TIME     (0),
        .MESSAGE_CONTROL     (0)
      ) u_bufB_row (
        .clka   (clk),
        .ena    (ram_ena),
        .wea    (ram_en_write && (write_row == r[$clog2(TILE_H)-1:0]) &&  active_write_buf),
        .addra  (ram_addr),
        .dina   (mem_dout),
        .douta  (rowB_dout[r]),
        .regcea (1'b1),
        .rsta   (1'b0),
        .sleep  (1'b0)
      );
    end
  endgenerate

  // ---------------- Prefetch FSM ----------------
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      write_active       <= 1'b0;
      write_col          <= {BUF_AW{1'b0}};
      write_row          <= {($clog2(TILE_H)){1'b0}};
      fetch_row          <= {($clog2(OUT_H)){1'b0}};
      base_tile_row      <= {($clog2(OUT_H)){1'b0}};
      prefetch_done_reg  <= 1'b0;
      active_write_buf   <= 1'b0;
      just_completed_buf <= 1'b0;
      switch_pending     <= 1'b0;
      mem_en             <= 1'b0;
      mem_addr           <= {ADDR_W{1'b0}};
    end else begin
      mem_en            <= 1'b0;
      prefetch_done_reg <= 1'b0;

      // start prefetch
      if (prefetch_start && !write_active) begin
        write_active  <= 1'b1;
        write_col     <= {BUF_AW{1'b0}};
        write_row     <= {($clog2(TILE_H)){1'b0}};
        base_tile_row <= tile_row;
        fetch_row     <= tile_row;

        mem_en   <= 1'b1;
        mem_addr <= calc_addr(tile_row, {BUF_AW{1'b0}});
      end
      // active fetching
      else if (write_active) begin
        if (mem_valid) begin
          // Writes happen inside XPM RAMs via wea for selected row.

          if (write_row < TILE_H - 1) begin
            write_row <= write_row + 1;
            fetch_row <= fetch_row + 1;

            mem_en   <= 1'b1;
            mem_addr <= calc_addr(fetch_row + 1, write_col);
          end else begin
            write_row <= {($clog2(TILE_H)){1'b0}};
            fetch_row <= base_tile_row;

            if (write_col < PADDED_W - 1) begin
              write_col <= write_col + 1;

              mem_en   <= 1'b1;
              mem_addr <= calc_addr(base_tile_row, write_col + 1);
            end else begin
              // completed tile
              write_active       <= 1'b0;
              prefetch_done_reg  <= 1'b1;
              write_col          <= {BUF_AW{1'b0}};
              just_completed_buf <= active_write_buf;
              active_write_buf   <= ~active_write_buf;
              switch_pending     <= 1'b1;
              mem_addr           <= {ADDR_W{1'b0}};
            end
          end
        end
      end
      else begin
        mem_addr <= {ADDR_W{1'b0}};
      end

      if (switch_pending_d)
        switch_pending <= 1'b0;
    end
  end

  // ---------------- Buffer ready + read buffer switch ----------------
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      active_read_buf  <= 1'b0;
      buffer_ready     <= 1'b0;
      switch_pending_d <= 1'b0;
    end else begin
      switch_pending_d <= switch_pending;
      if (switch_pending_d) begin
        active_read_buf <= just_completed_buf;
        buffer_ready    <= 1'b0;
      end else begin
        if (!write_active && !switch_pending)
          buffer_ready <= 1'b1;
      end
    end
  end

  // ---------------- Output register (accounts for RAM read latency = 1) ----------------
  reg read_en_d1;
  reg active_read_buf_d1;
  integer i;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      buffer_out          <= {TILE_H*WORD_W{1'b0}};
      read_en_d1          <= 1'b0;
      active_read_buf_d1  <= 1'b0;
    end else begin
      read_en_d1         <= ram_en_read;       // read request accepted
      active_read_buf_d1 <= active_read_buf;   // align with data

      if (read_en_d1) begin
        for (i = 0; i < TILE_H; i = i + 1) begin
          buffer_out[i*WORD_W +: WORD_W] <=
            active_read_buf_d1 ? rowB_dout[i] : rowA_dout[i];
        end
      end else begin
        buffer_out <= {TILE_H*WORD_W{1'b0}};
      end
    end
  end

endmodule


