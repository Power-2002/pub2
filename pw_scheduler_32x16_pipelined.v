`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/22 19:20:45
// Design Name: 
// Module Name: pw_scheduler_32x16_pipelined
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
module pw_scheduler_32x16_pipelined #(
    parameter NUM_ROWS = 32,
    parameter NUM_COLS = 16,
    parameter A_BITS   = 8,
    parameter W_BITS   = 8,
    parameter ACC_BITS = 32,
    parameter ADDR_W   = 19,

    parameter integer FAST_SIM_EN = 0,
    parameter integer FAST_COUT_SUBSAMPLE = 16,
    parameter integer FAST_PX_SUBSAMPLE   = 16,

  
    parameter integer FAST_SHIFT = 12,

    // FAST_SIM prefetch ? cycle ? done
    parameter integer PREFETCH_TIMEOUT = 5000
)(
    input  wire CLK,
    input  wire RESET,   // ? RESET  active-low!RESET
    input  wire start,
    output reg  done,

    input  wire [10:0] cin,
    input  wire [10:0] cout,
    input  wire [7:0]  img_w,
    input  wire [7:0]  img_h,
    input  wire [ADDR_W-1:0] w_base_in,

    // ???
    output reg  weight_req,
    input  wire weight_grant,
    output reg  [ADDR_W-1:0] weight_base,
    output reg  [10:0] weight_count,
    input  wire weight_valid,
    input  wire [127:0] weight_data,
    input  wire weight_done,

    // ?
    output reg  feat_rd_en,
    output reg  [15:0] feat_rd_addr,
    input  wire [127:0] feat_rd_data,
    input  wire feat_rd_valid,

    // PE§ß?
    output reg  arr_W_EN,
    output reg  [NUM_COLS*W_BITS-1:0] in_weight_above,
    output reg  [NUM_ROWS*A_BITS-1:0] active_left,
    input  wire [NUM_COLS*ACC_BITS-1:0] out_sum_final,

    output reg  y_valid,
    output reg  [NUM_COLS*ACC_BITS-1:0] y_data,
    output reg  y_tile_sel
);

    // ============================================================
    // ?
    // ============================================================
    wire [10:0] cin_tiles  = (cin + 31) >> 5;   // ?tile 32ch
    wire [10:0] cout_tiles = (cout + 15) >> 4;  // ?tile 16ch
    wire [15:0] total_px   = img_w * img_h;

    localparam integer PE_LAT = NUM_ROWS - 1;  // 31 cycles

    // ============================================================
    // workload ?¨°???
    // ============================================================
    reg [31:0] work_left;
    wire [31:0] work_full = total_px * cin_tiles * cout_tiles;

    // ? ??
    // -  FAST_SHIFT != 0 shift
    // -  FAST_SHIFT=0? subsample 
    wire [31:0] work_scaled_shift = (FAST_SHIFT != 0) ? (work_full >> FAST_SHIFT) : work_full;

    wire [31:0] subsample_div =
        (FAST_COUT_SUBSAMPLE < 2) ? 32'd1 :
        (FAST_PX_SUBSAMPLE   < 2) ? 32'd1 :
        (FAST_COUT_SUBSAMPLE * FAST_PX_SUBSAMPLE);

    wire [31:0] work_scaled =
        (FAST_SHIFT != 0) ? work_scaled_shift :
        (work_full / subsample_div);

    wire [31:0] work_init = (work_scaled < 32'd16) ? 32'd16 : work_scaled;

    // ============================================================
    // ?
    // ============================================================
    reg [255:0] act_buf [0:1];
    reg act_buf_valid [0:1];
    reg act_buf_sel;
    reg act_load_sel;

    reg [127:0] weight_buf [0:1][0:3];
    reg [2:0] weight_buf_cnt [0:1];
    reg weight_buf_valid [0:1];
    reg weight_buf_sel;
    reg weight_load_sel;

    // ============================================================
    // pipeline stage ??
    // ============================================================
    reg load_active;
    reg [1:0] load_act_phase;
    reg [127:0] load_act_low;

    reg load_weight_active;

    reg pe_active;
    reg [5:0] pe_cycle_cnt;

    reg capture_active;
    reg [4:0] capture_col;


    // ============================================================
    reg [15:0] px_idx;
    reg [10:0] cin_idx;
    reg [10:0] cout_idx;

    wire [15:0] px_next   = (px_idx + 1 < total_px)     ? (px_idx + 1) : 16'd0;
    wire [10:0] cin_next  = (cin_idx + 1 < cin_tiles)   ? (cin_idx + 1) : 11'd0;
    wire [10:0] cout_next = (cout_idx + 1 < cout_tiles) ? (cout_idx + 1) : 11'd0;

    // ============================================================
    // accum
    // ============================================================
    reg signed [ACC_BITS-1:0] psum [0:NUM_COLS-1];

    // ============================================================
    // FSM
    // ============================================================
    localparam S_IDLE     = 3'd0;
    localparam S_PREFETCH = 3'd1;
    localparam S_COMPUTE  = 3'd2;
    localparam S_OUTPUT   = 3'd3;
    localparam S_DONE     = 3'd4;

    reg [2:0] state;

    // prefetch timeout counter
    reg [31:0] prefetch_wait_cnt;

    integer i;

    // ?
    wire [15:0] act_addr_base =
        (px_idx * cin_tiles * 2) + (cin_idx * 2);

    wire [ADDR_W-1:0] weight_addr_base =
        w_base_in + (cout_idx * cin_tiles + cin_idx) * 4;

    // ============================================================
    // ?
    // ============================================================
    always @(posedge CLK or negedge RESET) begin
        if (!RESET) begin
            state <= S_IDLE;
            done <= 0;
            y_valid <= 0;
            arr_W_EN <= 0;
            weight_req <= 0;
            feat_rd_en <= 0;

            load_active <= 0;
            load_act_phase <= 0;
            load_weight_active <= 0;

            pe_active <= 0;
            pe_cycle_cnt <= 0;

            capture_active <= 0;
            capture_col <= 0;

            px_idx <= 0;
            cin_idx <= 0;
            cout_idx <= 0;

            act_buf_sel <= 0;
            act_load_sel <= 0;
            weight_buf_sel <= 0;
            weight_load_sel <= 0;

            work_left <= 0;
            prefetch_wait_cnt <= 0;

            for (i = 0; i < NUM_COLS; i = i + 1) psum[i] <= 0;
            for (i = 0; i < 2; i = i + 1) begin
                act_buf_valid[i] <= 0;
                weight_buf_valid[i] <= 0;
                weight_buf_cnt[i] <= 0;
            end
        end else begin
            done <= 0;
            y_valid <= 0;
            arr_W_EN <= 0;

            // ====================================================
            // Stage1: 
            // ====================================================
            if (load_active) begin
                case (load_act_phase)
                    2'd0: begin
                        feat_rd_en <= 1;
                        feat_rd_addr <= act_addr_base;
                        load_act_phase <= 2'd1;
                    end
                    2'd1: begin
                        feat_rd_en <= 0;
                        if (feat_rd_valid) begin
                            load_act_low <= feat_rd_data;
                            feat_rd_en <= 1;
                            feat_rd_addr <= act_addr_base + 16'd1;
                            load_act_phase <= 2'd2;
                        end
                    end
                    2'd2: begin
                        feat_rd_en <= 0;
                        if (feat_rd_valid) begin
                            act_buf[act_load_sel] <= {feat_rd_data, load_act_low};
                            act_buf_valid[act_load_sel] <= 1;
                            load_act_phase <= 2'd0;
                            load_active <= 0;
                        end
                    end
                endcase
            end

            // ====================================================
            // Stage1: ??
            // ====================================================
            if (load_weight_active) begin
                if (weight_buf_cnt[weight_load_sel] == 0 && !weight_req) begin
                    weight_req <= 1;
                    weight_base <= weight_addr_base;
                    weight_count <= 11'd4;
                end

                if (weight_grant) weight_req <= 0;

                if (weight_valid) begin
                    weight_buf[weight_load_sel][weight_buf_cnt[weight_load_sel]] <= weight_data;
                    weight_buf_cnt[weight_load_sel] <= weight_buf_cnt[weight_load_sel] + 1;
                end

                if (weight_done) begin
                    weight_buf_valid[weight_load_sel] <= 1;
                    load_weight_active <= 0;
                end
            end

            // ====================================================
            // Stage2: PE compute
            // ====================================================
            if (pe_active) begin
                if (pe_cycle_cnt == 0) begin
    active_left     <= act_buf[act_buf_sel];
    in_weight_above <= weight_buf[weight_buf_sel][0];
                    arr_W_EN <= 1;
                end

                pe_cycle_cnt <= pe_cycle_cnt + 1;

                if (pe_cycle_cnt >= PE_LAT) begin
                    pe_active <= 0;
                    capture_active <= 1;
                    capture_col <= 0;

                    act_buf_valid[act_buf_sel] <= 0;
                    weight_buf_valid[weight_buf_sel] <= 0;
                    weight_buf_cnt[weight_buf_sel] <= 0;

                    act_buf_sel <= ~act_buf_sel;
                    weight_buf_sel <= ~weight_buf_sel;
                end
            end

            // ====================================================
            // Stage3: ?
            // ====================================================
            if (capture_active) begin
                psum[capture_col] <= psum[capture_col] +
                    $signed(out_sum_final[capture_col*ACC_BITS +: ACC_BITS]);
                capture_col <= capture_col + 1;

                if (capture_col == NUM_COLS-1) begin
                    capture_active <= 0;
                    if (work_left != 0)
                        work_left <= work_left - 1;
                end
            end

            // ====================================================
            // FSM
            // ====================================================
            case (state)
                S_IDLE: begin
                    prefetch_wait_cnt <= 0;

                    if (start) begin
                        px_idx <= 0;
                        cin_idx <= 0;
                        cout_idx <= 0;

                        work_left <= (FAST_SIM_EN != 0) ? work_init : work_full;

                        for (i=0;i<NUM_COLS;i=i+1) psum[i] <= 0;

                        act_buf_valid[0] <= 0;
                        act_buf_valid[1] <= 0;
                        weight_buf_valid[0] <= 0;
                        weight_buf_valid[1] <= 0;
                        weight_buf_cnt[0] <= 0;
                        weight_buf_cnt[1] <= 0;

                        act_buf_sel <= 0;
                        weight_buf_sel <= 0;

                        act_load_sel <= 0;
                        weight_load_sel <= 0;

                        load_active <= 1;
                        load_act_phase <= 0;

                        load_weight_active <= 1;
                        weight_req <= 0;

                        state <= S_PREFETCH;
                    end
                end

                S_PREFETCH: begin
                    if (FAST_SIM_EN != 0) begin
                        prefetch_wait_cnt <= prefetch_wait_cnt + 1;
                        if (prefetch_wait_cnt > PREFETCH_TIMEOUT) begin
                            state <= S_OUTPUT;
                        end
                    end

                    if (act_buf_valid[act_buf_sel] && weight_buf_valid[weight_buf_sel]) begin
                        prefetch_wait_cnt <= 0;

                        pe_active <= 1;
                        pe_cycle_cnt <= 0;

                        state <= S_COMPUTE;
                    end
                end

                S_COMPUTE: begin
                    if (!pe_active && !capture_active) begin
                        if (work_left == 0) begin
                            state <= S_OUTPUT;
                        end else begin
                            px_idx   <= px_next;
                            cin_idx  <= cin_next;
                            cout_idx <= cout_next;

                            act_load_sel    <= act_buf_sel;
                            weight_load_sel <= weight_buf_sel;

                            load_active <= 1;
                            load_act_phase <= 0;

                            load_weight_active <= 1;

                            state <= S_PREFETCH;
                        end
                    end
                end

                S_OUTPUT: begin
                    y_valid <= 1;
                    for (i=0;i<NUM_COLS;i=i+1)
                        y_data[i*ACC_BITS +: ACC_BITS] <= psum[i];
                    y_tile_sel <= cout_idx[0];

                    state <= S_DONE;
                end

                S_DONE: begin
                    done <= 1;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
