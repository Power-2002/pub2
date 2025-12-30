`ifndef MOBILENET_DEFINES_VH
`define MOBILENET_DEFINES_VH

// ============================================================
// MobileNetV1 全局参数定义
// ============================================================

// 数据位宽
`define DATA_W          8
`define ACC_W           32
`define PSUM_W          18
`define PROD_W          16

// PE阵列配置
`define NUM_ROWS        16
`define NUM_COLS        16
`define UNIT_NUM        16
`define LANES           16

// MobileNetV1 层数配置
`define TOTAL_LAYERS    28
`define TOTAL_CONV_DW_PW_LAYERS 27  // 不含FC

// 最大尺寸配置
`define MAX_IMG_W       224
`define MAX_IMG_H       224
`define MAX_CHANNELS    1024
`define MAX_OUT_W       112
`define MAX_OUT_H       112

// 特征图缓冲配置 (112*112*64 / 16 = 50176 words)
`define FEATURE_BUF_DEPTH   50176

// 权重存储基地址 (Layer0从0开始, Layer2从64开始)
`define L0_WEIGHT_BASE      9'd0
`define L2_WEIGHT_BASE      9'd64

// 数据路径 (根据实际环境修改)
`define DATA_PATH           "D:/NoC/mycode/mobilenet_acc2/data/"

// ============ Layer Type Definitions (3-bit) ============
localparam [2:0] LAYER_TYPE_CONV = 3'd0;  // 第一层标准3x3卷积
localparam [2:0] LAYER_TYPE_DW   = 3'd1;  // Depthwise Convolution
localparam [2:0] LAYER_TYPE_PW   = 3'd2;  // Pointwise Convolution
localparam [2:0] LAYER_TYPE_AP   = 3'd3;  // Global Average Pooling
localparam [2:0] LAYER_TYPE_FC   = 3'd4;  // Fully Connected

// ============ MobileNetV1 Layer Count ============
localparam integer TOTAL_LAYERS = 29;  // Layer 0-28 (共29层)
localparam integer MAX_LAYER_ID = 28;  // 最后一层ID

// ============ Data Width ============
localparam integer DATA_W    = 8;      // INT8
localparam integer ACC_W     = 32;     // 累加器位宽
localparam integer WEIGHT_W  = 8;      // 权重位宽

// ============ PE Array Size ============
localparam integer PE_ROWS   = 16;
localparam integer PE_COLS   = 16;

// ============ Memory Parameters ============
localparam integer WEIGHT_ADDR_W = 16;
localparam integer BIAS_ADDR_W   = 12;
localparam integer FEAT_ADDR_W   = 17;

`endif