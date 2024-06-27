func.func @main(
    %query : tensor<1x4096x64xf32>,
    %key : tensor<1x4096x64xf32>,
    %value : tensor<1x4096x64xf32>,
    %scale : tensor<f32>) -> tensor<1x4096x64xf32> {

    %scalef32 = tensor.extract %scale[] : tensor<f32>

    %empty = tensor.empty() : tensor<1x4096x64xf32>
    %c0 = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<1x4096x64xf32>)  -> tensor<1x4096x64xf32>
    %atten = iree_linalg_ext.attention ins(%query, %key, %value, %scalef32 : tensor<1x4096x64xf32>, tensor<1x4096x64xf32>, tensor<1x4096x64xf32>, f32) outs(%fill : tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    return %atten : tensor<1x4096x64xf32>
}
