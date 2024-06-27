
func.func @main(%arg0 : tensor<128x128xf16>, %arg1 : tensor<128x128xf16>) -> tensor<128x128xf32> {
    %arg0_f8 = arith.truncf %arg0 : tensor<128x128xf16> to tensor<128x128xf8E4M3FNUZ>
    %arg1_f8 = arith.truncf %arg1 : tensor<128x128xf16> to tensor<128x128xf8E4M3FNUZ>

    %empty = tensor.empty() : tensor<128x128xf32>
    %c0 = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<128x128xf32>)  -> tensor<128x128xf32>
    %mm = linalg.matmul ins(%arg0_f8, %arg1_f8 : tensor<128x128xf8E4M3FNUZ>, tensor<128x128xf8E4M3FNUZ>) outs(%fill : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %mm : tensor<128x128xf32>
}
