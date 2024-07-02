
#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func private @scale(%arg0 : tensor<1x4096x64xf16>, %arg1 : f16) -> tensor<1x4096x64xf16> {
    %empty = tensor.empty() : tensor<1x4096x64xf16>
    %generic = linalg.generic  {
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel", "parallel", "parallel"]}      
      ins(%arg0 : tensor<1x4096x64xf16>)
      outs(%empty : tensor<1x4096x64xf16>) {
    ^bb0(%b0 : f16, %b1 : f16):
        %0 = arith.mulf %b0, %arg1 : f16
        linalg.yield %0 : f16
      } -> tensor<1x4096x64xf16>

    return %generic : tensor<1x4096x64xf16>
}

func.func @main(
    %q : tensor<1x4096x64xf16>,
    %k : tensor<1x4096x64xf16>,
    %v : tensor<1x4096x64xf16>) -> tensor<1x4096x64xf16> {

    %scalef16  = arith.constant 1.0 : f16
    %qscalef16 = arith.constant 1.0 : f16
    %kscalef16 = arith.constant 1.0 : f16
    %vscalef16 = arith.constant 1.0 : f16

    %qk = arith.mulf %qscalef16, %kscalef16 : f16
    %qks = arith.mulf %qk, %scalef16 : f16

    %empty = tensor.empty() : tensor<1x4096x64xf16>
    %c0 = arith.constant 0.0 : f16
    %fill = linalg.fill ins(%c0 : f16) outs(%empty : tensor<1x4096x64xf16>)  -> tensor<1x4096x64xf16>
    %atten = iree_linalg_ext.attention ins(%q, %k, %v, %qks : tensor<1x4096x64xf16>, tensor<1x4096x64xf16>, tensor<1x4096x64xf16>, f16) outs(%fill : tensor<1x4096x64xf16>) -> tensor<1x4096x64xf16>

    %atten_scale = call @scale(%atten, %vscalef16) : (tensor<1x4096x64xf16>, f16) -> tensor<1x4096x64xf16>

    return %atten_scale : tensor<1x4096x64xf16>
}
