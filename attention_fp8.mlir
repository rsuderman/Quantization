
#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func private @scale(%arg0 : tensor<1x4096x64xf32>, %arg1 : f32) -> tensor<1x4096x64xf32> {
    %empty = tensor.empty() : tensor<1x4096x64xf32>
    %generic = linalg.generic  {
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel", "parallel", "parallel"]}      
      ins(%arg0 : tensor<1x4096x64xf32>)
      outs(%empty : tensor<1x4096x64xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
        %0 = arith.mulf %b0, %arg1 : f32
        linalg.yield %0 : f32
      } -> tensor<1x4096x64xf32>

    return %generic : tensor<1x4096x64xf32>
}

func.func @main(
    %query : tensor<1x4096x64xf32>,
    %key : tensor<1x4096x64xf32>,
    %value : tensor<1x4096x64xf32>,
    %scale : tensor<f32>,
    %qscale : tensor<f32>,
    %kscale : tensor<f32>,
    %vscale : tensor<f32>) -> tensor<1x4096x64xf32> {

    %scalef32 = tensor.extract %scale[] : tensor<f32>
    %qscalef32 = tensor.extract %qscale[] : tensor<f32>
    %kscalef32 = tensor.extract %kscale[] : tensor<f32>
    %vscalef32 = tensor.extract %vscale[] : tensor<f32>

    %q = call @scale(%query, %qscalef32) : (tensor<1x4096x64xf32>, f32) -> tensor<1x4096x64xf32>
    %k = call @scale(%key, %kscalef32) : (tensor<1x4096x64xf32>, f32) -> tensor<1x4096x64xf32>
    %v = call @scale(%value, %vscalef32) : (tensor<1x4096x64xf32>, f32) -> tensor<1x4096x64xf32>

    %empty = tensor.empty() : tensor<1x4096x64xf32>
    %c0 = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<1x4096x64xf32>)  -> tensor<1x4096x64xf32>
    %atten = iree_linalg_ext.attention ins(%q, %k, %v, %scalef32 : tensor<1x4096x64xf32>, tensor<1x4096x64xf32>, tensor<1x4096x64xf32>, f32) outs(%fill : tensor<1x4096x64xf32>) -> tensor<1x4096x64xf32>
    return %atten : tensor<1x4096x64xf32>
}
