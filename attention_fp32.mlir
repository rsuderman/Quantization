#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
module @module {
  func.func @main(%arg0: !torch.vtensor<[1,1,4096,64],f32>, %arg1: !torch.vtensor<[1,1,4096,64],f32>, %arg2: !torch.vtensor<[1,1,4096,64],f32>, %arg3: !torch.vtensor<[],f32>, %arg4: !torch.vtensor<[],f32>, %arg5: !torch.vtensor<[],f32>, %arg6: !torch.vtensor<[],f32>) -> !torch.vtensor<[1,1,4096,64],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.Float.Tensor %arg4 : !torch.vtensor<[],f32> -> !torch.float
    %int5 = torch.constant.int 5
    %none = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %1 = torch.aten.scalar_tensor %0, %int5, %none, %cpu, %false : !torch.float, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[],f32>
    %2 = torch.aten.Float.Tensor %arg5 : !torch.vtensor<[],f32> -> !torch.float
    %int5_0 = torch.constant.int 5
    %none_1 = torch.constant.none
    %cpu_2 = torch.constant.device "cpu"
    %false_3 = torch.constant.bool false
    %3 = torch.aten.scalar_tensor %2, %int5_0, %none_1, %cpu_2, %false_3 : !torch.float, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[],f32>
    %4 = torch.aten.Float.Tensor %arg6 : !torch.vtensor<[],f32> -> !torch.float
    %int5_4 = torch.constant.int 5
    %none_5 = torch.constant.none
    %cpu_6 = torch.constant.device "cpu"
    %false_7 = torch.constant.bool false
    %5 = torch.aten.scalar_tensor %4, %int5_4, %none_5, %cpu_6, %false_7 : !torch.float, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[],f32>
    %float1.250000e-01 = torch.constant.float 1.250000e-01
    %int6 = torch.constant.int 6
    %none_8 = torch.constant.none
    %cpu_9 = torch.constant.device "cpu"
    %false_10 = torch.constant.bool false
    %6 = torch.aten.scalar_tensor %float1.250000e-01, %int6, %none_8, %cpu_9, %false_10 : !torch.float, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[],f32>
    %7 = torch.aten.mul.Tensor %6, %1 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %8 = torch.aten.mul.Tensor %7, %3 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %int5_11 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %arg0, %int5_11 : !torch.vtensor<[1,1,4096,64],f32>, !torch.int -> !torch.vtensor<[1,1,4096,64],f32>
    %int5_12 = torch.constant.int 5
    %10 = torch.prims.convert_element_type %arg1, %int5_12 : !torch.vtensor<[1,1,4096,64],f32>, !torch.int -> !torch.vtensor<[1,1,4096,64],f32>
    %int5_13 = torch.constant.int 5
    %11 = torch.prims.convert_element_type %arg2, %int5_13 : !torch.vtensor<[1,1,4096,64],f32>, !torch.int -> !torch.vtensor<[1,1,4096,64],f32>
    %12 = torch_c.to_builtin_tensor %9 : !torch.vtensor<[1,1,4096,64],f32> -> tensor<1x1x4096x64xf32>
    %cast = tensor.cast %12 : tensor<1x1x4096x64xf32> to tensor<?x?x4096x64xf32>
    %13 = torch_c.to_builtin_tensor %10 : !torch.vtensor<[1,1,4096,64],f32> -> tensor<1x1x4096x64xf32>
    %cast_14 = tensor.cast %13 : tensor<1x1x4096x64xf32> to tensor<?x?x4096x64xf32>
    %14 = torch_c.to_builtin_tensor %11 : !torch.vtensor<[1,1,4096,64],f32> -> tensor<1x1x4096x64xf32>
    %cast_15 = tensor.cast %14 : tensor<1x1x4096x64xf32> to tensor<?x?x4096x64xf32>
    %15 = torch_c.to_builtin_tensor %8 : !torch.vtensor<[],f32> -> tensor<f32>
    %16 = util.call @sharktank_flash_attention_4096_4096_64_64_f32_f32_f32(%cast, %cast_14, %cast_15, %15) : (tensor<?x?x4096x64xf32>, tensor<?x?x4096x64xf32>, tensor<?x?x4096x64xf32>, tensor<f32>) -> tensor<?x?x4096x64xf32>
    %cast_16 = tensor.cast %16 : tensor<?x?x4096x64xf32> to tensor<1x1x4096x64xf32>
    %17 = torch_c.from_builtin_tensor %cast_16 : tensor<1x1x4096x64xf32> -> !torch.vtensor<[1,1,4096,64],f32>
    %18 = torch.aten.mul.Tensor %17, %5 : !torch.vtensor<[1,1,4096,64],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[1,1,4096,64],f32>
    return %18 : !torch.vtensor<[1,1,4096,64],f32>
  }
  util.func private @sharktank_flash_attention_4096_4096_64_64_f32_f32_f32(%arg0: tensor<?x?x4096x64xf32>, %arg1: tensor<?x?x4096x64xf32>, %arg2: tensor<?x?x4096x64xf32>, %arg3: tensor<f32>) -> tensor<?x?x4096x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x4096x64xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x4096x64xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<?x?x4096x64xf32>
    %dim_2 = tensor.dim %arg2, %c3 : tensor<?x?x4096x64xf32>
    %extracted = tensor.extract %arg3[] : tensor<f32>
    %0 = tensor.empty(%dim, %dim_0, %dim_1, %dim_2) : tensor<?x?x?x?xf32>
    %cast = tensor.cast %0 : tensor<?x?x?x?xf32> to tensor<?x?x4096x64xf32>
    %1 = iree_linalg_ext.attention {indexing_maps = [#map, #map1, #map2, #map3]} ins(%arg0, %arg1, %arg2, %extracted : tensor<?x?x4096x64xf32>, tensor<?x?x4096x64xf32>, tensor<?x?x4096x64xf32>, f32) outs(%cast : tensor<?x?x4096x64xf32>) -> tensor<?x?x4096x64xf32>
    util.return %1 : tensor<?x?x4096x64xf32>
  }
}
