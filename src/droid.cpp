#include <torch/extension.h> //pytorch c++ extension
#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> projective_transform_cuda(
  torch::Tensor poses,
  torch::Tensor disps, //再次确信就是视差 深度倒数？
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj);//这里声明的函数似乎没被调用？



torch::Tensor depth_filter_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh);


torch::Tensor frame_distance_cuda(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj,
  const float beta);

std::vector<torch::Tensor> projmap_cuda(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj);

torch::Tensor iproj_cuda(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics);
//ba 函数声明
std::vector<torch::Tensor> ba_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor disps_sens,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor ii,
    torch::Tensor jj,
    const int t0,
    const int t1,
    const int iterations,
    const float lm,
    const float ep,
    const bool motion_only);
//函数声明 corr.py
std::vector<torch::Tensor> corr_index_cuda_forward(
  torch::Tensor volume,
  torch::Tensor coords,
  int radius);

std::vector<torch::Tensor> corr_index_cuda_backward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius);

std::vector<torch::Tensor> altcorr_cuda_forward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  int radius);

std::vector<torch::Tensor> altcorr_cuda_backward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius);


#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

// 从python出来后 先到here
std::vector<torch::Tensor> ba(
    torch::Tensor poses, //(buffer=1000,7)
    torch::Tensor disps, //(,48,64)初始1
    torch::Tensor intrinsics, //(4)
    torch::Tensor disps_sens, //(,48,64)
    torch::Tensor targets, //(#e,2,48,64)
    torch::Tensor weights, //(#e,2,48,64)
    torch::Tensor eta, //damp (#v=12，48，64)
    torch::Tensor ii, // #edge
    torch::Tensor jj,// #edge
    const int t0, // 初始1
    const int t1, // 初始12 (warm up)
    const int iterations, //ba迭代次数 默认2
    const float lm, //1e-4
    const float ep, //0.1
    const bool motion_only) { //是否只优化位姿 默认false

  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(disps_sens);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);
  //继续调用
  return ba_cuda(poses, disps, intrinsics, disps_sens, targets, weights,
                 eta, ii, jj, t0, t1, iterations, lm, ep, motion_only);

}


torch::Tensor frame_distance(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj,
    const float beta) {

  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return frame_distance_cuda(poses, disps, intrinsics, ii, jj, beta);

}


std::vector<torch::Tensor> projmap(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj) {

  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return projmap_cuda(poses, disps, intrinsics, ii, jj);

}


torch::Tensor iproj(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics) {
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);

  return iproj_cuda(poses, disps, intrinsics);
}


// c++ python binding 从c++调用的函数 corr.py
std::vector<torch::Tensor> corr_index_forward(
    torch::Tensor volume,
    torch::Tensor coords,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);

  return corr_index_cuda_forward(volume, coords, radius);
}

std::vector<torch::Tensor> corr_index_backward(
    torch::Tensor volume,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

  auto volume_grad = corr_index_cuda_backward(volume, coords, corr_grad, radius);
  return {volume_grad};
}

std::vector<torch::Tensor> altcorr_forward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    int radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);

  return altcorr_cuda_forward(fmap1, fmap2, coords, radius);
}

std::vector<torch::Tensor> altcorr_backward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

  return altcorr_cuda_backward(fmap1, fmap2, coords, corr_grad, radius);
}


torch::Tensor depth_filter(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh) {

    CHECK_INPUT(poses);
    CHECK_INPUT(disps);
    CHECK_INPUT(intrinsics);
    CHECK_INPUT(ix);
    CHECK_INPUT(thresh);

    return depth_filter_cuda(poses, disps, intrinsics, ix, thresh);
}

//将python与C++11进行绑定， 注意这里def()的首个参数就是后来在python中可以引用的方法名）
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // bundle adjustment kernels 所有与ba有关的在cuda中的函数
  m.def("ba", &ba, "bundle adjustment");// update operator on factor graph 会被调用 depth_video.ba
  m.def("frame_distance", &frame_distance, "frame_distance");
  m.def("projmap", &projmap, "projmap");
  m.def("depth_filter", &depth_filter, "depth_filter");
  m.def("iproj", &iproj, "back projection");

  // correlation volume kernels
  m.def("altcorr_forward", &altcorr_forward, "ALTCORR forward");
  m.def("altcorr_backward", &altcorr_backward, "ALTCORR backward");
  m.def("corr_index_forward", &corr_index_forward, "INDEX forward");
  m.def("corr_index_backward", &corr_index_backward, "INDEX backward");
}