#pragma once

#include <tuple>

#include "paddle/include/experimental/phi/api/include/tensor.h"
#include "paddle/include/experimental/phi/common/scalar.h"
#include "paddle/include/experimental/phi/common/int_array.h"
#include "paddle/include/experimental/utils/optional.h"

namespace paddle {
namespace experimental {


PADDLE_API Tensor acos(const Tensor& x);

PADDLE_API Tensor acosh(const Tensor& x);

PADDLE_API Tensor angle(const Tensor& x);

PADDLE_API std::tuple<Tensor, Tensor> argsort(const Tensor& x, int axis = -1, bool descending = false);

PADDLE_API Tensor asin(const Tensor& x);

PADDLE_API Tensor asinh(const Tensor& x);

PADDLE_API Tensor atan(const Tensor& x);

PADDLE_API Tensor atan2(const Tensor& x, const Tensor& y);

PADDLE_API Tensor atanh(const Tensor& x);

PADDLE_API Tensor bernoulli(const Tensor& x);

PADDLE_API Tensor bmm(const Tensor& x, const Tensor& y);

PADDLE_API Tensor ceil(const Tensor& x);

PADDLE_API Tensor& ceil_(Tensor& x);

PADDLE_API Tensor cholesky(const Tensor& x, bool upper = false);

PADDLE_API Tensor cholesky_solve(const Tensor& x, const Tensor& y, bool upper = false);

PADDLE_API Tensor cos(const Tensor& x);

PADDLE_API Tensor cosh(const Tensor& x);

PADDLE_API Tensor cross(const Tensor& x, const Tensor& y, int axis = 9);

PADDLE_API Tensor det(const Tensor& x);

PADDLE_API Tensor diag(const Tensor& x, int offset = 0, float padding_value = 0.0);

PADDLE_API Tensor diagonal(const Tensor& x, int offset = 0, int axis1 = 0, int axis2 = 1);

PADDLE_API Tensor digamma(const Tensor& x);

PADDLE_API Tensor dist(const Tensor& x, const Tensor& y, float p = 2.0);

PADDLE_API Tensor dot(const Tensor& x, const Tensor& y);

PADDLE_API Tensor erf(const Tensor& x);

PADDLE_API Tensor erfinv(const Tensor& x);

PADDLE_API Tensor& erfinv_(Tensor& x);

PADDLE_API Tensor exp(const Tensor& x);

PADDLE_API Tensor& exp_(Tensor& x);

PADDLE_API Tensor expm1(const Tensor& x);

PADDLE_API Tensor fft_c2c(const Tensor& x, const std::vector<int64_t>& axes, const std::string& normalization, bool forward);

PADDLE_API Tensor fft_c2r(const Tensor& x, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, int64_t last_dim_size = 0L);

PADDLE_API Tensor fft_r2c(const Tensor& x, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, bool onesided);

PADDLE_API Tensor flip(const Tensor& x, const std::vector<int>& axis);

PADDLE_API Tensor floor(const Tensor& x);

PADDLE_API Tensor& floor_(Tensor& x);

PADDLE_API Tensor hardshrink(const Tensor& x, float threshold = 0.5);

PADDLE_API Tensor hardsigmoid(const Tensor& x, float slope = 0.2, float offset = 0.5);

PADDLE_API Tensor lgamma(const Tensor& x);

PADDLE_API Tensor log10(const Tensor& x);

PADDLE_API Tensor log1p(const Tensor& x);

PADDLE_API Tensor log2(const Tensor& x);

PADDLE_API Tensor logit(const Tensor& x, float eps = 1e-6f);

PADDLE_API Tensor logsigmoid(const Tensor& x);

PADDLE_API Tensor mv(const Tensor& x, const Tensor& vec);

PADDLE_API Tensor poisson(const Tensor& x);

PADDLE_API Tensor reciprocal(const Tensor& x);

PADDLE_API Tensor& reciprocal_(Tensor& x);

PADDLE_API Tensor round(const Tensor& x);

PADDLE_API Tensor& round_(Tensor& x);

PADDLE_API Tensor send_uv(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const std::string& message_op = "ADD");

PADDLE_API Tensor silu(const Tensor& x);

PADDLE_API Tensor sin(const Tensor& x);

PADDLE_API Tensor sinh(const Tensor& x);

PADDLE_API Tensor solve(const Tensor& x, const Tensor& y);

PADDLE_API Tensor tan(const Tensor& x);

PADDLE_API Tensor tanh(const Tensor& x);

PADDLE_API Tensor& tanh_(Tensor& x);

PADDLE_API Tensor trace(const Tensor& x, int offset = 0, int axis1 = 0, int axis2 = 1);

PADDLE_API Tensor trunc(const Tensor& input);

PADDLE_API Tensor abs(const Tensor& x);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> accuracy(const Tensor& x, const Tensor& indices, const Tensor& label);

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&> adadelta_(Tensor& param, const Tensor& grad, Tensor& avg_squared_grad, Tensor& avg_squared_update, float rho, float epsilon);

PADDLE_API std::tuple<Tensor&, Tensor&> adagrad_(Tensor& param, const Tensor& grad, Tensor& moment, const Tensor& learning_rate, float epsilon);

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, paddle::optional<Tensor>&> adam_(Tensor& param, const Tensor& grad, const Tensor& learning_rate, Tensor& moment1, Tensor& moment2, Tensor& beta1_pow, Tensor& beta2_pow, paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, const Scalar& beta1, const Scalar& beta2, const Scalar& epsilon, bool lazy_mode, int64_t min_row_size_to_use_multithread, bool multi_precision, bool use_global_beta_pow);

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&> adamax_(Tensor& param, const Tensor& grad, const Tensor& learning_rate, Tensor& moment, Tensor& inf_norm, const Tensor& beta1_pow, float beta1, float beta2, float epsilon);

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, paddle::optional<Tensor>&> adamw_(Tensor& param, const Tensor& grad, const Tensor& learning_rate, Tensor& moment1, Tensor& moment2, Tensor& beta1_pow, Tensor& beta2_pow, paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, const Scalar& beta1, const Scalar& beta2, const Scalar& epsilon, float lr_ratio, float coeff, bool with_decay, bool lazy_mode, int64_t min_row_size_to_use_multithread, bool multi_precision, bool use_global_beta_pow);

PADDLE_API Tensor add(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& add_(Tensor& x, const Tensor& y);

PADDLE_API Tensor add_n(const std::vector<Tensor>& inputs);

PADDLE_API Tensor addmm(const Tensor& input, const Tensor& x, const Tensor& y, float beta, float alpha);

PADDLE_API Tensor affine_grid(const Tensor& input, const IntArray& outputShape, bool use_cudnn = true, bool align_corners = true);

PADDLE_API Tensor all(const Tensor& x, const std::vector<int64_t>& axis = {}, bool keepdim = false);

PADDLE_API Tensor allclose(const Tensor& x, const Tensor& y, const Scalar& rtol, const Scalar& atol, bool equal_nan);

PADDLE_API Tensor amax(const Tensor& x, const std::vector<int64_t>& axis = {}, bool keepdim = false);

PADDLE_API Tensor amin(const Tensor& x, const std::vector<int64_t>& axis = {}, bool keepdim = false);

PADDLE_API Tensor any(const Tensor& x, const std::vector<int64_t>& axis = {}, bool keepdim = false);

PADDLE_API Tensor arange(const Tensor& start, const Tensor& end, const Tensor& step, DataType dtype, const Place& place = {});

PADDLE_API Tensor argmax(const Tensor& x, const Scalar& axis, bool keepdims, bool flatten, int dtype);

PADDLE_API Tensor argmin(const Tensor& x, const Scalar& axis, bool keepdims, bool flatten, int dtype);

PADDLE_API Tensor as_complex(const Tensor& x);

PADDLE_API Tensor as_real(const Tensor& x);

PADDLE_API Tensor assign(const Tensor& x);

PADDLE_API Tensor& assign_out_(const Tensor& x, Tensor& output);

PADDLE_API Tensor& assign_value_(Tensor& output, const std::vector<int>& shape, DataType dtype, const std::vector<phi::Scalar>& values, const Place& place = {});

PADDLE_API std::tuple<Tensor, Tensor, Tensor> auc(const Tensor& x, const Tensor& label, const Tensor& stat_pos, const Tensor& stat_neg, const paddle::optional<Tensor>& ins_tag_weight, const std::string& curve, int num_thresholds, int slide_steps);

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, Tensor&> average_accumulates_(const Tensor& param, Tensor& in_sum_1, Tensor& in_sum_2, Tensor& in_sum_3, Tensor& in_num_accumulates, Tensor& in_old_num_accumulates, Tensor& in_num_updates, float average_window, int64_t max_average_window, int64_t min_average_window);

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> batch_norm(const Tensor& x, const Tensor& mean, const Tensor& variance, const Tensor& scale, const Tensor& bias, bool is_test, float momentum, float epsilon, const std::string& data_layout, bool use_global_stats, bool trainable_statistics);

PADDLE_API Tensor bce_loss(const Tensor& input, const Tensor& label);

PADDLE_API Tensor bicubic_interp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode);

PADDLE_API Tensor bilinear_interp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode);

PADDLE_API Tensor bilinear_tensor_product(const Tensor& x, const Tensor& y, const Tensor& weight, const paddle::optional<Tensor>& bias);

PADDLE_API Tensor bitwise_and(const Tensor& x, const Tensor& y);

PADDLE_API Tensor bitwise_not(const Tensor& x);

PADDLE_API Tensor bitwise_or(const Tensor& x, const Tensor& y);

PADDLE_API Tensor bitwise_xor(const Tensor& x, const Tensor& y);

PADDLE_API Tensor box_coder(const Tensor& prior_box, const paddle::optional<Tensor>& prior_box_var, const Tensor& target_box, const std::string& code_type, bool box_normalized, int axis, const std::vector<float>& variance);

PADDLE_API Tensor cast(const Tensor& x, DataType dtype);

PADDLE_API Tensor celu(const Tensor& x, float alpha);

PADDLE_API std::tuple<std::vector<Tensor>&, Tensor&> check_finite_and_unscale_(std::vector<Tensor>& x, const Tensor& scale, Tensor& input_found_infinite);

PADDLE_API std::tuple<Tensor, Tensor> class_center_sample(const Tensor& label, int num_classes, int num_samples, int ring_id, int rank, int nranks, bool fix_seed, int seed);

PADDLE_API Tensor clip(const Tensor& x, const Scalar& min, const Scalar& max);

PADDLE_API Tensor& clip_(Tensor& x, const Scalar& min, const Scalar& max);

PADDLE_API Tensor clip_by_norm(const Tensor& x, float max_norm);

PADDLE_API std::tuple<std::vector<Tensor>, Tensor> coalesce_tensor(const std::vector<Tensor>& input, DataType dtype, bool copy_data = false, bool set_constant = false, bool persist_output = false, float constant = 0.0, bool use_align = true, int align_size = -1, int size_of_dtype = -1, const std::vector<int64_t>& concated_shapes = {}, const std::vector<int64_t>& concated_ranks = {});

PADDLE_API Tensor complex(const Tensor& real, const Tensor& imag);

PADDLE_API Tensor concat(const std::vector<Tensor>& x, const Scalar& axis);

PADDLE_API Tensor conj(const Tensor& x);

PADDLE_API Tensor conv2d(const Tensor& input, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, const std::vector<int>& dilations, int groups, const std::string& data_format);

PADDLE_API Tensor conv2d_transpose(const Tensor& x, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const IntArray& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format);

PADDLE_API Tensor conv3d(const Tensor& input, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format);

PADDLE_API Tensor conv3d_transpose(const Tensor& x, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::vector<int>& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format);

PADDLE_API Tensor copy_to(const Tensor& x, const Place& place, bool blocking);

PADDLE_API Tensor crop(const Tensor& x, const IntArray& shape, const IntArray& offsets);

PADDLE_API std::tuple<Tensor, Tensor> cross_entropy_with_softmax(const Tensor& input, const Tensor& label, bool soft_label, bool use_softmax, bool numeric_stable_mode, int ignore_index, int axis);

PADDLE_API Tensor cumprod(const Tensor& x, int dim);

PADDLE_API Tensor cumsum(const Tensor& x, const Scalar& axis, bool flatten, bool exclusive, bool reverse);

PADDLE_API Tensor decode_jpeg(const Tensor& x, const std::string& mode, const Place& place);

PADDLE_API Tensor deformable_conv(const Tensor& x, const Tensor& offset, const Tensor& filter, const paddle::optional<Tensor>& mask, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations, int deformable_groups, int groups, int im2col_step);

PADDLE_API Tensor depthwise_conv2d(const Tensor& x, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, bool use_gpudnn);

PADDLE_API Tensor depthwise_conv2d_transpose(const Tensor& x, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const IntArray& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format);

PADDLE_API Tensor diag_embed(const Tensor& input, int offset, int dim1, int dim2);

PADDLE_API std::tuple<std::vector<Tensor>, std::vector<Tensor>, Tensor> distribute_fpn_proposals(const Tensor& fpn_rois, const paddle::optional<Tensor>& rois_num, int min_level, int max_level, int refer_level, int refer_scale, bool pixel_offset);

PADDLE_API Tensor divide(const Tensor& x, const Tensor& y);

PADDLE_API std::tuple<Tensor, Tensor> dropout(const Tensor& x, const paddle::optional<Tensor>& seed_tensor, const Scalar& p, bool is_test, const std::string& mode, int seed, bool fix_seed);

PADDLE_API std::tuple<Tensor, Tensor> edit_distance(const Tensor& hyps, const Tensor& refs, const paddle::optional<Tensor>& hypslength, const paddle::optional<Tensor>& refslength, bool normalized = false);

PADDLE_API std::tuple<Tensor, Tensor> eigh(const Tensor& x, const std::string& UPLO);

PADDLE_API Tensor eigvals(const Tensor& x);

PADDLE_API std::tuple<Tensor, Tensor> eigvalsh(const Tensor& x, const std::string& uplo, bool is_test);

PADDLE_API std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>> einsum(const std::vector<Tensor>& x, const std::string& equation);

PADDLE_API Tensor elementwise_pow(const Tensor& x, const Tensor& y);

PADDLE_API Tensor elu(const Tensor& x, float alpha);

PADDLE_API Tensor& elu_(Tensor& x, float alpha);

PADDLE_API Tensor embedding(const Tensor& x, const Tensor& weight, int64_t padding_idx = -1, bool sparse = false);

PADDLE_API Tensor empty(const IntArray& shape, DataType dtype = DataType::FLOAT32, const Place& place = CPUPlace());

PADDLE_API Tensor empty_like(const Tensor& x, DataType dtype = DataType::UNDEFINED, const Place& place = {});

PADDLE_API Tensor equal(const Tensor& x, const Tensor& y, int axis = -1);

PADDLE_API Tensor equal_all(const Tensor& x, const Tensor& y);

PADDLE_API Tensor expand(const Tensor& x, const IntArray& shape);

PADDLE_API Tensor expand_as(const Tensor& x, const paddle::optional<Tensor>& y, const std::vector<int>& target_shape);

PADDLE_API Tensor& exponential_(Tensor& x, float lam);

PADDLE_API Tensor eye(const Scalar& num_rows, const Scalar& num_columns, DataType dtype = DataType::FLOAT32, const Place& place = {});

PADDLE_API Tensor fill(const Tensor& x, const Scalar& value);

PADDLE_API Tensor& fill_(Tensor& x, const Scalar& value);

PADDLE_API Tensor fill_diagonal(const Tensor& x, float value, int offset, bool wrap);

PADDLE_API Tensor& fill_diagonal_(Tensor& x, float value, int offset, bool wrap);

PADDLE_API Tensor fill_diagonal_tensor(const Tensor& x, const Tensor& y, int64_t offset, int dim1, int dim2);

PADDLE_API Tensor& fill_diagonal_tensor_(Tensor& x, const Tensor& y, int64_t offset, int dim1, int dim2);

PADDLE_API Tensor flatten(const Tensor& x, int start_axis, int stop_axis);

PADDLE_API Tensor& flatten_(Tensor& x, int start_axis, int stop_axis);

PADDLE_API Tensor floor_divide(const Tensor& x, const Tensor& y);

PADDLE_API Tensor fmax(const Tensor& x, const Tensor& y, int axis);

PADDLE_API Tensor fmin(const Tensor& x, const Tensor& y, int axis);

PADDLE_API Tensor frame(const Tensor& x, int frame_length, int hop_length, int axis);

PADDLE_API Tensor frobenius_norm(const Tensor& x, const std::vector<int64_t>& axis, bool keep_dim, bool reduce_all);

PADDLE_API Tensor full(const IntArray& shape, const Scalar& value, DataType dtype = DataType::FLOAT32, const Place& place = CPUPlace());

PADDLE_API Tensor& full_(Tensor& output, const IntArray& shape, const Scalar& value, DataType dtype = DataType::FLOAT32, const Place& place = CPUPlace());

PADDLE_API Tensor full_batch_size_like(const Tensor& input, const std::vector<int>& shape, DataType dtype, const Scalar& value, int input_dim_idx, int output_dim_idx, const Place& place = CPUPlace());

PADDLE_API Tensor full_like(const Tensor& x, const Scalar& value, DataType dtype = DataType::UNDEFINED, const Place& place = {});

PADDLE_API Tensor gather(const Tensor& x, const Tensor& index, const Scalar& axis = 0);

PADDLE_API Tensor gather_nd(const Tensor& x, const Tensor& index);

PADDLE_API Tensor gather_tree(const Tensor& ids, const Tensor& parents);

PADDLE_API Tensor gaussian(const IntArray& shape, float mean, float std, int seed, DataType dtype, const Place& place = {});

PADDLE_API Tensor gelu(const Tensor& x, bool approximate);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> generate_proposals(const Tensor& scores, const Tensor& bbox_deltas, const Tensor& im_shape, const Tensor& anchors, const Tensor& variances, int pre_nms_top_n, int post_nms_top_n, float nms_thresh, float min_size, float eta, bool pixel_offset = true);

PADDLE_API Tensor greater_equal(const Tensor& x, const Tensor& y, int axis = -1);

PADDLE_API Tensor greater_than(const Tensor& x, const Tensor& y, int axis = -1);

PADDLE_API Tensor grid_sample(const Tensor& x, const Tensor& grid, const std::string& mode, const std::string& padding_mode, bool align_corners);

PADDLE_API Tensor group_norm(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon, int groups, const std::string& data_layout);

PADDLE_API Tensor gumbel_softmax(const Tensor& x, float temperature, bool hard, int axis);

PADDLE_API Tensor hardswish(const Tensor& x, float threshold = 6.0, float scale = 6.0, float offset = 3.0);

PADDLE_API Tensor hardtanh(const Tensor& x, float t_min, float t_max);

PADDLE_API Tensor histogram(const Tensor& input, int64_t bins, int min, int max);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> hsigmoid_loss(const Tensor& x, const Tensor& w, const Tensor& label, const paddle::optional<Tensor>& path, const paddle::optional<Tensor>& code, const paddle::optional<Tensor>& bias, int num_classes, bool remote_prefetch, int trainer_id, const std::vector<int64_t>& height_sections, const std::vector<std::string>& epmap, const std::vector<std::string>& table_names, bool is_sparse);

PADDLE_API std::tuple<Tensor, Tensor> huber_loss(const Tensor& input, const Tensor& label, float delta);

PADDLE_API Tensor imag(const Tensor& x);

PADDLE_API Tensor increment(const Tensor& x, float value);

PADDLE_API Tensor& increment_(Tensor& x, float value);

PADDLE_API Tensor index_add(const Tensor& x, const Tensor& index, const Tensor& add_value, int axis);

PADDLE_API Tensor& index_add_(Tensor& x, const Tensor& index, const Tensor& add_value, int axis);

PADDLE_API Tensor index_sample(const Tensor& x, const Tensor& index);

PADDLE_API Tensor index_select(const Tensor& x, const Tensor& index, int axis);

PADDLE_API Tensor instance_norm(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon);

PADDLE_API Tensor inverse(const Tensor& x);

PADDLE_API Tensor is_empty(const Tensor& x);

PADDLE_API Tensor isclose(const Tensor& x, const Tensor& y, const Scalar& rtol, const Scalar& atol, bool equal_nan);

PADDLE_API Tensor isfinite(const Tensor& x);

PADDLE_API Tensor isinf(const Tensor& x);

PADDLE_API Tensor isnan(const Tensor& x);

PADDLE_API Tensor kldiv_loss(const Tensor& x, const Tensor& label, const std::string& reduction);

PADDLE_API Tensor kron(const Tensor& x, const Tensor& y);

PADDLE_API std::tuple<Tensor, Tensor> kthvalue(const Tensor& x, int k, int axis, bool keepdim);

PADDLE_API Tensor label_smooth(const Tensor& label, const paddle::optional<Tensor>& prior_dist, float epsilon);

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, paddle::optional<Tensor>&> lamb_(Tensor& param, const Tensor& grad, const Tensor& learning_rate, Tensor& moment1, Tensor& moment2, Tensor& beta1_pow, Tensor& beta2_pow, paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, float weight_decay, float beta1, float beta2, float epsilon, bool multi_precision);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> layer_norm(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon, int begin_norm_axis, bool is_test);

PADDLE_API Tensor leaky_relu(const Tensor& x, float negative_slope);

PADDLE_API Tensor lerp(const Tensor& x, const Tensor& y, const Tensor& weight);

PADDLE_API Tensor& lerp_(Tensor& x, const Tensor& y, const Tensor& weight);

PADDLE_API Tensor less_equal(const Tensor& x, const Tensor& y, int axis = -1);

PADDLE_API Tensor less_than(const Tensor& x, const Tensor& y, int axis = -1);

PADDLE_API Tensor linear_interp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode);

PADDLE_API Tensor linspace(const Tensor& start, const Tensor& stop, const Tensor& number, DataType dtype, const Place& place);

PADDLE_API Tensor log(const Tensor& x);

PADDLE_API Tensor log_loss(const Tensor& input, const Tensor& label, float epsilon);

PADDLE_API Tensor log_softmax(const Tensor& x, int axis);

PADDLE_API Tensor logcumsumexp(const Tensor& x, int axis, bool flatten, bool exclusive, bool reverse);

PADDLE_API Tensor logical_and(const Tensor& x, const Tensor& y);

PADDLE_API Tensor logical_not(const Tensor& x);

PADDLE_API Tensor logical_or(const Tensor& x, const Tensor& y);

PADDLE_API Tensor logical_xor(const Tensor& x, const Tensor& y);

PADDLE_API Tensor logsumexp(const Tensor& x, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all);

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor> lstsq(const Tensor& x, const Tensor& y, const Scalar& rcond, const std::string& driver);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> lu(const Tensor& x, bool pivot);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> lu_unpack(const Tensor& x, const Tensor& y, bool unpack_ludata, bool unpack_pivots);

PADDLE_API std::tuple<Tensor, Tensor> margin_cross_entropy(const Tensor& logits, const Tensor& label, bool return_softmax, int ring_id, int rank, int nranks, float margin1, float margin2, float margin3, float scale);

PADDLE_API Tensor masked_select(const Tensor& x, const Tensor& mask);

PADDLE_API Tensor matmul(const Tensor& x, const Tensor& y, bool transpose_x = false, bool transpose_y = false);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> matrix_nms(const Tensor& bboxes, const Tensor& scores, float score_threshold, int nms_top_k, int keep_top_k, float post_threshold = 0., bool use_gaussian = false, float gaussian_sigma = 2.0, int background_label = 0, bool normalized = true);

PADDLE_API Tensor matrix_power(const Tensor& x, int n);

PADDLE_API Tensor matrix_rank(const Tensor& x, float tol, bool use_default_tol = true, bool hermitian = false);

PADDLE_API Tensor matrix_rank_tol(const Tensor& x, const Tensor& atol_tensor, bool use_default_tol = true, bool hermitian = false);

PADDLE_API Tensor max(const Tensor& x, const IntArray& axis = {}, bool keepdim = false);

PADDLE_API std::tuple<Tensor, Tensor> max_pool2d_with_index(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool global_pooling, bool adaptive);

PADDLE_API std::tuple<Tensor, Tensor> max_pool3d_with_index(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool global_pooling, bool adaptive);

PADDLE_API Tensor maximum(const Tensor& x, const Tensor& y);

PADDLE_API Tensor maxout(const Tensor& x, int groups, int axis);

PADDLE_API Tensor mean(const Tensor& x, const IntArray& axis = {}, bool keepdim = false);

PADDLE_API Tensor mean_all(const Tensor& x);

PADDLE_API Tensor merge_selected_rows(const Tensor& x);

PADDLE_API std::tuple<std::vector<Tensor>&, std::vector<Tensor>&, std::vector<Tensor>&, std::vector<Tensor>&, std::vector<Tensor>&, paddle::optional<std::vector<Tensor>>&> merged_adam_(std::vector<Tensor>& param, const std::vector<Tensor>& grad, const std::vector<Tensor>& learning_rate, std::vector<Tensor>& moment1, std::vector<Tensor>& moment2, std::vector<Tensor>& beta1_pow, std::vector<Tensor>& beta2_pow, paddle::optional<std::vector<Tensor>>& master_param, const Scalar& beta1, const Scalar& beta2, const Scalar& epsilon, bool multi_precision, bool use_global_beta_pow);

PADDLE_API std::tuple<std::vector<Tensor>&, std::vector<Tensor>&, paddle::optional<std::vector<Tensor>>&> merged_momentum_(std::vector<Tensor>& param, const std::vector<Tensor>& grad, std::vector<Tensor>& velocity, const std::vector<Tensor>& learning_rate, paddle::optional<std::vector<Tensor>>& master_param, float mu, bool use_nesterov = false, const std::vector<std::string>& regularization_method = {}, const std::vector<float>& regularization_coeff = {}, bool multi_precision = false, float rescale_grad = 1.0f);

PADDLE_API std::vector<Tensor> meshgrid(const std::vector<Tensor>& inputs);

PADDLE_API Tensor min(const Tensor& x, const IntArray& axis = {}, bool keepdim = false);

PADDLE_API Tensor minimum(const Tensor& x, const Tensor& y);

PADDLE_API Tensor mish(const Tensor& x, float lambda);

PADDLE_API std::tuple<Tensor, Tensor> mode(const Tensor& x, int axis, bool keepdim);

PADDLE_API std::tuple<Tensor&, Tensor&, paddle::optional<Tensor>&> momentum_(Tensor& param, const Tensor& grad, Tensor& velocity, const Tensor& learning_rate, paddle::optional<Tensor>& master_param, float mu, bool use_nesterov = false, const std::string& regularization_method = "", float regularization_coeff = 0.0, bool multi_precision = false, float rescale_grad = 1.0f);

PADDLE_API Tensor multi_dot(const std::vector<Tensor>& x);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> multiclass_nms3(const Tensor& bboxes, const Tensor& scores, const paddle::optional<Tensor>& rois_num, float score_threshold, int nms_top_k, int keep_top_k, float nms_threshold = 0.3, bool normalized = true, float nms_eta = 1.0, int background_label = 0);

PADDLE_API Tensor multinomial(const Tensor& x, const Scalar& num_samples, bool replacement);

PADDLE_API Tensor multiplex(const std::vector<Tensor>& inputs, const Tensor& index);

PADDLE_API Tensor multiply(const Tensor& x, const Tensor& y);

PADDLE_API Tensor nearest_interp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode);

PADDLE_API std::tuple<Tensor, Tensor> nll_loss(const Tensor& input, const Tensor& label, const paddle::optional<Tensor>& weight, int64_t ignore_index, const std::string& reduction);

PADDLE_API Tensor nms(const Tensor& x, float threshold);

PADDLE_API Tensor nonzero(const Tensor& condition);

PADDLE_API std::tuple<Tensor, Tensor> norm(const Tensor& x, int axis, float epsilon, bool is_test);

PADDLE_API Tensor not_equal(const Tensor& x, const Tensor& y, int axis = -1);

PADDLE_API Tensor numel(const Tensor& x);

PADDLE_API Tensor one_hot(const Tensor& x, const Scalar& num_classes);

PADDLE_API Tensor ones(const IntArray& shape, DataType dtype = DataType::FLOAT32, const Place& place = CPUPlace());

PADDLE_API Tensor ones_like(const Tensor& x, DataType dtype = DataType::UNDEFINED, const Place& place = {});

PADDLE_API Tensor p_norm(const Tensor& x, float porder, int axis, float epsilon, bool keepdim, bool asvector = false);

PADDLE_API Tensor pad(const Tensor& x, const std::vector<int>& paddings, const Scalar& pad_value);

PADDLE_API Tensor pad3d(const Tensor& x, const IntArray& paddings, const std::string& mode, float pad_value, const std::string& data_format);

PADDLE_API Tensor pixel_shuffle(const Tensor& x, int upscale_factor, const std::string& data_format);

PADDLE_API Tensor pool2d(const Tensor& x, const IntArray& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm, bool use_gpudnn);

PADDLE_API Tensor pool3d(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm, bool use_gpudnn);

PADDLE_API Tensor pow(const Tensor& x, const Scalar& y);

PADDLE_API Tensor prelu(const Tensor& x, const Tensor& alpha, const std::string& data_format, const std::string& mode);

PADDLE_API std::tuple<Tensor, Tensor> prior_box(const Tensor& input, const Tensor& image, const std::vector<float>& min_sizes, const std::vector<float>& aspect_ratios, const std::vector<float>& variances, const std::vector<float>& max_sizes = {}, bool flip = true, bool clip = true, float step_w = 0.0, float step_h = 0.0, float offset = 0.5, bool min_max_aspect_ratios_order = false);

PADDLE_API Tensor prod(const Tensor& x, const IntArray& dims, bool keep_dim, bool reduce_all);

PADDLE_API Tensor psroi_pool(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height, int pooled_width, int output_channels, float spatial_scale);

PADDLE_API Tensor put_along_axis(const Tensor& arr, const Tensor& indices, const Tensor& values, int axis, const std::string& reduce);

PADDLE_API Tensor& put_along_axis_(Tensor& arr, const Tensor& indices, const Tensor& values, int axis, const std::string& reduce);

PADDLE_API std::tuple<Tensor, Tensor> qr(const Tensor& x, const std::string& mode);

PADDLE_API Tensor randint(int low, int high, const IntArray& shape, DataType dtype = DataType::INT64, const Place& place = {});

PADDLE_API Tensor randperm(int n, DataType dtype, const Place& place = {});

PADDLE_API Tensor real(const Tensor& x);

PADDLE_API Tensor relu(const Tensor& x);

PADDLE_API Tensor& relu_(Tensor& x);

PADDLE_API Tensor relu6(const Tensor& x, float threshold);

PADDLE_API Tensor remainder(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& remainder_(Tensor& x, const Tensor& y);

PADDLE_API Tensor renorm(const Tensor& x, float p, int axis, float max_norm);

PADDLE_API Tensor repeat_interleave(const Tensor& x, int repeats, int axis);

PADDLE_API Tensor repeat_interleave_with_tensor_index(const Tensor& x, const Tensor& repeats, int axis);

PADDLE_API Tensor reshape(const Tensor& x, const IntArray& shape);

PADDLE_API Tensor& reshape_(Tensor& x, const IntArray& shape);

PADDLE_API Tensor reverse(const Tensor& x, const IntArray& axis);

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> rmsprop_(Tensor& param, Tensor& mean_square, const Tensor& grad, Tensor& moment, const Tensor& learning_rate, Tensor& mean_grad, float epsilon, float decay, float momentum, bool centered);

PADDLE_API Tensor roi_align(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height, int pooled_width, float spatial_scale, int sampling_ratio, bool aligned);

PADDLE_API Tensor roi_pool(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height, int pooled_width, float spatial_scale);

PADDLE_API Tensor roll(const Tensor& x, const IntArray& shifts, const std::vector<int64_t>& axis);

PADDLE_API Tensor rsqrt(const Tensor& x);

PADDLE_API Tensor& rsqrt_(Tensor& x);

PADDLE_API Tensor scale(const Tensor& x, const Scalar& scale, float bias, bool bias_after_scale);

PADDLE_API Tensor& scale_(Tensor& x, const Scalar& scale, float bias, bool bias_after_scale);

PADDLE_API Tensor scatter(const Tensor& x, const Tensor& index, const Tensor& updates, bool overwrite);

PADDLE_API Tensor& scatter_(Tensor& x, const Tensor& index, const Tensor& updates, bool overwrite);

PADDLE_API Tensor scatter_nd_add(const Tensor& x, const Tensor& index, const Tensor& updates);

PADDLE_API Tensor searchsorted(const Tensor& sorted_sequence, const Tensor& values, bool out_int32, bool right);

PADDLE_API std::tuple<Tensor, Tensor> segment_pool(const Tensor& x, const Tensor& segment_ids, const std::string& pooltype);

PADDLE_API Tensor selu(const Tensor& x, float scale, float alpha);

PADDLE_API Tensor send_u_recv(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const std::string& reduce_op = "SUM", const IntArray& out_size = {0});

PADDLE_API Tensor send_ue_recv(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const std::string& message_op, const std::string& reduce_op, const IntArray& out_size);

PADDLE_API std::tuple<Tensor&, paddle::optional<Tensor>&> sgd_(Tensor& param, const Tensor& learning_rate, const Tensor& grad, paddle::optional<Tensor>& master_param, bool multi_precision);

PADDLE_API Tensor shape(const Tensor& input);

PADDLE_API Tensor shard_index(const Tensor& input, int index_num, int nshards, int shard_id, int ignore_value);

PADDLE_API Tensor sigmoid(const Tensor& x);

PADDLE_API Tensor sigmoid_cross_entropy_with_logits(const Tensor& x, const Tensor& label, bool normalize, int ignore_index);

PADDLE_API Tensor sign(const Tensor& x);

PADDLE_API Tensor slice(const Tensor& input, const std::vector<int64_t>& axes, const IntArray& starts, const IntArray& ends, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis);

PADDLE_API Tensor slogdet(const Tensor& x);

PADDLE_API Tensor softmax(const Tensor& x, int axis);

PADDLE_API Tensor& softmax_(Tensor& x, int axis);

PADDLE_API Tensor softplus(const Tensor& x, float beta, float threshold);

PADDLE_API Tensor softshrink(const Tensor& x, float threshold);

PADDLE_API Tensor softsign(const Tensor& x);

PADDLE_API Tensor spectral_norm(const Tensor& weight, const Tensor& u, const Tensor& v, int dim, int power_iters, float eps);

PADDLE_API std::vector<Tensor> split(const Tensor& x, const IntArray& sections, const Scalar& axis);

PADDLE_API std::vector<Tensor> split_with_num(const Tensor& x, int num, const Scalar& axis);

PADDLE_API Tensor sqrt(const Tensor& x);

PADDLE_API Tensor& sqrt_(Tensor& x);

PADDLE_API Tensor square(const Tensor& x);

PADDLE_API Tensor squared_l2_norm(const Tensor& x);

PADDLE_API Tensor squeeze(const Tensor& x, const IntArray& axis);

PADDLE_API Tensor& squeeze_(Tensor& x, const IntArray& axis);

PADDLE_API Tensor stack(const std::vector<Tensor>& x, int axis);

PADDLE_API Tensor strided_slice(const Tensor& x, const std::vector<int>& axes, const IntArray& starts, const IntArray& ends, const IntArray& strides);

PADDLE_API Tensor subtract(const Tensor& x, const Tensor& y);

PADDLE_API Tensor& subtract_(Tensor& x, const Tensor& y);

PADDLE_API Tensor sum(const Tensor& x, const IntArray& axis = {}, DataType dtype = DataType::UNDEFINED, bool keepdim = false);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& x, bool full_matrices);

PADDLE_API Tensor swish(const Tensor& x, float beta = 1.0);

PADDLE_API std::tuple<Tensor, Tensor&, Tensor&, Tensor, Tensor, Tensor> sync_batch_norm_(const Tensor& x, Tensor& mean, Tensor& variance, const Tensor& scale, const Tensor& bias, bool is_test, float momentum, float epsilon, const std::string& data_layout, bool use_global_stats, bool trainable_statistics);

PADDLE_API Tensor take_along_axis(const Tensor& arr, const Tensor& indices, int axis);

PADDLE_API Tensor tanh_shrink(const Tensor& x);

PADDLE_API Tensor temporal_shift(const Tensor& x, int seg_num, float shift_ratio, const std::string& data_format_str);

PADDLE_API Tensor thresholded_relu(const Tensor& x, float threshold);

PADDLE_API Tensor tile(const Tensor& x, const IntArray& repeat_times);

PADDLE_API std::tuple<Tensor, Tensor> topk(const Tensor& x, const Scalar& k, int axis = -1, bool largest = true, bool sorted = true);

PADDLE_API Tensor transpose(const Tensor& x, const std::vector<int>& perm);

PADDLE_API Tensor triangular_solve(const Tensor& x, const Tensor& y, bool upper, bool transpose, bool unitriangular);

PADDLE_API Tensor tril(const Tensor& x, int diagonal, bool lower);

PADDLE_API Tensor tril_indices(int rows, int cols, int offset, DataType dtype, const Place& place = {});

PADDLE_API Tensor trilinear_interp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode);

PADDLE_API Tensor triu_indices(int row, int col, int offset, DataType dtype, const Place& place = {});

PADDLE_API Tensor truncated_gaussian_random(const std::vector<int>& shape, float mean, float std, int seed, DataType dtype = DataType::FLOAT32, const Place& place = {});

PADDLE_API std::vector<Tensor> unbind(const Tensor& input, int axis);

PADDLE_API Tensor unfold(const Tensor& x, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations);

PADDLE_API Tensor uniform(const IntArray& shape, DataType dtype, const Scalar& min, const Scalar& max, int seed, const Place& place = {});

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor> unique(const Tensor& x, bool return_index, bool return_inverse, bool return_counts, const std::vector<int>& axis, DataType dtype = DataType::INT64);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> unique_consecutive(const Tensor& x, bool return_inverse, bool return_counts, const std::vector<int>& axis, int dtype);

PADDLE_API Tensor unsqueeze(const Tensor& x, const IntArray& axis);

PADDLE_API Tensor& unsqueeze_(Tensor& x, const IntArray& axis);

PADDLE_API std::vector<Tensor> unstack(const Tensor& x, int axis, int num);

PADDLE_API std::tuple<std::vector<Tensor>&, Tensor&, Tensor&, Tensor&> update_loss_scaling_(std::vector<Tensor>& x, const Tensor& found_infinite, Tensor& prev_loss_scaling, Tensor& in_good_steps, Tensor& in_bad_steps, int incr_every_n_steps, int decr_every_n_nan_or_inf, float incr_ratio, float decr_ratio, const Scalar& stop_update);

PADDLE_API std::tuple<Tensor, Tensor> viterbi_decode(const Tensor& potentials, const Tensor& transition_params, const Tensor& lengths, bool include_bos_eos_tag);

PADDLE_API Tensor warpctc(const Tensor& logits, const Tensor& label, const paddle::optional<Tensor>& logits_length, const paddle::optional<Tensor>& labels_length, int blank, bool norm_by_times);

PADDLE_API Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y);

PADDLE_API std::tuple<Tensor, Tensor> yolo_box(const Tensor& x, const Tensor& img_size, const std::vector<int>& anchors, int class_num, float conf_thresh, int downsample_ratio, bool clip_bbox, float scale_x_y = 1.0, bool iou_aware = false, float iou_aware_factor = 0.5);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> yolo_loss(const Tensor& x, const Tensor& gt_box, const Tensor& gt_label, const paddle::optional<Tensor>& gt_score, const std::vector<int>& anchors, const std::vector<int>& anchor_mask, int class_num, float ignore_thresh, int downsample_ratio, bool use_label_smooth = true, float scale_x_y = 1.0);

PADDLE_API Tensor zeros(const IntArray& shape, DataType dtype = DataType::FLOAT32, const Place& place = CPUPlace());

PADDLE_API Tensor zeros_like(const Tensor& x, DataType dtype = DataType::UNDEFINED, const Place& place = {});

PADDLE_API Tensor bincount(const Tensor& x, const paddle::optional<Tensor>& weights, const Scalar& minlength);

PADDLE_API std::vector<Tensor> broadcast_tensors(const std::vector<Tensor>& input);

PADDLE_API Tensor dirichlet(const Tensor& alpha);

PADDLE_API std::tuple<Tensor, Tensor> eig(const Tensor& x);

PADDLE_API Tensor fold(const Tensor& x, const std::vector<int>& output_sizes, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations);

PADDLE_API Tensor overlap_add(const Tensor& x, int hop_length, int axis);

PADDLE_API std::tuple<Tensor, Tensor, std::vector<Tensor>> rnn(const Tensor& x, const std::vector<Tensor>& pre_state, const std::vector<Tensor>& weight_list, const paddle::optional<Tensor>& sequence_length, const Tensor& dropout_state_in, float dropout_prob = 0.0, bool is_bidirec = false, int input_size = 10, int hidden_size = 100, int num_layers = 1, const std::string& mode = "RNN_TANH", int seed = 0, bool is_test = false);

PADDLE_API Tensor uniform_inplace(const Tensor& x, float min, float max, int seed, int diag_num, int diag_step, float diag_val);

PADDLE_API Tensor& uniform_inplace_(Tensor& x, float min, float max, int seed, int diag_num, int diag_step, float diag_val);

PADDLE_API Tensor unpool(const Tensor& x, const Tensor& indices, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& padding, const IntArray& output_size, const std::string& data_format);

PADDLE_API Tensor unpool3d(const Tensor& x, const Tensor& indices, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& padding, const std::vector<int>& output_size, const std::string& data_format);


}  // namespace experimental
}  // namespace paddle
