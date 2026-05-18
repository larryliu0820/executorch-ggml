/**
 * benchmark_needle.cpp — Needle encoder/decoder benchmark + smoke test.
 *
 * Loads needle.pte (three methods: token_embedding, encoder, decoder),
 * runs a fixed-length encoder pass and N decode steps, and reports tok/s.
 * Also prints output sums as a smoke test that the run did not hit NaN.
 *
 * Usage:
 *   ./benchmark_needle <model.pte> [--n-decode N] [--enc-len L] [--vocab V]
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/result.h>

using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::from_blob;
using executorch::runtime::Error;
using executorch::runtime::EValue;

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// Fetch a metadata constant method that returns a single int.
static int64_t get_int_method(Module& m, const char* name, int64_t fallback) {
  auto out = m.execute(name);
  if (!out.ok() || out.get().empty()) return fallback;
  const auto& ev = out.get()[0];
  if (ev.isInt()) return ev.toInt();
  return fallback;
}

// Reduce a tensor to (sum, abs_sum, finite-count) for a sanity print.
struct TensorStats {
  double sum;
  double abs_sum;
  int64_t finite;
  int64_t total;
};

static TensorStats stat_tensor(const executorch::aten::Tensor& t) {
  TensorStats s{0.0, 0.0, 0, 0};
  if (t.scalar_type() != executorch::aten::ScalarType::Float) {
    fprintf(stderr, "  (warn: stat_tensor expects float, got %d)\n",
            (int)t.scalar_type());
    return s;
  }
  const float* data = t.template const_data_ptr<float>();
  int64_t n = t.numel();
  s.total = n;
  for (int64_t i = 0; i < n; ++i) {
    float v = data[i];
    if (std::isfinite(v)) {
      s.sum += v;
      s.abs_sum += std::abs(v);
      s.finite++;
    }
  }
  return s;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
        "Usage: %s <model.pte> [--n-decode N] [--enc-len L] [--vocab V]\n",
        argv[0]);
    return 1;
  }
  const char* model_path = argv[1];
  int n_decode = 32;
  int enc_len = 24;
  int vocab_size = 8192;

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--n-decode") == 0 && i + 1 < argc)
      n_decode = atoi(argv[++i]);
    else if (strcmp(argv[i], "--enc-len") == 0 && i + 1 < argc)
      enc_len = atoi(argv[++i]);
    else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc)
      vocab_size = atoi(argv[++i]);
  }

  fprintf(stderr, "Loading %s ...\n", model_path);
  Module model(model_path);

  // Load methods up front so latency below is just compute, not load time.
  for (const char* method : {"token_embedding", "encoder", "decoder"}) {
    auto err = model.load_method(method);
    if (err != Error::Ok) {
      fprintf(stderr, "load_method('%s') failed: 0x%x\n", method, (int)err);
      return 1;
    }
  }

  // Pull metadata constants exported alongside the methods.
  vocab_size = (int)get_int_method(model, "vocab_size", vocab_size);
  int d_model = (int)get_int_method(model, "d_model", 512);
  int max_enc_len = (int)get_int_method(model, "max_enc_len", 1024);
  int max_gen_len = (int)get_int_method(model, "max_gen_len", 512);
  fprintf(stderr,
          "Metadata: vocab=%d d_model=%d max_enc_len=%d max_gen_len=%d\n",
          vocab_size, d_model, max_enc_len, max_gen_len);

  if (enc_len > max_enc_len) enc_len = max_enc_len;
  if (n_decode > max_gen_len - 1) n_decode = max_gen_len - 1;

  // Build a deterministic encoder prompt so reruns are comparable AND so a
  // Python-side eager reference can produce the same tokens trivially.
  // Pattern: tok[i] = 1 + (i * 7 + 11) % (vocab-1) — chosen for spread
  // without overflow concerns.
  std::vector<int64_t> enc_tokens(enc_len);
  for (int i = 0; i < enc_len; i++) {
    enc_tokens[i] = 1 + ((int64_t)i * 7 + 11) % (vocab_size - 1);
  }
  fprintf(stderr,
          "Prompt: enc_len=%d, n_decode=%d, vocab=%d\n",
          enc_len, n_decode, vocab_size);

  auto src = from_blob(enc_tokens.data(), {1, enc_len},
                       executorch::aten::ScalarType::Long);

  // --- Warmup: 1 encoder + 1 decoder step ---
  fprintf(stderr, "Warmup ...\n");
  std::vector<int64_t> warm_tgt = {1};
  auto warm_t = from_blob(warm_tgt.data(), {1, 1},
                          executorch::aten::ScalarType::Long);

  std::vector<EValue> enc_inputs = {EValue(*src)};
  auto enc_out_w = model.execute("encoder", enc_inputs);
  if (!enc_out_w.ok()) {
    fprintf(stderr, "encoder warmup failed: 0x%x\n",
            (int)enc_out_w.error());
    return 1;
  }
  // cross_mask: [1, 1, 1, T_src] of ones (no padding in our synthetic prompt)
  std::vector<uint8_t> cross_mask(enc_len, 1);
  auto cm = from_blob(cross_mask.data(), {1, 1, 1, enc_len},
                      executorch::aten::ScalarType::Bool);

  auto enc_out_t = enc_out_w.get()[0].toTensor();

  // Decoder uses KV cache: signature is (token_id [1,1], input_pos [1],
  // encoder_out, cross_mask). Position 0 starts the cache fresh.
  int64_t warm_pos_val = 0;
  auto warm_pos = from_blob(&warm_pos_val, {1},
                            executorch::aten::ScalarType::Long);
  std::vector<EValue> dec_warm_in = {EValue(*warm_t), EValue(*warm_pos),
                                     EValue(enc_out_t), EValue(*cm)};
  auto dec_w = model.execute("decoder", dec_warm_in);
  if (!dec_w.ok()) {
    fprintf(stderr, "decoder warmup failed: 0x%x\n", (int)dec_w.error());
    return 1;
  }
  warm_pos_val = 1;
  auto dec_w2 = model.execute("decoder", {EValue(*warm_t), EValue(*warm_pos),
                                          EValue(enc_out_t), EValue(*cm)});
  if (!dec_w2.ok()) {
    fprintf(stderr, "decoder warmup #2 failed: 0x%x\n", (int)dec_w2.error());
    return 1;
  }

  // --- Encode (timed) ---
  double t_enc0 = now_ms();
  auto enc_res = model.execute("encoder", enc_inputs);
  double t_enc = now_ms() - t_enc0;
  if (!enc_res.ok()) {
    fprintf(stderr, "encoder failed: 0x%x\n", (int)enc_res.error());
    return 1;
  }
  auto enc_tensor = enc_res.get()[0].toTensor();
  auto enc_stats = stat_tensor(enc_tensor);
  fprintf(stderr,
          "Encoder: %.2f ms, sum=%.4g abs_sum=%.4g finite=%lld/%lld\n",
          t_enc, enc_stats.sum, enc_stats.abs_sum,
          (long long)enc_stats.finite, (long long)enc_stats.total);

  // --- Decode (timed): KV-cached autoregressive decode.
  // Each step feeds (current_token, current_pos) and reads logits for that
  // single token. The first call's input_pos=0 effectively resets the
  // self-attn cache via the dynamic mask; subsequent steps see positions
  // 1, 2, ... and the cache grows.
  int64_t cur_token = 1;  // BOS-ish placeholder
  int64_t pos_val = 0;
  // Reload the model so warmup doesn't pollute the cache state.  Cheaper
  // approach: just step from pos=0 again — KV cache is overwritten in
  // place at input_pos, so as long as we walk positions 0..N-1 in order,
  // earlier warmup writes get rewritten and don't affect numerics.
  pos_val = 0;
  auto pos_tensor = from_blob(&pos_val, {1}, executorch::aten::ScalarType::Long);
  int64_t tok_buf = cur_token;
  auto tok_tensor = from_blob(&tok_buf, {1, 1}, executorch::aten::ScalarType::Long);

  double total_dec = 0.0;
  double last_sum = 0.0;
  int64_t last_token = 0;
  for (int step = 0; step < n_decode; ++step) {
    pos_val = step;
    tok_buf = cur_token;
    std::vector<EValue> dec_in = {EValue(*tok_tensor), EValue(*pos_tensor),
                                  EValue(enc_tensor), EValue(*cm)};
    double t0 = now_ms();
    auto dec_res = model.execute("decoder", dec_in);
    total_dec += now_ms() - t0;
    if (!dec_res.ok()) {
      fprintf(stderr, "decoder step %d failed: 0x%x\n",
              step, (int)dec_res.error());
      return 1;
    }

    auto logits = dec_res.get()[0].toTensor();
    int64_t V = logits.size(2);
    const float* row = logits.template const_data_ptr<float>();
    int64_t argmax = 0;
    float max_v = row[0];
    double sum = row[0];
    for (int64_t v = 1; v < V; ++v) {
      sum += row[v];
      if (row[v] > max_v) { max_v = row[v]; argmax = v; }
    }
    last_sum = sum;
    last_token = argmax;
    if (step <= 2) {
      fprintf(stderr,
              "[debug] step=%d pos=%lld sum=%.4f argmax=%lld first8: "
              "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
              step, (long long)pos_val, sum, (long long)argmax,
              row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]);
    }
    cur_token = argmax;
  }

  double avg_dec = total_dec / n_decode;
  double tok_per_s = 1000.0 / avg_dec;
  fprintf(stdout,
          "Needle benchmark: enc=%.2f ms, decode=%.2f ms/tok (avg over %d), "
          "%.1f tok/s, last_token=%lld, last_logit_sum=%.4g\n",
          t_enc, avg_dec, n_decode, tok_per_s,
          (long long)last_token, last_sum);

  // Emit a stable single-line summary that a Python test can grep for to
  // compare against the eager PyTorch reference.  Format:
  //   NUMERICS: enc_sum=<f> enc_abs_sum=<f> last_token=<i> last_logit_sum=<f>
  fprintf(stdout,
          "NUMERICS: enc_sum=%.6f enc_abs_sum=%.6f last_token=%lld "
          "last_logit_sum=%.6f\n",
          enc_stats.sum, enc_stats.abs_sum,
          (long long)last_token, last_sum);
  return 0;
}
