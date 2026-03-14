/*
 * Live microphone transcription demo using the Parakeet TDT model.
 *
 * Records audio from a USB microphone (ALSA), runs the Parakeet pipeline
 * (preprocessor -> encoder -> TDT greedy decode), and prints the transcript.
 *
 * Usage:
 *   ./parakeet_demo --model /path/to/model.pte --tokenizer /path/to/tokenizer.model
 *                   [--device 1] [--max_seconds 40]
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <alsa/asoundlib.h>
#include <termios.h>
#include <unistd.h>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/tokenizer.h>

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

// ---------------------------------------------------------------------------
// Terminal raw mode for non-blocking keypress detection
// ---------------------------------------------------------------------------
struct RawTerminal {
  struct termios orig;

  RawTerminal() {
    tcgetattr(STDIN_FILENO, &orig);
    struct termios raw = orig;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 1; // 100ms timeout
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
  }
  ~RawTerminal() { tcsetattr(STDIN_FILENO, TCSANOW, &orig); }

  bool key_pressed() {
    char c;
    return read(STDIN_FILENO, &c, 1) == 1;
  }
};

// ---------------------------------------------------------------------------
// ALSA microphone capture
// ---------------------------------------------------------------------------
class MicCapture {
 public:
  MicCapture(int device_index, int sample_rate)
      : handle_(nullptr), sample_rate_(sample_rate) {
    std::string device = "hw:" + std::to_string(device_index);
    int err = snd_pcm_open(&handle_, device.c_str(), SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
      // Fallback: try plughw which handles format conversion
      device = "plughw:" + std::to_string(device_index);
      err = snd_pcm_open(&handle_, device.c_str(), SND_PCM_STREAM_CAPTURE, 0);
    }
    if (err < 0) {
      std::cerr << "Cannot open audio device " << device_index << ": "
                << snd_strerror(err) << std::endl;
      handle_ = nullptr;
      return;
    }

    snd_pcm_hw_params_t* params;
    snd_pcm_hw_params_alloca(&params);
    snd_pcm_hw_params_any(handle_, params);
    snd_pcm_hw_params_set_access(
        handle_, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(handle_, params, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(handle_, params, 1);
    unsigned int rate = static_cast<unsigned int>(sample_rate);
    snd_pcm_hw_params_set_rate_near(handle_, params, &rate, nullptr);
    sample_rate_ = static_cast<int>(rate);

    // ~64ms period for responsive recording
    snd_pcm_uframes_t period = static_cast<snd_pcm_uframes_t>(rate / 16);
    snd_pcm_hw_params_set_period_size_near(handle_, params, &period, nullptr);

    err = snd_pcm_hw_params(handle_, params);
    if (err < 0) {
      std::cerr << "Cannot set audio params: " << snd_strerror(err)
                << std::endl;
      snd_pcm_close(handle_);
      handle_ = nullptr;
    }
  }

  ~MicCapture() {
    if (handle_) {
      snd_pcm_close(handle_);
    }
  }

  bool is_open() const { return handle_ != nullptr; }
  int sample_rate() const { return sample_rate_; }

  // Read a chunk of audio. Returns number of frames read, or -1 on error.
  // Appends normalized float samples to `out`.
  int read_chunk(std::vector<float>& out, int frames) {
    std::vector<int16_t> buf(static_cast<size_t>(frames));
    snd_pcm_sframes_t n = snd_pcm_readi(handle_, buf.data(), frames);
    if (n == -EPIPE) {
      snd_pcm_prepare(handle_);
      return 0;
    }
    if (n < 0) {
      return -1;
    }
    for (snd_pcm_sframes_t i = 0; i < n; ++i) {
      out.push_back(static_cast<float>(buf[static_cast<size_t>(i)]) / 32768.0f);
    }
    return static_cast<int>(n);
  }

 private:
  snd_pcm_t* handle_;
  int sample_rate_;
};

// ---------------------------------------------------------------------------
// Model helper: query int64 constant method
// ---------------------------------------------------------------------------
static int64_t query_int(Module& model, const char* name) {
  std::vector<EValue> empty;
  auto result = model.execute(name, empty);
  if (!result.ok()) {
    std::cerr << "Failed to query metadata: " << name << std::endl;
    return -1;
  }
  return result.get()[0].toInt();
}

static double query_double(Module& model, const char* name) {
  std::vector<EValue> empty;
  auto result = model.execute(name, empty);
  if (!result.ok()) {
    std::cerr << "Failed to query metadata: " << name << std::endl;
    return -1.0;
  }
  return result.get()[0].toDouble();
}

// ---------------------------------------------------------------------------
// Model helper: get expected dtype for a method input
// ---------------------------------------------------------------------------
static ::executorch::aten::ScalarType get_input_dtype(
    Module& model,
    const char* method,
    size_t idx) {
  auto meta = model.method_meta(method);
  if (!meta.ok())
    return ::executorch::aten::ScalarType::Float;
  auto input_meta = meta.get().input_tensor_meta(idx);
  if (input_meta.error() != Error::Ok)
    return ::executorch::aten::ScalarType::Float;
  return input_meta.get().scalar_type();
}

// ---------------------------------------------------------------------------
// TDT greedy decode (same algorithm as upstream parakeet runner)
// ---------------------------------------------------------------------------
static const std::vector<int> DURATIONS = {0, 1, 2, 3, 4};

static std::vector<uint64_t> greedy_decode(
    Module& model,
    const ::executorch::aten::Tensor& f_proj,
    int64_t encoder_len,
    int64_t blank_id,
    int64_t num_rnn_layers,
    int64_t pred_hidden) {
  std::vector<uint64_t> hypothesis;
  size_t proj_dim = static_cast<size_t>(f_proj.sizes()[2]);

  auto h_dtype = get_input_dtype(model, "decoder_step", 1);
  auto c_dtype = get_input_dtype(model, "decoder_step", 2);

  size_t h_elem = ::executorch::runtime::elementSize(h_dtype);
  size_t c_elem = ::executorch::runtime::elementSize(c_dtype);
  size_t num_elements =
      static_cast<size_t>(num_rnn_layers) * static_cast<size_t>(pred_hidden);

  std::vector<uint8_t> h_data(num_elements * h_elem, 0);
  std::vector<uint8_t> c_data(num_elements * c_elem, 0);

  auto h = from_blob(
      h_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers), 1,
       static_cast<::executorch::aten::SizesType>(pred_hidden)},
      h_dtype);
  auto c = from_blob(
      c_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers), 1,
       static_cast<::executorch::aten::SizesType>(pred_hidden)},
      c_dtype);

  // Prime decoder with SOS (= blank_id)
  std::vector<int64_t> sos_data = {blank_id};
  auto sos = from_blob(sos_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);
  auto init = model.execute("decoder_step", std::vector<EValue>{sos, h, c});
  if (!init.ok())
    return hypothesis;

  auto& init_out = init.get();
  std::memcpy(h_data.data(), init_out[1].toTensor().const_data_ptr(), h_data.size());
  std::memcpy(c_data.data(), init_out[2].toTensor().const_data_ptr(), c_data.size());

  auto f_dtype = get_input_dtype(model, "joint", 0);
  auto g_dtype = get_input_dtype(model, "joint", 1);
  size_t f_elem = ::executorch::runtime::elementSize(f_dtype);
  size_t g_elem = ::executorch::runtime::elementSize(g_dtype);
  size_t g_bytes = static_cast<size_t>(init_out[0].toTensor().numel()) * g_elem;
  std::vector<uint8_t> g_data(g_bytes);
  std::memcpy(g_data.data(), init_out[0].toTensor().const_data_ptr(), g_bytes);

  const uint8_t* f_ptr = static_cast<const uint8_t*>(f_proj.const_data_ptr());
  size_t f_t_bytes = proj_dim * f_elem;

  int64_t t = 0;
  int64_t sym_count = 0;
  constexpr int64_t max_sym = 10;

  while (t < encoder_len) {
    std::vector<uint8_t> f_t_data(f_t_bytes);
    std::memcpy(f_t_data.data(), f_ptr + static_cast<size_t>(t) * f_t_bytes, f_t_bytes);

    auto f_t = from_blob(
        f_t_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        f_dtype);
    auto g_proj = from_blob(
        g_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        g_dtype);

    auto joint_result = model.execute("joint", std::vector<EValue>{f_t, g_proj});
    if (!joint_result.ok())
      return hypothesis;

    int64_t k = joint_result.get()[0].toTensor().const_data_ptr<int64_t>()[0];
    int64_t dur_idx = joint_result.get()[1].toTensor().const_data_ptr<int64_t>()[0];
    int64_t dur = DURATIONS[dur_idx];

    if (k == blank_id) {
      t += std::max(dur, int64_t(1));
      sym_count = 0;
    } else {
      hypothesis.push_back(static_cast<uint64_t>(k));

      std::vector<int64_t> tok_data = {k};
      auto tok = from_blob(tok_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);
      auto dec = model.execute("decoder_step", std::vector<EValue>{tok, h, c});
      if (!dec.ok())
        return hypothesis;

      auto& out = dec.get();
      std::memcpy(h_data.data(), out[1].toTensor().const_data_ptr(), h_data.size());
      std::memcpy(c_data.data(), out[2].toTensor().const_data_ptr(), c_data.size());
      std::memcpy(g_data.data(), out[0].toTensor().const_data_ptr(), g_data.size());

      t += dur;
      if (dur == 0) {
        sym_count++;
        if (sym_count >= max_sym) {
          t++;
          sym_count = 0;
        }
      } else {
        sym_count = 0;
      }
    }
  }
  return hypothesis;
}

// ---------------------------------------------------------------------------
// Decode token IDs to text using SentencePiece tokenizer
// ---------------------------------------------------------------------------
static std::string decode_tokens(
    const std::vector<uint64_t>& tokens,
    const tokenizers::Tokenizer& tokenizer) {
  std::string result;
  uint64_t prev = tokenizer.bos_tok();
  for (uint64_t tok : tokens) {
    auto r = tokenizer.decode(prev, tok);
    if (r.ok()) {
      result += r.get();
    }
    prev = tok;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Arg parsing
// ---------------------------------------------------------------------------
struct Args {
  std::string model_path = "/mnt/nvme/parakeet_ggml_fp32/model_f32.pte";
  std::string tokenizer_path = "/mnt/nvme/parakeet_ggml_fp32/tokenizer.model";
  int device_index = 0;
  int max_seconds = 40;
};

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--model" || arg == "-m") && i + 1 < argc)
      a.model_path = argv[++i];
    else if ((arg == "--tokenizer" || arg == "-t") && i + 1 < argc)
      a.tokenizer_path = argv[++i];
    else if ((arg == "--device" || arg == "-d") && i + 1 < argc)
      a.device_index = std::stoi(argv[++i]);
    else if (arg == "--max_seconds" && i + 1 < argc)
      a.max_seconds = std::stoi(argv[++i]);
    else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n"
          << "  --model, -m PATH       Model .pte file (default: "
          << a.model_path << ")\n"
          << "  --tokenizer, -t PATH   Tokenizer .model file\n"
          << "  --device, -d INDEX     ALSA device index (default: 0)\n"
          << "  --max_seconds N        Max recording duration (default: 40)\n";
      exit(0);
    }
  }
  return a;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  // Load model
  std::cout << "Loading model: " << args.model_path << std::endl;
  Module model(args.model_path, Module::LoadMode::Mmap);
  auto err = model.load();
  if (err != Error::Ok) {
    std::cerr << "Failed to load model." << std::endl;
    return 1;
  }

  // Pre-load all methods
  for (const char* method :
       {"preprocessor", "encoder", "decoder_step", "joint"}) {
    if (model.load_method(method) != Error::Ok) {
      std::cerr << "Failed to load method: " << method << std::endl;
      return 1;
    }
  }
  std::cout << "Model loaded." << std::endl;

  // Query model metadata
  int64_t num_rnn_layers = query_int(model, "num_rnn_layers");
  int64_t pred_hidden = query_int(model, "pred_hidden");
  int64_t blank_id = query_int(model, "blank_id");
  int64_t sample_rate = query_int(model, "sample_rate");
  double window_stride = query_double(model, "window_stride");
  int64_t encoder_sub = query_int(model, "encoder_subsampling_factor");

  if (sample_rate <= 0 || num_rnn_layers <= 0) {
    std::cerr << "Invalid model metadata." << std::endl;
    return 1;
  }

  int max_samples = args.max_seconds * static_cast<int>(sample_rate);
  std::cout << "Sample rate: " << sample_rate
            << "  Max recording: " << args.max_seconds << "s ("
            << max_samples << " samples)" << std::endl;

  // Load tokenizer
  std::cout << "Loading tokenizer: " << args.tokenizer_path << std::endl;
  auto tokenizer =
      ::executorch::extension::llm::load_tokenizer(args.tokenizer_path);
  if (!tokenizer || !tokenizer->is_loaded()) {
    std::cerr << "Failed to load tokenizer." << std::endl;
    return 1;
  }

  // Open microphone
  MicCapture mic(args.device_index, static_cast<int>(sample_rate));
  if (!mic.is_open()) {
    std::cerr << "Failed to open microphone (device " << args.device_index
              << "). Try a different --device index." << std::endl;
    return 1;
  }
  std::cout << "Microphone opened (device " << args.device_index
            << ", rate " << mic.sample_rate() << " Hz)." << std::endl;

  RawTerminal term;

  // Main loop
  while (true) {
    std::cout << "\n--- Press SPACE to start recording (q to quit) ---"
              << std::endl;

    // Wait for space or q
    bool quit = false;
    while (true) {
      char c;
      if (read(STDIN_FILENO, &c, 1) == 1) {
        if (c == 'q' || c == 'Q') {
          quit = true;
          break;
        }
        if (c == ' ') {
          break;
        }
      }
    }
    if (quit)
      break;

    // Record
    std::cout << "Recording... (press SPACE to stop, max "
              << args.max_seconds << "s)" << std::endl;

    std::vector<float> audio;
    audio.reserve(static_cast<size_t>(max_samples));
    int chunk_frames = static_cast<int>(sample_rate) / 16; // ~62ms chunks

    auto rec_start = std::chrono::steady_clock::now();

    while (true) {
      // Check for keypress to stop
      char c;
      if (read(STDIN_FILENO, &c, 1) == 1 && c == ' ') {
        break;
      }

      // Check max duration
      if (static_cast<int>(audio.size()) >= max_samples) {
        std::cout << "Max duration reached." << std::endl;
        break;
      }

      // Read audio
      int remaining = max_samples - static_cast<int>(audio.size());
      int to_read = std::min(chunk_frames, remaining);
      int n = mic.read_chunk(audio, to_read);
      if (n < 0) {
        std::cerr << "Audio read error." << std::endl;
        break;
      }
    }

    auto rec_end = std::chrono::steady_clock::now();
    double rec_secs =
        std::chrono::duration<double>(rec_end - rec_start).count();

    if (audio.empty()) {
      std::cout << "No audio recorded." << std::endl;
      continue;
    }

    std::cout << "Recorded " << audio.size() << " samples ("
              << std::fixed << std::setprecision(1) << rec_secs << "s)"
              << std::endl;

    // Run inference
    std::cout << "Transcribing..." << std::endl;
    auto inf_start = std::chrono::steady_clock::now();

    // Preprocessor
    auto audio_tensor = from_blob(
        audio.data(),
        {static_cast<::executorch::aten::SizesType>(audio.size())},
        ::executorch::aten::ScalarType::Float);
    std::vector<int64_t> audio_len_data = {static_cast<int64_t>(audio.size())};
    auto audio_len = from_blob(
        audio_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

    auto proc = model.execute(
        "preprocessor", std::vector<EValue>{audio_tensor, audio_len});
    if (!proc.ok()) {
      std::cerr << "Preprocessor failed." << std::endl;
      continue;
    }
    auto mel = proc.get()[0].toTensor();
    int64_t mel_len = proc.get()[1].toTensor().const_data_ptr<int64_t>()[0];

    // Encoder
    std::vector<int64_t> mel_len_data = {mel_len};
    auto mel_len_t = from_blob(
        mel_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);
    auto enc = model.execute(
        "encoder", std::vector<EValue>{mel, mel_len_t});
    if (!enc.ok()) {
      std::cerr << "Encoder failed." << std::endl;
      continue;
    }
    auto f_proj = enc.get()[0].toTensor();
    int64_t encoded_len =
        enc.get()[1].toTensor().const_data_ptr<int64_t>()[0];

    // Decode
    auto tokens = greedy_decode(
        model, f_proj, encoded_len, blank_id, num_rnn_layers, pred_hidden);

    auto inf_end = std::chrono::steady_clock::now();
    double inf_ms =
        std::chrono::duration<double, std::milli>(inf_end - inf_start).count();

    // Convert to text
    std::string text = decode_tokens(tokens, *tokenizer);

    std::cout << "\n>>> " << text << std::endl;
    std::cout << "[" << tokens.size() << " tokens, " << std::fixed
              << std::setprecision(0) << inf_ms << " ms inference]"
              << std::endl;
  }

  std::cout << "Goodbye." << std::endl;
  return 0;
}
