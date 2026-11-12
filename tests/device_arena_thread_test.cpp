#include "device_arena.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <thread>

static const auto owner = std::this_thread::get_id();
using Setter = decltype(ggml_backend_buffer_i::set_tensor);
static std::map<ggml_backend_buffer_t, Setter> setters;
static bool parallel = false;
static std::atomic<int> worker_uploads{0};
static decltype(ggml_backend_reg_i::get_proc_address) original_proc;
static ggml_backend_buffer_type_t streaming_type(ggml_backend_dev_t dev, bool) {
    return ggml_backend_dev_buffer_type(dev);
}
static void * test_proc(ggml_backend_reg_t reg, const char *name) {
    if (parallel && std::strcmp(name, "ggml_backend_streaming_buffer_type_v1") == 0)
        return reinterpret_cast<void *>(streaming_type);
    return original_proc ? original_proc(reg, name) : nullptr;
}
static void checked_set(ggml_backend_buffer_t b, ggml_tensor * t, const void * data, size_t off, size_t size) {
    const bool worker = std::this_thread::get_id() != owner;
    if (worker != parallel) std::abort();
    if (worker) ++worker_uploads;
    setters.at(b)(b, t, data, off, size);
    if (!parallel) t->extra = reinterpret_cast<void *>(uintptr_t{123});
}
static void watch(ggml_tensor * t) {
    auto b = t->buffer;
    if (setters.emplace(b, b->iface.set_tensor).second) b->iface.set_tensor = checked_set;
}
static void require(bool ok) { if (!ok) std::abort(); }

int main() {
    ggml_backend_load_all();
    auto dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    require(dev != nullptr);
    const char * path = "arena-thread-test.bin";
    std::vector<float> bytes(32 * 32 * 8);
    std::ofstream file(path, std::ios::binary);
    for (int layer = 0; layer < 4; ++layer) {
        std::fill(bytes.begin(), bytes.end(), float(layer + 1));
        file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size() * sizeof(float));
    }
    file.close();
    auto reg = ggml_backend_dev_backend_reg(dev);
    original_proc = reg->iface.get_proc_address;
    reg->iface.get_proc_address = test_proc;
    for (bool mode : {false, true}) {
    parallel = mode;
    for (int slots : {1, 2}) {
        ggml_init_params ip{1024 * 1024, nullptr, true};
        auto ctx = ggml_init(ip);
        std::vector<bmoe::LayerExperts> layers(4);
        std::vector<std::vector<ggml_tensor *>> dense(4);
        for (int i = 0; i < 4; ++i) {
            layers[i].bound = true;
            layers[i].proj[0].tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 32, 8);
            layers[i].proj[0].file_idx = 0;
            layers[i].proj[0].file_off = i * bytes.size() * sizeof(float);
            dense[i].push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32));
        }
        auto host = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev));
        for (int i = 0; i < 4; ++i) std::fill_n(static_cast<float *>(dense[i][0]->data), 32, float(i + 11));
        {
            bmoe::DeviceExpertArena arena;
            std::string error;
            require(arena.init(dev, {path}, layers, 8, false, slots, false, error));
            std::unordered_set<const ggml_tensor *> non_matrix;
            for (auto & d : dense) non_matrix.insert(d[0]);
            require(arena.init_dense(dev, dense, non_matrix, error));
            arena.place(true);
            for (auto & l : layers) watch(l.proj[0].tensor);
            for (auto & d : dense) watch(d[0]);
            for (int graph = 0; graph < 3; ++graph) {
                arena.begin_graph();
                for (int i = 0; i < (graph == 1 ? 1 : 4); ++i) {
                    arena.barrier(i, nullptr);
                    ggml_backend_tensor_get(layers[i].proj[0].tensor, bytes.data(), 0, bytes.size() * sizeof(float));
                    for (float v : bytes) require(v == float(i + 1));
                    float value;
                    ggml_backend_tensor_get(dense[i][0], &value, 0, sizeof(value));
                    require(value == float(i + 11));
                    require(dense[i][0]->extra == (parallel ? nullptr : reinterpret_cast<void *>(uintptr_t{123})));
                    arena.dense_barrier(i);
                }
                arena.end_graph();
                require(!arena.failed());
            }
            for (auto & entry : setters) entry.first->iface.set_tensor = entry.second;
            setters.clear();
        }
        ggml_backend_buffer_free(host);
        ggml_free(ctx);
    }
    }
    require(worker_uploads > 0);
    reg->iface.get_proc_address = original_proc;
    std::remove(path);
    std::puts("serialized and streaming-capability uploads: ownership, bytes, slot reuse and early exit pass");
}
