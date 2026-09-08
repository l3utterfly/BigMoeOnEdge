// Optional llama.cpp embedding seam. The policy-only session.h does not import llama.h.
#pragma once

#include "bmoe/session.h"
#include "llama.h"

namespace bmoe {

struct LlamaSessionConfig {
    // The host owns process-global backend lifetime; a session must not free it.
    bool backend_initialized = false;
    // Called before installing the streaming callback. Do not change context/batch sizes:
    // those are supplied by SessionConfig and are also used by the generation loop.
    std::function<void(llama_context_params &)> configure_context;
};

struct LlamaSessionAccess {
    // Borrowed, valid for the session lifetime. No concurrent use with generate().
    static llama_context * context(Session &session);
    // Raw-prompt sessions only. Cache files contain sequence 0 and its exact token prefix.
    static bool save_cache(Session &session, const std::string &path);
    static bool restore_cache(Session &session, const std::string &path);
};

} // namespace bmoe
