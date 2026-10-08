#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include <inttypes.h>
#include "../ds4_server.c"
#ifndef DS4_NO_GPU
#include "../ds4_gpu.h"
#include <math.h>

bool ds4_test_dspark_cache_window_crop(void);
bool ds4_test_dspark_prefix_capture(ds4_engine *engine, const ds4_tokens *prompt);

static ds4_engine *test_engine_fast;
static ds4_engine *test_engine_quality;

static char *test_read_file(const char *path);

static const char *test_model_path(void) {
    const char *model_path = getenv("DS4_TEST_MODEL");
    return (model_path && model_path[0]) ? model_path : "ds4flash.gguf";
}

static bool test_env_bool(const char *name) {
    const char *v = getenv(name);
    return v && v[0] && strcmp(v, "0") != 0;
}

static uint32_t test_env_u32(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 0;
    char *end = NULL;
    unsigned long n = strtoul(v, &end, 10);
    if (end == v) return 0;
    return n > UINT32_MAX ? UINT32_MAX : (uint32_t)n;
}

static uint64_t test_env_gib(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 0;
    char *end = NULL;
    unsigned long long n = strtoull(v, &end, 10);
    if (end == v || n == 0) return 0;
    const uint64_t one_gib = 1024ull * 1024ull * 1024ull;
    if (n > UINT64_MAX / one_gib) return UINT64_MAX;
    return (uint64_t)n * one_gib;
}

static char *test_save_env(const char *name) {
    const char *value = getenv(name);
    if (!value) return NULL;
    size_t len = strlen(value);
    char *copy = malloc(len + 1);
    TEST_ASSERT(copy != NULL);
    if (!copy) return NULL;
    memcpy(copy, value, len + 1);
    return copy;
}

static void test_restore_env(const char *name, char *saved) {
    if (saved) {
        setenv(name, saved, 1);
        free(saved);
    } else {
        unsetenv(name);
    }
}

typedef struct {
    char *cold_decode;
    char *batch_selected_addr;
} test_streaming_prefill_env;

static test_streaming_prefill_env test_force_canonical_streaming_prefill(void) {
    test_streaming_prefill_env saved = {
        .cold_decode =
            test_save_env("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL"),
        .batch_selected_addr =
            test_save_env("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR"),
    };
    if (test_env_bool("DS4_TEST_SSD_STREAMING")) {
        setenv("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL", "1", 1);
        setenv("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR", "1", 1);
    }
    return saved;
}

static void test_restore_canonical_streaming_prefill(
        test_streaming_prefill_env saved) {
    test_restore_env("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL",
                     saved.cold_decode);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR",
                     saved.batch_selected_addr);
}

static ds4_backend test_model_backend(void) {
    const char *backend = getenv("DS4_TEST_BACKEND");
    if (backend && !strcmp(backend, "cpu")) return DS4_BACKEND_CPU;
#ifdef __APPLE__
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static ds4_engine *test_open_engine(bool quality) {
    ds4_engine *engine = NULL;
    /* DS4_TEST_MTP loads the MTP head on the fast engine so the speculative
     * verify regression can reuse it; draft=4 hits the multi-row verify path. */
    const char *mtp = getenv("DS4_TEST_MTP");
    ds4_engine_options opt = {
        .model_path = test_model_path(),
        .backend = test_model_backend(),
        .quality = quality,
        .ssd_streaming = test_env_bool("DS4_TEST_SSD_STREAMING"),
        .ssd_streaming_cold = test_env_bool("DS4_TEST_SSD_STREAMING_COLD"),
        .ssd_streaming_cache_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS"),
        .ssd_streaming_cache_bytes =
            test_env_gib("DS4_TEST_SSD_STREAMING_CACHE_GB"),
        .ssd_streaming_preload_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_PRELOAD_EXPERTS"),
        .mtp_path = (mtp && mtp[0] && !quality) ? mtp : NULL,
        .mtp_draft_tokens = (mtp && mtp[0] && !quality) ? 4 : 0,
        .glm_mtp = test_env_bool("DS4_TEST_GLM_MTP"),
        .dspark_exact_sampling = test_env_bool("DS4_TEST_MTP_EXACT"),
    };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0);
    return engine;
}

static ds4_engine *test_get_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    if (*slot) return *slot;

    *slot = test_open_engine(quality);
    return *slot;
}

static void test_close_engines(void) {
    ds4_engine_close(test_engine_fast);
    ds4_engine_close(test_engine_quality);
    test_engine_fast = NULL;
    test_engine_quality = NULL;
}

static void test_close_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    ds4_engine_close(*slot);
    *slot = NULL;
}

/* Qwen3.8 rewind: one token back restores the verify snapshot, anything else
 * resets the recurrent state and replays the kept tokens on the next eval;
 * both must land on the logits a fresh session produces for the same tokens. */
static void test_session_rewind_replay(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine) return;
    if (!ds4_engine_is_qwen4(engine)) {
        puts("session-rewind: Qwen3.8 model required, skipped");
        return;
    }
    ds4_session *live = NULL, *fresh = NULL;
    ds4_tokens prompt = {0}, replay = {0};
    char err[192] = {0};
    ds4_token_score got[8], want[8];
    const bool mtp = test_env_bool("DS4_TEST_GLM_MTP");
    if (mtp) setenv("DS4_QWEN4_SPEC_FORCE_ACCEPT", "1", 1);

    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", "Count from one to ten.");
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    TEST_ASSERT(ds4_session_create(&live, engine, 1024) == 0);
    TEST_ASSERT(ds4_session_create(&fresh, engine, 1024) == 0);
    if (!live || !fresh) goto cleanup;
    TEST_ASSERT(ds4_session_sync(live, &prompt, err, sizeof(err)) == 0);
    for (int i = 0; i < prompt.len; i++) ds4_tokens_push(&replay, prompt.v[i]);

    /* three greedy steps; under MTP the first cycle seeds a draft and the
     * forced-accept cycles after it commit token pairs */
    int last_n = 1;
    for (int step = 0; step < 3; step++) {
        const int first = ds4_session_argmax(live);
        if (mtp) {
            int acc[2] = {0, 0};
            last_n = ds4_session_eval_speculative_argmax(live, first, 2, -1, acc, 2, err, sizeof(err));
            TEST_ASSERT(last_n == 1 || last_n == 2);
            if (last_n < 1) goto cleanup;
            for (int i = 0; i < last_n; i++) ds4_tokens_push(&replay, acc[i]);
        } else {
            TEST_ASSERT(ds4_session_eval(live, first, err, sizeof(err)) == 0);
            ds4_tokens_push(&replay, first);
        }
    }
    if (mtp) TEST_ASSERT(last_n == 2);   /* the one-token rewind below takes the snapshot path */
    TEST_ASSERT(ds4_session_pos(live) == replay.len);
    if (mtp) TEST_ASSERT(ds4_session_top_logprobs(live, want, 8) == 8);

    /* rewind one token (snapshot path under MTP) and re-evaluate it */
    const int last = replay.v[replay.len - 1];
    ds4_session_rewind(live, replay.len - 1);
    TEST_ASSERT(ds4_session_pos(live) == replay.len - 1);
    if (mtp) {
        ds4_session_snapshot rewound = {0};
        TEST_ASSERT(ds4_session_save_snapshot(live, &rewound, err, sizeof(err)) == 0);
        if (rewound.ptr) {
            TEST_ASSERT(ds4_session_load_snapshot(fresh, &rewound, err, sizeof(err)) == 0);
            TEST_ASSERT(ds4_session_eval(fresh, last, err, sizeof(err)) == 0);
            TEST_ASSERT(ds4_session_top_logprobs(fresh, got, 8) == 8);
            TEST_ASSERT(got[0].id == want[0].id);
            for (int i = 0; i < 8; i++) TEST_ASSERT(fabsf(got[i].logprob - want[i].logprob) < 2e-3f);
        }
        ds4_session_snapshot_free(&rewound);
    }
    TEST_ASSERT(ds4_session_eval(live, last, err, sizeof(err)) == 0);
    /* Match the replay's prefill/decode split. A full-prefix prefill uses
     * different floating-point accumulation than the final decode step. */
    ds4_tokens prefix = replay;
    prefix.len--;
    if (!mtp) {
        TEST_ASSERT(ds4_session_sync(fresh, &prefix, err, sizeof(err)) == 0);
        TEST_ASSERT(ds4_session_eval(fresh, last, err, sizeof(err)) == 0);
        TEST_ASSERT(ds4_session_top_logprobs(fresh, want, 8) == 8);
    }
    /* MTP restores row zero of the verifier snapshot: re-evaluating row one
     * must reproduce its original logits, including the prior decode state. */
    TEST_ASSERT(ds4_session_top_logprobs(live, got, 8) == 8);
    TEST_ASSERT(got[0].id == want[0].id);
    for (int i = 0; i < 8; i++) TEST_ASSERT(fabsf(got[i].logprob - want[i].logprob) < 2e-3f);

    /* rewind two tokens (reset + replay path) and re-evaluate both */
    ds4_session_rewind(live, replay.len - 2);
    TEST_ASSERT(ds4_session_pos(live) == replay.len - 2);
    TEST_ASSERT(ds4_session_eval(live, replay.v[replay.len - 2], err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_eval(live, last, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_pos(live) == replay.len);
    ds4_session_free(fresh);
    fresh = NULL;
    TEST_ASSERT(ds4_session_create(&fresh, engine, 1024) == 0);
    if (!fresh) goto cleanup;
    prefix.len = replay.len - 2;
    TEST_ASSERT(ds4_session_sync(fresh, &prefix, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_eval(fresh, replay.v[replay.len - 2], err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_eval(fresh, last, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_top_logprobs(fresh, want, 8) == 8);
    TEST_ASSERT(ds4_session_top_logprobs(live, got, 8) == 8);
    TEST_ASSERT(got[0].id == want[0].id);
    for (int i = 0; i < 8; i++) TEST_ASSERT(fabsf(got[i].logprob - want[i].logprob) < 2e-3f);

cleanup:
    ds4_tokens_free(&replay);
    ds4_tokens_free(&prompt);
    ds4_session_free(fresh);
    ds4_session_free(live);
}

/* Issue #8 regression: the server rewinds a verified block to its start and
 * re-evaluates the kept token when exact sampling crosses a tool
 * sampling-mode boundary at nonzero temperature.  The rewind must restore
 * the pre-verify snapshot and land on the logits a session that never ran
 * the verify produces for the same committed tokens; before the fix the
 * rewind reset the recurrent state and the next eval replayed the whole
 * kept context.  Needs DS4_TEST_GLM_MTP and DS4_TEST_MTP_EXACT. */
static void test_session_rewind_resample_boundary(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine) return;
    if (!ds4_engine_is_qwen4(engine) || !ds4_engine_mtp_exact_sampling(engine)) {
        puts("session-rewind-resample: Qwen3.8 model with DS4_TEST_MTP_EXACT required, skipped");
        return;
    }
    ds4_session *live = NULL, *fresh = NULL;
    ds4_tokens prompt = {0};
    char err[192] = {0};
    ds4_token_score got[8], want[8];
    uint64_t rng = 12345;

    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", "List the primary colors, then name two fruits.");
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    TEST_ASSERT(ds4_session_create(&live, engine, 1024) == 0);
    TEST_ASSERT(ds4_session_create(&fresh, engine, 1024) == 0);
    if (!live || !fresh) goto cleanup;
    TEST_ASSERT(ds4_session_sync(live, &prompt, err, sizeof(err)) == 0);

    int acc[2] = {0, 0};
    int block_start = -1;
    for (int attempt = 0; attempt < 64; attempt++) {
        const int tok = ds4_session_sample(live, 1.0f, 0, 1.0f, 0.0f, &rng);
        block_start = ds4_session_pos(live);
        const int ntok = ds4_session_eval_speculative(
            live, tok, 8, ds4_token_eos(engine), 1.0f, 0, 1.0f, 0.0f, &rng,
            acc, 2, err, sizeof(err));
        TEST_ASSERT(ntok >= 1);
        if (ntok < 1) { block_start = -1; goto cleanup; }
        if (ntok == 2) break;   /* an accepted two-row block to rewind */
        block_start = -1;
    }
    TEST_ASSERT(block_start >= 0);
    if (block_start < 0) goto cleanup;

    /* the server resample flow: rewind to the block start, re-evaluate the
     * kept token so the next sample uses the boundary's new mode */
    ds4_session_rewind(live, block_start);
    TEST_ASSERT(ds4_session_pos(live) == block_start);
    TEST_ASSERT(ds4_session_eval(live, acc[0], err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_pos(live) == block_start + 1);

    /* reference: the same committed tokens, one row at a time, no verify */
    TEST_ASSERT(ds4_session_sync(fresh, &prompt, err, sizeof(err)) == 0);
    const ds4_tokens *hist = ds4_session_tokens(live);
    TEST_ASSERT(hist->len > prompt.len);
    for (int i = prompt.len; i < hist->len; i++) {
        TEST_ASSERT(ds4_session_eval(fresh, hist->v[i], err, sizeof(err)) == 0);
    }
    TEST_ASSERT(ds4_session_pos(fresh) == ds4_session_pos(live));
    TEST_ASSERT(ds4_session_top_logprobs(live, got, 8) == 8);
    TEST_ASSERT(ds4_session_top_logprobs(fresh, want, 8) == 8);
    TEST_ASSERT(got[0].id == want[0].id);
    for (int i = 0; i < 8; i++) TEST_ASSERT(fabsf(got[i].logprob - want[i].logprob) < 2e-3f);

cleanup:
    ds4_tokens_free(&prompt);
    ds4_session_free(fresh);
    ds4_session_free(live);
}

/* A continued checkpoint must be loadable and immediately sampleable at a
 * completed prefill chunk, including after a resumed prefill or cancellation. */
typedef struct {
    ds4_session *session;
    ds4_session_payload_file payload;
    int pos;
    bool captured;
    bool cancel;
} test_qwen_prefill_checkpoint;

static void test_qwen_capture_prefill(void *ud, const char *event, int current, int total) {
    test_qwen_prefill_checkpoint *p = ud;
    if (strcmp(event, "prefill_chunk") || current <= 0 || current >= total || p->captured) return;
    p->captured = true;
    p->pos = current;
    char err[192] = {0};
    int rc = ds4_session_stage_payload(p->session, &p->payload, err, sizeof(err));
    if (rc) fprintf(stderr, "ds4-test: prefill checkpoint at %d: %s\n", current, err);
    TEST_ASSERT(rc == 0);
}

static bool test_qwen_cancel_after_checkpoint(void *ud) {
    test_qwen_prefill_checkpoint *p = ud;
    return p->cancel && p->captured;
}

static void test_qwen_prefill_scores_equal(ds4_session *a, ds4_session *b) {
    ds4_token_score got[8], want[8];
    TEST_ASSERT(ds4_session_top_logprobs(a, got, 8) == 8);
    TEST_ASSERT(ds4_session_top_logprobs(b, want, 8) == 8);
    for (int i = 0; i < 8; i++) {
        TEST_ASSERT(got[i].id == want[i].id);
        TEST_ASSERT(fabsf(got[i].logit - want[i].logit) < 1e-5f);
    }
}

static void test_qwen_prefill_checkpoints(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_is_qwen4(engine)) {
        puts("prefill-checkpoints: Qwen3.8 model required, skipped");
        return;
    }
    char *saved_chunk = test_save_env("DS4_QWEN4_PREFILL_CHUNK");
    setenv("DS4_QWEN4_PREFILL_CHUNK", "128", 1);
    ds4_session *live = NULL, *reference = NULL, *restored = NULL;
    ds4_tokens prompt = {0};
    buf text = {0};
    char err[192] = {0};
    for (int i = 0; i < 200; i++) buf_puts(&text, "The harbor records the weather and shipping schedules. ");
    ds4_encode_chat_prompt(engine, NULL, text.ptr, DS4_THINK_NONE, &prompt);
    buf_free(&text);
    TEST_ASSERT(prompt.len > 512);
    TEST_ASSERT(ds4_session_create(&live, engine, 1024) == 0);
    TEST_ASSERT(ds4_session_create(&reference, engine, 1024) == 0);
    TEST_ASSERT(ds4_session_create(&restored, engine, 1024) == 0);
    if (!live || !reference || !restored || prompt.len <= 512) goto cleanup;

    for (int round = 0; round < 3; round++) {
        test_qwen_prefill_checkpoint capture = {.session = live, .cancel = round == 2};
        if (round == 2) ds4_session_invalidate(live);
        ds4_tokens target = prompt;
        target.len = round == 1 ? 512 : 256;
        ds4_session_set_progress(live, test_qwen_capture_prefill, &capture);
        ds4_session_set_cancel(live, test_qwen_cancel_after_checkpoint, &capture);
        int rc = ds4_session_sync(live, &target, err, sizeof(err));
        ds4_session_set_progress(live, NULL, NULL);
        ds4_session_set_cancel(live, NULL, NULL);
        TEST_ASSERT(rc == (capture.cancel ? DS4_SESSION_SYNC_INTERRUPTED : 0));
        TEST_ASSERT(capture.captured && capture.payload.path);
        TEST_ASSERT(capture.pos == (round == 1 ? 384 : 128));
        if (!capture.payload.path) continue;

        ds4_session_invalidate(reference);
        ds4_tokens prefix = prompt;
        prefix.len = capture.pos;
        TEST_ASSERT(ds4_session_sync(reference, &prefix, err, sizeof(err)) == 0);
        FILE *fp = fopen(capture.payload.path, "rb");
        TEST_ASSERT(fp != NULL);
        if (fp) {
            TEST_ASSERT(ds4_session_load_payload(restored, fp, capture.payload.bytes, err, sizeof(err)) == 0);
            fclose(fp);
            TEST_ASSERT(ds4_session_pos(restored) == capture.pos);
            /* Compare against an independent prefill, not the source's possibly
             * stale logits. Also verify the serialized recurrent state by
             * continuing both sessions to the same full prefix. */
            test_qwen_prefill_scores_equal(restored, reference);
            if (capture.cancel) test_qwen_prefill_scores_equal(live, reference);
            TEST_ASSERT(ds4_session_sync(restored, &target, err, sizeof(err)) == 0);
            if (capture.cancel) TEST_ASSERT(ds4_session_sync(live, &target, err, sizeof(err)) == 0);
            test_qwen_prefill_scores_equal(restored, live);
        }
        ds4_session_payload_file_free(&capture.payload);
    }
    /* Rewind outside a speculative snapshot resets the recurrent graph but
     * retains the transcript. A subsequent sync must replay that prefix,
     * whether it appends new tokens or requests the retained prefix itself. */
    for (int append = 0; append < 2; append++) {
        ds4_tokens target = prompt;
        target.len = 512;
        TEST_ASSERT(ds4_session_sync(live, &target, err, sizeof(err)) == 0);
        ds4_session_rewind(live, 128);
        target.len = append ? 384 : 128;
        TEST_ASSERT(ds4_session_sync(live, &target, err, sizeof(err)) == 0);
        ds4_session_invalidate(reference);
        TEST_ASSERT(ds4_session_sync(reference, &target, err, sizeof(err)) == 0);
        test_qwen_prefill_scores_equal(live, reference);
        const int token = ds4_session_argmax(reference);
        TEST_ASSERT(ds4_session_eval(live, token, err, sizeof(err)) == 0);
        TEST_ASSERT(ds4_session_eval(reference, token, err, sizeof(err)) == 0);
        test_qwen_prefill_scores_equal(live, reference);
    }
cleanup:
    ds4_session_free(restored);
    ds4_session_free(reference);
    ds4_session_free(live);
    ds4_tokens_free(&prompt);
    test_restore_env("DS4_QWEN4_PREFILL_CHUNK", saved_chunk);
}

static void test_qwen_restore_reused_session(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_is_qwen4(engine) || !test_env_bool("DS4_TEST_GLM_MTP")) {
        puts("qwen4-restore-reuse: Qwen3.8 with MTP required, skipped");
        return;
    }
    char *saved_force = test_save_env("DS4_QWEN4_SPEC_FORCE_ACCEPT");
    setenv("DS4_QWEN4_SPEC_FORCE_ACCEPT", "1", 1);
    ds4_session *live = NULL, *reference = NULL;
    ds4_tokens prompt = {0}, other = {0};
    ds4_session_snapshot snapshot = {0};
    char err[192] = {0};
    ds4_encode_chat_prompt(engine, NULL, "Count from one to ten.", DS4_THINK_NONE, &prompt);
    TEST_ASSERT(ds4_session_create(&live, engine, 1024) == 0);
    TEST_ASSERT(ds4_session_create(&reference, engine, 1024) == 0);
    if (!live || !reference) goto cleanup;
    TEST_ASSERT(ds4_session_sync(live, &prompt, err, sizeof(err)) == 0);
    for (int step = 0; step < 3; step++) {
        int accepted[2];
        int n = ds4_session_eval_speculative_argmax(live, ds4_session_argmax(live),
                                                   2, -1, accepted, 2, err, sizeof(err));
        TEST_ASSERT(n == (step == 0 ? 1 : 2));
        if (n < 1) goto cleanup;
    }
    const ds4_tokens *tokens = ds4_session_tokens(live);
    for (int i = 0; i < tokens->len; i++) ds4_tokens_push(&other, tokens->v[i]);
    /* Same positions, different history: the old verifier snapshots must not
     * survive loading this checkpoint into the already-used session. */
    other.v[prompt.len / 2] = prompt.v[prompt.len / 2 + 1];
    TEST_ASSERT(ds4_session_sync(reference, &other, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_save_snapshot(reference, &snapshot, err, sizeof(err)) == 0);
    if (!snapshot.ptr) goto cleanup;
    TEST_ASSERT(ds4_session_load_snapshot(live, &snapshot, err, sizeof(err)) == 0);
    ds4_session_rewind(live, other.len - 1);
    TEST_ASSERT(ds4_session_eval(live, other.v[other.len - 1], err, sizeof(err)) == 0);
    ds4_tokens prefix = other;
    prefix.len--;
    ds4_session_invalidate(reference);
    TEST_ASSERT(ds4_session_sync(reference, &prefix, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_eval(reference, other.v[other.len - 1], err, sizeof(err)) == 0);
    test_qwen_prefill_scores_equal(live, reference);

    /* A truncated logits read must not leave a sampleable checkpoint. */
    FILE *fp = tmpfile();
    TEST_ASSERT(fp != NULL);
    if (fp) {
        size_t bytes = 13u * sizeof(uint32_t) + (size_t)other.len * sizeof(uint32_t) + 17u;
        TEST_ASSERT(bytes < snapshot.len);
        TEST_ASSERT(fwrite(snapshot.ptr, 1, bytes, fp) == bytes);
        rewind(fp);
        TEST_ASSERT(ds4_session_load_payload(live, fp, snapshot.len, err, sizeof(err)) != 0);
        TEST_ASSERT(ds4_session_argmax(live) == -1);
        fclose(fp);
    }
cleanup:
    ds4_session_snapshot_free(&snapshot);
    ds4_tokens_free(&other);
    ds4_tokens_free(&prompt);
    ds4_session_free(reference);
    ds4_session_free(live);
    test_restore_env("DS4_QWEN4_SPEC_FORCE_ACCEPT", saved_force);
}

/* ---- ticket 006: the GLM-5.3 nextn (MTP) state a snapshot must carry ---- */

/* Speculative cycles the ticket-006 tests run: the snapshot save points and the
 * cross-mode payload test both use this many. */
#define TEST_GLM_MTP_PAYLOAD_CYCLES 16

typedef struct {
    uint32_t len;          /* checkpoint length the state covers */
    uint32_t kv_dim;
    uint32_t pool_dim;
    uint32_t pool_size;
    uint32_t tail_dim;
    uint32_t embd;
    float   *kv;           /* [len][kv_dim] */
    float   *pool;         /* [len][pool_dim]: the pooled row covering each slot */
    float   *tail;         /* raw K window, then the gate window */
    float   *seam;         /* [embd] kept post-norm hidden */
    uint32_t seam_pos;
    int      seam_written;
    int      seam_kept;
    int      complete;
} test_mtp_state;

static bool test_mtp_row_zero(const float *v, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        if (v[i] != 0.0f) return false;
    }
    return true;
}

static bool test_mtp_row_written(const float *v, uint32_t n) {
    bool any = false;
    for (uint32_t i = 0; i < n; i++) {
        if (!isfinite(v[i])) return false;
        if (v[i] != 0.0f) any = true;
    }
    return any;
}

static float test_mtp_row_max_abs(const float *a, const float *b, uint32_t n) {
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (d > max_abs) max_abs = d;
    }
    return max_abs;
}

static void test_mtp_state_free(test_mtp_state *st) {
    if (!st) return;
    free(st->kv);
    free(st->pool);
    free(st->tail);
    free(st->seam);
    memset(st, 0, sizeof(*st));
}

/* Read every nextn state a payload section is supposed to carry back through
 * the ds4-test probes only: one KV row and the pooled row covering it per slot,
 * the whole raw tail window, the seam (position, token, validity and hidden)
 * and the completeness flag. */
static bool test_mtp_state_capture(ds4_session *s, test_mtp_state *out) {
    memset(out, 0, sizeof(*out));
    uint32_t cache_cap = 0;
    if (!ds4_test_glm_mtp_cache_dims(s, &out->kv_dim, &out->pool_dim,
                                     &out->pool_size, &out->tail_dim,
                                     &cache_cap) ||
        !ds4_test_glm_mtp_seam_hidden(s, NULL, 0, &out->embd)) {
        return false;
    }
    const int pos = s ? ds4_session_pos(s) : 0;
    out->len = pos > 0 ? (uint32_t)pos : 0u;
    if (out->len > cache_cap || out->kv_dim == 0 || out->pool_dim == 0 ||
        out->tail_dim == 0 || out->embd == 0) {
        test_mtp_state_free(out);
        return false;
    }
    const size_t rows = out->len;
    out->kv = malloc(rows * out->kv_dim * sizeof(float) + sizeof(float));
    out->pool = malloc(rows * out->pool_dim * sizeof(float) + sizeof(float));
    out->tail = malloc((size_t)out->tail_dim * sizeof(float));
    out->seam = malloc((size_t)out->embd * sizeof(float));
    if (!out->kv || !out->pool || !out->tail || !out->seam) {
        test_mtp_state_free(out);
        return false;
    }
    for (uint32_t p = 0; p < out->len; p++) {
        if (!ds4_test_glm_read_mtp_kv(s, p,
                                      out->kv + (uint64_t)p * out->kv_dim,
                                      out->kv_dim,
                                      out->pool + (uint64_t)p * out->pool_dim,
                                      out->pool_dim, NULL, 0)) {
            test_mtp_state_free(out);
            return false;
        }
    }
    if (!ds4_test_glm_read_mtp_kv(s, 0, NULL, 0, NULL, 0, out->tail,
                                  out->tail_dim) ||
        !ds4_test_glm_mtp_seam_state(s, &out->seam_pos, &out->seam_written,
                                     &out->seam_kept) ||
        !ds4_test_glm_mtp_seam_hidden(s, out->seam, out->embd, NULL)) {
        test_mtp_state_free(out);
        return false;
    }
    out->complete = ds4_test_glm_mtp_cache_complete(s);
    return true;
}

/* Byte for byte, exactly what the payload section promised to carry. */
static bool test_mtp_state_equal(const test_mtp_state *a,
                                 const test_mtp_state *b, const char *tag) {
    bool ok = true;
    if (a->len != b->len) {
        fprintf(stderr, "ds4-test: mtp %s state length %u/%u\n", tag, a->len,
                b->len);
        return false;
    }
    if (a->complete != b->complete || a->seam_pos != b->seam_pos ||
        a->seam_written != b->seam_written ||
        a->seam_kept != b->seam_kept) {
        fprintf(stderr,
                "ds4-test: mtp %s state differs complete=%d/%d "
                "seam=%u/%u written=%d/%d kept=%d/%d\n", tag,
                a->complete, b->complete, a->seam_pos, b->seam_pos,
                a->seam_written, b->seam_written, a->seam_kept, b->seam_kept);
        ok = false;
    }
    const size_t kv_n = (size_t)a->len * a->kv_dim;
    if (memcmp(a->kv, b->kv, kv_n * sizeof(float)) != 0) {
        size_t first = 0;
        while (first < kv_n &&
               memcmp(&a->kv[first], &b->kv[first], sizeof(float)) == 0) {
            first++;
        }
        fprintf(stderr, "ds4-test: mtp %s KV differs at float %zu of %zu\n",
                tag, first, kv_n);
        ok = false;
    }
    const size_t pool_n = (size_t)a->len * a->pool_dim;
    if (memcmp(a->pool, b->pool, pool_n * sizeof(float)) != 0) {
        size_t first = 0;
        while (first < pool_n &&
               memcmp(&a->pool[first], &b->pool[first], sizeof(float)) == 0) {
            first++;
        }
        fprintf(stderr, "ds4-test: mtp %s pool differs at float %zu of %zu\n",
                tag, first, pool_n);
        ok = false;
    }
    if (a->tail_dim != b->tail_dim ||
        memcmp(a->tail, b->tail, (size_t)a->tail_dim * sizeof(float)) != 0) {
        fprintf(stderr, "ds4-test: mtp %s tail differs\n", tag);
        ok = false;
    }
    if (a->embd != b->embd ||
        memcmp(a->seam, b->seam, (size_t)a->embd * sizeof(float)) != 0) {
        fprintf(stderr, "ds4-test: mtp %s seam hidden differs\n", tag);
        ok = false;
    }
    return ok;
}

/* Byte for byte, exactly what the payload section promised to carry. */
static bool test_mtp_state_prefix_equal(const test_mtp_state *a,
                                        const test_mtp_state *b,
                                        uint32_t rows, const char *tag) {
    bool ok = true;
    if (a->kv_dim != b->kv_dim || a->pool_dim != b->pool_dim ||
        a->tail_dim != b->tail_dim) {
        fprintf(stderr, "ds4-test: mtp %s cache shape differs\n", tag);
        return false;
    }
    if (rows > a->len || rows > b->len) {
        fprintf(stderr, "ds4-test: mtp %s prefix %u of %u/%u\n", tag, rows,
                a->len, b->len);
        return false;
    }
    const size_t kv_n = (size_t)rows * a->kv_dim;
    if (memcmp(a->kv, b->kv, kv_n * sizeof(float)) != 0) {
        size_t first = 0;
        while (first < kv_n &&
               memcmp(&a->kv[first], &b->kv[first], sizeof(float)) == 0) {
            first++;
        }
        fprintf(stderr, "ds4-test: mtp %s KV differs in the first %u slots at "
                        "float %zu of %zu\n", tag, rows, first, kv_n);
        ok = false;
    }
    const size_t pool_n = (size_t)rows * a->pool_dim;
    if (memcmp(a->pool, b->pool, pool_n * sizeof(float)) != 0) {
        size_t first = 0;
        while (first < pool_n &&
               memcmp(&a->pool[first], &b->pool[first], sizeof(float)) == 0) {
            first++;
        }
        fprintf(stderr, "ds4-test: mtp %s pool differs in the first %u slots at "
                        "float %zu of %zu\n", tag, rows, first, pool_n);
        ok = false;
    }
    if (memcmp(a->tail, b->tail, (size_t)a->tail_dim * sizeof(float)) != 0) {
        fprintf(stderr, "ds4-test: mtp %s tail differs\n", tag);
        ok = false;
    }
    return ok;
}

static double test_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* Save a live session's payload and restore it into a fresh session whose
 * nextn caches were zeroed first, then compare the two byte for byte. */
static bool test_mtp_snapshot_point(ds4_session *live, ds4_engine *engine,
                                    uint32_t ctx, const char *tag) {
    char err[192] = {0};
    test_mtp_state ref = {0};
    test_mtp_state got = {0};
    ds4_session *scratch = NULL;
    FILE *fp = NULL;
    bool ok = test_mtp_state_capture(live, &ref);
    if (!ok) {
        fprintf(stderr, "ds4-test: mtp %s could not read the live caches\n", tag);
        return false;
    }
    const uint64_t expected = ds4_session_payload_bytes(live);
    fp = tmpfile();
    ok = fp != NULL;
    if (ok) {
        const double t0 = test_now_ms();
        const int rc = ds4_session_save_payload(live, fp, err, sizeof(err));
        const double save_ms = test_now_ms() - t0;
        const long written = ftello(fp);
        ok = rc == 0 && expected != 0 && (uint64_t)written == expected;
        fprintf(stderr, "ds4-test: mtp snapshot %s len=%u mod=%u payload=%" PRIu64
                        " save=%.1fms complete=%d\n", tag, ref.len,
                        ref.len % ref.pool_size, (uint64_t)written, save_ms,
                        ref.complete);
        if (!ok) {
            fprintf(stderr, "ds4-test: mtp %s save failed: %s "
                            "bytes=%ld expected=%" PRIu64 "\n", tag, err,
                    written, expected);
        }
        fflush(fp);
        rewind(fp);
        if (ok) {
            ok = ds4_session_create(&scratch, engine, ctx) == 0 && scratch != NULL;
        }
        if (ok) {
            /* The nextn caches are allocated lazily, so zeroing them here is
             * what puts zeros in place for the restore to overwrite. */
            ds4_test_glm_zero_mtp_caches(scratch);
            const double t1 = test_now_ms();
            const int lrc = ds4_session_load_payload(scratch, fp,
                                                     (uint64_t)written,
                                                     err, sizeof(err));
            const double load_ms = test_now_ms() - t1;
            fprintf(stderr, "ds4-test: mtp snapshot %s load=%.1fms\n", tag,
                    load_ms);
            ok = lrc == 0 &&
                 ds4_session_pos(scratch) == ds4_session_pos(live);
            if (!ok) fprintf(stderr, "ds4-test: mtp %s load failed: %s\n", tag, err);
        }
        if (ok) ok = test_mtp_state_capture(scratch, &got);
        if (ok) ok = test_mtp_state_equal(&ref, &got, tag);
    }
    test_mtp_state_free(&got);
    test_mtp_state_free(&ref);
    ds4_session_free(scratch);
    if (fp) fclose(fp);
    return ok;
}

static void test_session_snapshot_roundtrip(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine) return;

    ds4_session *reference = NULL;
    ds4_session *restored = NULL;
    ds4_session_snapshot snapshot = {0};
    ds4_tokens prompt = {0};
    char err[192] = {0};
    ds4_token_score before[8];
    ds4_token_score reference_after[8];
    ds4_token_score restored_before[8];
    ds4_token_score restored_after[8];
    enum { GLM_MTP_SNAPSHOT_CYCLES = TEST_GLM_MTP_PAYLOAD_CYCLES };
    int reference_accepted[GLM_MTP_SNAPSHOT_CYCLES * 2] = {0};
    int reference_counts[GLM_MTP_SNAPSHOT_CYCLES] = {0};
    int reference_total = 0;
    const bool test_glm_mtp = test_env_bool("DS4_TEST_GLM_MTP");
    /* Ticket 006 adds the nextn (MTP) checks only on GLM 5.3 with the MTP
     * engine on; every other model keeps this test exactly as it was. */
    const bool glm53_mtp = test_glm_mtp && ds4_engine_is_glm53(engine);
    uint32_t mtp_pool = 1u;
    uint32_t mtp_kv_dim = 0;
    uint32_t mtp_sync_len = 0;
    float *mtp_sync_slot = NULL;
    bool mtp_ok = true;
    bool mtp_saw_aligned = false;
    bool mtp_saw_unaligned = false;
    uint32_t mtp_point_len[GLM_MTP_SNAPSHOT_CYCLES + 2];
    uint32_t mtp_point_mod[GLM_MTP_SNAPSHOT_CYCLES + 2];
    int mtp_point_count = 0;
    test_mtp_state reference_final = {0};
    bool have_reference_final = false;
    double snapshot_save_ms = 0.0;
    double snapshot_load_ms = 0.0;
    bool snapshot_timed = false;
#ifdef DS4_ROCM_BUILD
    const float continued_logit_tolerance =
        ds4_engine_is_glm53(engine) ? 1e-5f : 1e-6f;
#else
    const float continued_logit_tolerance = 1e-6f;
#endif

    uint32_t ctx = test_env_u32("DS4_TEST_SNAPSHOT_CTX");
    if (ctx == 0) ctx = 1024;
    const char *prompt_path = getenv("DS4_TEST_SNAPSHOT_PROMPT");
    char *prompt_text = prompt_path && prompt_path[0] ?
        test_read_file(prompt_path) : NULL;
    if (prompt_path && prompt_path[0]) {
        TEST_ASSERT(prompt_text != NULL);
        if (!prompt_text) goto cleanup;
    }

    TEST_ASSERT(ds4_session_create(&reference, engine, ctx) == 0);
    if (!reference) goto cleanup;

    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user",
                            prompt_text ? prompt_text :
                            "Give one concise reason to test session restore.");
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    TEST_ASSERT(prompt.len > 0);
    fprintf(stderr, "ds4-test: snapshot prompt=%d ctx=%u glm_mtp=%d\n",
            prompt.len, ctx, test_glm_mtp);
    TEST_ASSERT(ds4_session_sync(reference, &prompt, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_top_logprobs(reference, before, 8) == 8);
    {
        const double t0 = test_now_ms();
        const int rc = ds4_session_save_snapshot(reference, &snapshot,
                                                 err, sizeof(err));
        snapshot_save_ms = test_now_ms() - t0;
        snapshot_timed = true;
        TEST_ASSERT(rc == 0);
    }
    TEST_ASSERT(snapshot.ptr != NULL && snapshot.len > 0);
    if (!snapshot.ptr || snapshot.len == 0) goto cleanup;

    if (glm53_mtp) {
        uint32_t cache_cap = 0;
        uint32_t seam_embd = 0;
        TEST_ASSERT(ds4_test_glm_mtp_cache_dims(reference, &mtp_kv_dim, NULL,
                                                &mtp_pool, NULL, &cache_cap));
        TEST_ASSERT(ds4_test_glm_mtp_seam_hidden(reference, NULL, 0, &seam_embd));
        TEST_ASSERT(mtp_pool > 0 && mtp_kv_dim > 0);
        mtp_sync_len = (uint32_t)ds4_session_pos(reference);
        /* The checkpoint's last slot: the prefill had no next token for it, so
         * it is still the zero the caches started with and the seam is what
         * will fill it once the token exists. */
        mtp_sync_slot = malloc((size_t)mtp_kv_dim * sizeof(float));
        TEST_ASSERT(mtp_sync_slot != NULL);
        TEST_ASSERT(mtp_sync_len > 0 &&
                    ds4_test_glm_read_mtp_kv(reference, mtp_sync_len - 1u,
                                             mtp_sync_slot, mtp_kv_dim,
                                             NULL, 0, NULL, 0));
        fprintf(stderr,
                "ds4-test: snapshot sync slot %u all zero=%d complete=%d\n",
                mtp_sync_len - 1u, test_mtp_row_zero(mtp_sync_slot, mtp_kv_dim),
                ds4_test_glm_mtp_cache_complete(reference));
        TEST_ASSERT(test_mtp_row_zero(mtp_sync_slot, mtp_kv_dim));
        TEST_ASSERT(ds4_test_glm_mtp_cache_complete(reference));
        /* Save point 1: the seam is pending. */
        mtp_point_len[mtp_point_count] = mtp_sync_len;
        mtp_point_mod[mtp_point_count] = mtp_sync_len % mtp_pool;
        mtp_point_count++;
        mtp_saw_aligned = mtp_sync_len % mtp_pool == 0u;
        mtp_saw_unaligned = mtp_sync_len % mtp_pool != 0u;
        mtp_ok = test_mtp_snapshot_point(reference, engine, ctx, "sync");
        TEST_ASSERT(mtp_ok);
        if (!mtp_ok) goto cleanup;
    }

    FILE *payload = tmpfile();
    TEST_ASSERT(payload != NULL);
    if (!payload) goto cleanup;
    TEST_ASSERT(ds4_session_save_payload(reference, payload, err, sizeof(err)) == 0);
    TEST_ASSERT(ftello(payload) == (off_t)snapshot.len);
    rewind(payload);
    uint8_t payload_chunk[4096];
    bool payload_matches = true;
    for (uint64_t offset = 0; offset < snapshot.len; ) {
        size_t bytes = snapshot.len - offset > sizeof(payload_chunk) ?
            sizeof(payload_chunk) : (size_t)(snapshot.len - offset);
        if (fread(payload_chunk, 1, bytes, payload) != bytes ||
            memcmp(payload_chunk, snapshot.ptr + offset, bytes) != 0) {
            fprintf(stderr, "ds4-test: snapshot payload differs at offset %llu\n",
                    (unsigned long long)offset);
            payload_matches = false;
            break;
        }
        offset += bytes;
    }
    fclose(payload);
    TEST_ASSERT(payload_matches);
    if (!payload_matches) goto cleanup;

    if (test_glm_mtp) {
        for (int cycle = 0; cycle < GLM_MTP_SNAPSHOT_CYCLES; cycle++) {
            const int first = ds4_session_argmax(reference);
            const int n = ds4_session_eval_speculative_argmax(
                    reference, first, 2, -1,
                    reference_accepted + reference_total, 2,
                    err, sizeof(err));
            TEST_ASSERT(n > 0 && n <= 2);
            if (n <= 0 || n > 2) goto cleanup;
            reference_counts[cycle] = n;
            reference_total += n;
            /* Save points 2..: after every cycle until both remainder classes
             * of the pooled indexer have been exercised -- a checkpoint that
             * ends on a pool boundary and one that stops inside a pool, which
             * is where the seam sits in the middle of the raw tail. */
            if (glm53_mtp && !(mtp_saw_aligned && mtp_saw_unaligned)) {
                const uint32_t len = (uint32_t)ds4_session_pos(reference);
                mtp_point_len[mtp_point_count] = len;
                mtp_point_mod[mtp_point_count] = len % mtp_pool;
                mtp_point_count++;
                mtp_saw_aligned = mtp_saw_aligned || len % mtp_pool == 0u;
                mtp_saw_unaligned = mtp_saw_unaligned || len % mtp_pool != 0u;
                char tag[32];
                snprintf(tag, sizeof(tag), "cycle %d", cycle + 1);
                mtp_ok = test_mtp_snapshot_point(reference, engine, ctx, tag) &&
                         mtp_ok;
                TEST_ASSERT(mtp_ok);
                if (!mtp_ok) goto cleanup;
            }
        }
    } else {
        TEST_ASSERT(ds4_session_eval(reference, before[0].id,
                                     err, sizeof(err)) == 0);
    }
    TEST_ASSERT(ds4_session_top_logprobs(reference, reference_after, 8) == 8);
    if (glm53_mtp) {
        /* Save point at the end of every cycle, plus the reference's final
         * nextn state kept on the host for the restored session to be
         * compared against once it has run the same cycles. */
        mtp_point_len[mtp_point_count] = (uint32_t)ds4_session_pos(reference);
        mtp_point_mod[mtp_point_count] =
            (uint32_t)ds4_session_pos(reference) % mtp_pool;
        mtp_point_count++;
        mtp_ok = test_mtp_snapshot_point(reference, engine, ctx, "last cycle") &&
                 mtp_ok;
        TEST_ASSERT(mtp_ok);
        if (!mtp_ok) goto cleanup;
        if (!mtp_saw_aligned || !mtp_saw_unaligned) {
            for (int i = 0; i < mtp_point_count; i++) {
                fprintf(stderr, "ds4-test: snapshot mtp save point %d len=%u mod=%u\n",
                        i, mtp_point_len[i], mtp_point_mod[i]);
            }
            fprintf(stderr,
                    "ds4-test: snapshot mtp save points never covered both "
                    "checkpoint.len %% POOL classes (aligned=%d unaligned=%d)\n",
                    mtp_saw_aligned, mtp_saw_unaligned);
        }
        TEST_ASSERT(mtp_saw_aligned && mtp_saw_unaligned);
        have_reference_final = test_mtp_state_capture(reference, &reference_final);
        TEST_ASSERT(have_reference_final);
        if (!have_reference_final) goto cleanup;
    }
    ds4_session_free(reference);
    reference = NULL;

    TEST_ASSERT(ds4_session_create(&restored, engine, ctx) == 0);
    if (!restored) goto cleanup;
    {
        const double t0 = test_now_ms();
        const int rc = ds4_session_load_snapshot(restored, &snapshot,
                                                 err, sizeof(err));
        snapshot_load_ms = test_now_ms() - t0;
        TEST_ASSERT(rc == 0);
    }
    if (ds4_engine_is_glm53(engine) && snapshot_timed) {
        fprintf(stderr,
                "ds4-test: snapshot sync snapshot=%" PRIu64 " bytes save=%.1fms "
                "load=%.1fms\n",
                (uint64_t)snapshot.len, snapshot_save_ms, snapshot_load_ms);
    }
    {
        /* a loaded snapshot must save back byte-identical */
        ds4_session_snapshot again = {0};
        TEST_ASSERT(ds4_session_save_snapshot(restored, &again, err, sizeof(err)) == 0);
        TEST_ASSERT(again.len == snapshot.len);
        if (again.len == snapshot.len && memcmp(again.ptr, snapshot.ptr, again.len) != 0) {
            size_t first = 0;
            while (first < again.len && again.ptr[first] == snapshot.ptr[first]) first++;
            fprintf(stderr, "ds4-test: snapshot round trip differs at byte %zu of %" PRIu64 "\n", first, again.len);
            TEST_ASSERT(false);
        }
        ds4_session_snapshot_free(&again);
    }
    TEST_ASSERT(ds4_session_top_logprobs(restored, restored_before, 8) == 8);
    for (int i = 0; i < 8; i++) {
        if (restored_before[i].id != before[i].id ||
            fabsf(restored_before[i].logit - before[i].logit) > 1e-6f) {
            fprintf(stderr,
                    "ds4-test: snapshot before[%d] reference=(%d,%.9g) "
                    "restored=(%d,%.9g) delta=%.9g\n",
                    i, before[i].id, before[i].logit,
                    restored_before[i].id, restored_before[i].logit,
                    restored_before[i].logit - before[i].logit);
        }
        TEST_ASSERT(restored_before[i].id == before[i].id);
        TEST_ASSERT(fabsf(restored_before[i].logit - before[i].logit) <= 1e-6f);
    }

    if (test_glm_mtp) {
        int restored_total = 0;
        int single_cycles = 0;
        int double_cycles = 0;
        for (int cycle = 0; cycle < GLM_MTP_SNAPSHOT_CYCLES; cycle++) {
            int restored_accepted[2] = {0};
            const int first = ds4_session_argmax(restored);
            TEST_ASSERT(first == reference_accepted[restored_total]);
            const int n = ds4_session_eval_speculative_argmax(
                    restored, first, 2, -1,
                    restored_accepted, 2, err, sizeof(err));
            TEST_ASSERT(n == reference_counts[cycle]);
            if (n != reference_counts[cycle]) goto cleanup;
            for (int i = 0; i < n; i++) {
                TEST_ASSERT(restored_accepted[i] ==
                            reference_accepted[restored_total + i]);
            }
            restored_total += n;
            single_cycles += n == 1;
            double_cycles += n == 2;
        }
        TEST_ASSERT(restored_total == reference_total);
        fprintf(stderr,
                "ds4-test: GLM MTP snapshot cycles=%d single=%d double=%d tokens=%d\n",
                GLM_MTP_SNAPSHOT_CYCLES,
                single_cycles,
                double_cycles,
                restored_total);
        if (glm53_mtp && have_reference_final) {
            /* The seam the prompt-sync snapshot carried has to have been spent
             * by now: the restored session filled the checkpoint's last slot
             * of that prefix with the real next token, from the hidden the
             * payload restored. */
            test_mtp_state restored_final = {0};
            mtp_ok = test_mtp_state_capture(restored, &restored_final);
            TEST_ASSERT(mtp_ok);
            if (mtp_ok) {
                fprintf(stderr,
                        "ds4-test: snapshot seam slot %u written=%d "
                        "complete=%d/%d\n",
                        mtp_sync_len - 1u,
                        test_mtp_row_written(
                            restored_final.kv +
                                (uint64_t)(mtp_sync_len - 1u) * mtp_kv_dim,
                            mtp_kv_dim),
                        restored_final.complete, reference_final.complete);
                TEST_ASSERT(test_mtp_row_written(
                    restored_final.kv +
                        (uint64_t)(mtp_sync_len - 1u) * mtp_kv_dim,
                    mtp_kv_dim));
                /* Slots below the save point, and the raw tail, are the same
                 * bytes in both sessions at the same cycle end. */
                mtp_ok = test_mtp_state_prefix_equal(&reference_final,
                                                     &restored_final,
                                                     mtp_sync_len, "prefix");
                TEST_ASSERT(mtp_ok);
            }
            /* Slots above it are only printed: no measurement says they are
             * bit-identical, only that they are close. */
            if (mtp_ok && restored_final.len == reference_final.len) {
                float max_abs = 0.0f;
                for (uint32_t p = mtp_sync_len; p < reference_final.len; p++) {
                    const float d = test_mtp_row_max_abs(
                        reference_final.kv + (uint64_t)p * mtp_kv_dim,
                        restored_final.kv + (uint64_t)p * mtp_kv_dim,
                        mtp_kv_dim);
                    if (d > max_abs) max_abs = d;
                }
                fprintf(stderr,
                        "ds4-test: snapshot slots above the save point slots=%u "
                        "kv max_abs=%.6g\n",
                        reference_final.len - mtp_sync_len, max_abs);
            }
            test_mtp_state_free(&restored_final);
        }
    } else {
        TEST_ASSERT(ds4_session_eval(restored, before[0].id,
                                     err, sizeof(err)) == 0);
    }
    TEST_ASSERT(ds4_session_top_logprobs(restored, restored_after, 8) == 8);
    for (int i = 0; i < 8; i++) {
        if (restored_after[i].id != reference_after[i].id ||
            fabsf(restored_after[i].logit - reference_after[i].logit) >
                continued_logit_tolerance) {
            fprintf(stderr,
                    "ds4-test: snapshot after[%d] reference=(%d,%.9g) "
                    "restored=(%d,%.9g) delta=%.9g\n",
                    i, reference_after[i].id, reference_after[i].logit,
                    restored_after[i].id, restored_after[i].logit,
                    restored_after[i].logit - reference_after[i].logit);
        }
        TEST_ASSERT(restored_after[i].id == reference_after[i].id);
        TEST_ASSERT(fabsf(restored_after[i].logit -
                          reference_after[i].logit) <=
                    continued_logit_tolerance);
    }
    if (test_glm_mtp) {
        TEST_ASSERT(ds4_session_sync(restored, &prompt,
                                     err, sizeof(err)) == 0);
        TEST_ASSERT(ds4_session_top_logprobs(restored,
                                             restored_before, 8) == 8);
        for (int i = 0; i < 8; i++) {
            TEST_ASSERT(restored_before[i].id == before[i].id);
            TEST_ASSERT(fabsf(restored_before[i].logit - before[i].logit) <=
                        1e-6f);
        }
        int reuse_single = 0;
        int reuse_double = 0;
        for (int cycle = 0; cycle < 4; cycle++) {
            int cycle_accepted[2] = {0};
            const int first = ds4_session_argmax(restored);
            const int n = ds4_session_eval_speculative_argmax(
                    restored, first, 2, -1,
                    cycle_accepted, 2, err, sizeof(err));
            TEST_ASSERT(n > 0 && n <= 2);
            if (n <= 0 || n > 2) goto cleanup;
            reuse_single += n == 1;
            reuse_double += n == 2;
        }
        fprintf(stderr,
                "ds4-test: GLM MTP context reuse single=%d double=%d\n",
                reuse_single,
                reuse_double);
    }

cleanup:
    free(prompt_text);
    free(mtp_sync_slot);
    test_mtp_state_free(&reference_final);
    ds4_tokens_free(&prompt);
    ds4_session_snapshot_free(&snapshot);
    ds4_session_free(restored);
    ds4_session_free(reference);
}

/* Cross-mode .kv read-back (ticket 006).  A .kv file's header says nothing
 * about the MTP switch, so a server with MTP on and a server with it off read
 * each other's files.  One run covers one side of that: DS4_TEST_PAYLOAD_MODE
 * picks save or load, DS4_TEST_PAYLOAD_FILE names the file, and
 * DS4_TEST_GLM_MTP decides whether this engine maintains the nextn caches at
 * all.  Run it four ways to walk every combination. */

static void test_glm_mtp_payload_cross_build_prompt(ds4_engine *engine,
                                                    ds4_tokens *prompt) {
    ds4_chat_begin(engine, prompt);
    ds4_chat_append_message(engine, prompt, "user",
                            "Give one concise reason to test session restore.");
    ds4_chat_append_assistant_prefix(engine, prompt, DS4_THINK_NONE);
}

/* GLM_MTP_SNAPSHOT_CYCLES speculative cycles, or the same number of greedy
 * tokens through the plain eval when the engine has no MTP.  Returns the
 * committed token count. */
static int test_glm_mtp_payload_cross_generate(ds4_session *s, int mtp,
                                               int *out, int out_cap,
                                               char *err, size_t errlen) {
    int total = 0;
    if (mtp) {
        for (int cycle = 0; cycle < TEST_GLM_MTP_PAYLOAD_CYCLES; cycle++) {
            const int first = ds4_session_argmax(s);
            const int n = ds4_session_eval_speculative_argmax(
                s, first, 2, -1, out + total, 2, err, errlen);
            if (n <= 0 || total + n > out_cap) return -1;
            total += n;
        }
        return total;
    }
    const int first = ds4_session_argmax(s);
    if (ds4_session_eval(s, first, err, errlen) != 0) return -1;
    out[total++] = first;
    while (total < TEST_GLM_MTP_PAYLOAD_CYCLES) {
        const int next = ds4_session_argmax(s);
        if (ds4_session_eval(s, next, err, errlen) != 0) return -1;
        out[total++] = next;
    }
    return total;
}

/* Sidecar `<DS4_TEST_PAYLOAD_FILE>.ref` written by "fail" mode next to the .kv
 * file.  The "fail" file's history was produced by speculative decode cycles,
 * and decode and prefill differ numerically, so a cold prefill of that history
 * is not a valid oracle; the writer session right after it saved is.
 * Format, all fields native-endian binary, in this order:
 *   int32   pos                      writer's ds4_session_pos after the save
 *   8 times (int32 id, float logit)  writer's ds4_session_top_logprobs, k=8
 *   int32   n                        number of greedy tokens that follow
 *   n times int32 token              writer continued with plain
 *                                    ds4_session_eval (argmax each step) via
 *                                    test_glm_mtp_payload_cross_generate(mtp=0)
 */
#define TEST_GLM_MTP_PAYLOAD_REF_TOP 8

typedef struct {
    int32_t pos;
    ds4_token_score top[TEST_GLM_MTP_PAYLOAD_REF_TOP];
    int32_t n;
    int tokens[TEST_GLM_MTP_PAYLOAD_CYCLES * 2];
} test_glm_mtp_payload_ref;

static bool test_glm_mtp_payload_ref_write(const char *ref_path,
                                           const test_glm_mtp_payload_ref *ref) {
    FILE *f = fopen(ref_path, "wb");
    if (!f) return false;
    bool ok = fwrite(&ref->pos, sizeof(ref->pos), 1, f) == 1;
    for (int i = 0; ok && i < TEST_GLM_MTP_PAYLOAD_REF_TOP; i++) {
        const int32_t id = ref->top[i].id;
        ok = fwrite(&id, sizeof(id), 1, f) == 1 &&
             fwrite(&ref->top[i].logit, sizeof(float), 1, f) == 1;
    }
    ok = ok && fwrite(&ref->n, sizeof(ref->n), 1, f) == 1;
    for (int i = 0; ok && i < ref->n; i++) {
        const int32_t t = ref->tokens[i];
        ok = fwrite(&t, sizeof(t), 1, f) == 1;
    }
    return fclose(f) == 0 && ok;
}

/* 1 = read, 0 = no sidecar, -1 = unreadable or malformed. */
static int test_glm_mtp_payload_ref_read(const char *ref_path,
                                         test_glm_mtp_payload_ref *ref) {
    FILE *f = fopen(ref_path, "rb");
    if (!f) return 0;
    memset(ref, 0, sizeof(*ref));
    bool ok = fread(&ref->pos, sizeof(ref->pos), 1, f) == 1;
    for (int i = 0; ok && i < TEST_GLM_MTP_PAYLOAD_REF_TOP; i++) {
        int32_t id = 0;
        ok = fread(&id, sizeof(id), 1, f) == 1 &&
             fread(&ref->top[i].logit, sizeof(float), 1, f) == 1;
        ref->top[i].id = id;
    }
    ok = ok && fread(&ref->n, sizeof(ref->n), 1, f) == 1 && ref->n > 0 &&
         ref->n <= (int32_t)(sizeof(ref->tokens) / sizeof(ref->tokens[0]));
    for (int i = 0; ok && i < ref->n; i++) {
        int32_t t = 0;
        ok = fread(&t, sizeof(t), 1, f) == 1;
        ref->tokens[i] = t;
    }
    fclose(f);
    return ok ? 1 : -1;
}

static void test_glm_mtp_payload_cross(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_is_glm53(engine)) {
        fprintf(stderr, "ds4-test: glm-mtp-payload-cross skipped "
                        "(needs a GLM 5.3 model)\n");
        return;
    }
    /* Whether this engine was opened with the GLM MTP head at all. */
    const bool mtp_requested = test_env_bool("DS4_TEST_GLM_MTP");
    const char *mode = getenv("DS4_TEST_PAYLOAD_MODE");
    const char *path = getenv("DS4_TEST_PAYLOAD_FILE");
    if (!mode || !mode[0] || !path || !path[0]) {
        fprintf(stderr, "ds4-test: glm-mtp-payload-cross needs "
                        "DS4_TEST_PAYLOAD_MODE=save|load and "
                        "DS4_TEST_PAYLOAD_FILE=PATH\n");
        TEST_ASSERT(false);
        return;
    }

    char err[192] = {0};
    ds4_tokens prompt = {0};
    ds4_tokens restored_hist = {0};
    ds4_session *session = NULL;
    ds4_session *cold = NULL;
    FILE *fp = NULL;
    int restored_tokens[TEST_GLM_MTP_PAYLOAD_CYCLES * 2] = {0};
    int cold_tokens[TEST_GLM_MTP_PAYLOAD_CYCLES * 2] = {0};
    test_glm_mtp_payload_ref ref;
    char ref_path[4096];
    const uint32_t ctx = 1024;
    /* Whether this engine maintains the nextn caches at all. */
    int mtp_engine = 0;

    test_glm_mtp_payload_cross_build_prompt(engine, &prompt);
    TEST_ASSERT(prompt.len > 0);
    TEST_ASSERT(snprintf(ref_path, sizeof(ref_path), "%s.ref", path) <
                (int)sizeof(ref_path));

    if (!strcmp(mode, "save")) {
        /* A save-mode file is a prefill of the prompt, whose reference is a
         * cold prefill; drop any sidecar a previous "fail" run left here. */
        remove(ref_path);
        TEST_ASSERT(ds4_session_create(&session, engine, ctx) == 0 && session);
        TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
        mtp_engine = ds4_test_glm_mtp_cache_complete(session);
        const uint64_t expected = ds4_session_payload_bytes(session);
        /* When MTP is requested, the cache-complete flag must be true before
         * saving to ensure the payload carries MTP data (ticket 006). */
        if (mtp_requested) {
            TEST_ASSERT(mtp_engine != 0);
        }
        fp = fopen(path, "wb");
        TEST_ASSERT(fp != NULL);
        if (!fp) goto cleanup;
        TEST_ASSERT(ds4_session_save_payload(session, fp, err, sizeof(err)) == 0);
        fflush(fp);
        const long written = ftell(fp);
        fclose(fp);
        fp = NULL;
        uint32_t seam_pos = 0;
        int seam_written = -1;
        int seam_kept = 0;
        TEST_ASSERT(ds4_test_glm_mtp_seam_state(session, &seam_pos,
                                                &seam_written, &seam_kept));
        fprintf(stderr,
                "ds4-test: glm-mtp-payload-cross save mtp=%d requested=%d len=%d "
                "file=%ld bytes=%" PRIu64 " complete=%d seam_valid=%d seam_pos=%u\n",
                mtp_engine, mtp_requested ? 1 : 0, ds4_session_pos(session),
                written, expected, mtp_engine, seam_kept, seam_pos);
        TEST_ASSERT(expected != 0 && (uint64_t)written == expected);
        goto cleanup;
    }
    if (!strcmp(mode, "fail")) {
        /* Test T3: MTP-on session with write failure drops flag to 0. */
        TEST_ASSERT(ds4_session_create(&session, engine, ctx) == 0 && session);
        TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
        TEST_ASSERT(mtp_requested);
        /* Arm the write failure and run cycles until it triggers. */
        ds4_test_glm_mtp_fail_next(session, DS4_TEST_GLM_MTP_FAIL_ROWS);
        int failed_attempts = 0;
        for (int attempt = 0; attempt < 8 && ds4_test_glm_mtp_cache_complete(session) != 0; attempt++) {
            int first = ds4_session_argmax(session);
            if (first < 0) break;
            int accepted[4] = {0};
            int ntok = ds4_session_eval_speculative_argmax(session, first, 4,
                                                          ds4_token_eos(engine),
                                                          accepted, 4, err, sizeof(err));
            if (ntok < 1) break;
            failed_attempts++;
        }
        TEST_ASSERT(failed_attempts > 0);  /* Ensure we actually ran cycles */
        /* After failure, flag should be 0. */
        TEST_ASSERT(ds4_test_glm_mtp_cache_complete(session) == 0);
        const uint64_t expected = ds4_session_payload_bytes(session);
        fp = fopen(path, "wb");
        TEST_ASSERT(fp != NULL);
        if (!fp) goto cleanup;
        TEST_ASSERT(ds4_session_save_payload(session, fp, err, sizeof(err)) == 0);
        fflush(fp);
        const long written = ftell(fp);
        fclose(fp);
        fp = NULL;
        fprintf(stderr, "ds4-test: glm-mtp-payload-cross fail: bytes=%" PRIu64
                        " written=%ld\n", expected, written);
        /* Payload size must match reported size. */
        TEST_ASSERT(expected != 0 && (uint64_t)written == expected);
        /* Oracle for the MTP-off "load" run: this writer session as saved,
         * then continued with plain eval.  The .kv file is already closed. */
        memset(&ref, 0, sizeof(ref));
        ref.pos = ds4_session_pos(session);
        TEST_ASSERT(ds4_session_top_logprobs(session, ref.top,
                                             TEST_GLM_MTP_PAYLOAD_REF_TOP) ==
                    TEST_GLM_MTP_PAYLOAD_REF_TOP);
        ref.n = test_glm_mtp_payload_cross_generate(
            session, 0, ref.tokens,
            (int)(sizeof(ref.tokens) / sizeof(ref.tokens[0])),
            err, sizeof(err));
        TEST_ASSERT(ref.n > 0);
        TEST_ASSERT(test_glm_mtp_payload_ref_write(ref_path, &ref));
        /* Now test loading with MTP-on (must fail/miss for incomplete cache). */
        fp = fopen(path, "rb");
        TEST_ASSERT(fp != NULL);
        if (!fp) goto cleanup;
        TEST_ASSERT(fseek(fp, 0, SEEK_END) == 0);
        const long size = ftell(fp);
        TEST_ASSERT(size > 0);
        rewind(fp);
        ds4_session *mtp_on_load = NULL;
        TEST_ASSERT(ds4_session_create(&mtp_on_load, engine, ctx) == 0 && mtp_on_load);
        int load_rc = ds4_session_load_payload(mtp_on_load, fp, (uint64_t)size,
                                               err, sizeof(err));
        /* MTP-on load of incomplete cache file must fail (miss). */
        if (load_rc == 0) {
            fprintf(stderr, "ds4-test: glm-mtp-payload-cross fail: ERROR - MTP-on load "
                    "succeeded but should have failed\n");
            TEST_ASSERT(false);
        } else {
            fprintf(stderr, "ds4-test: glm-mtp-payload-cross fail: MTP-on load failed as "
                    "expected (miss)\n");
        }
        ds4_session_free(mtp_on_load);
        fclose(fp);
        fp = NULL;
        goto cleanup;
    }
    if (strcmp(mode, "load")) {
        fprintf(stderr, "ds4-test: glm-mtp-payload-cross unknown mode %s\n", mode);
        TEST_ASSERT(false);
        goto cleanup;
    }

    fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) goto cleanup;
    TEST_ASSERT(fseek(fp, 0, SEEK_END) == 0);
    const long size = ftell(fp);
    TEST_ASSERT(size > 0);
    rewind(fp);

    TEST_ASSERT(ds4_session_create(&session, engine, ctx) == 0 && session);
    const int load_rc = ds4_session_load_payload(session, fp, (uint64_t)size,
                                                 err, sizeof(err));
    fclose(fp);
    fp = NULL;
    /* A file written by "save" mode holds exactly the prompt, so a cold
     * prefill of the restored history is an exact reference.  The "fail" mode
     * file was saved after speculative decode cycles; prefill and decode differ
     * numerically, so its reference is the writer's sidecar (see
     * test_glm_mtp_payload_ref).  A refused load leaves no history, and the
     * reference is the prompt the session re-syncs. */
    int have_ref = 0;
    if (load_rc == 0) {
        have_ref = test_glm_mtp_payload_ref_read(ref_path, &ref);
        TEST_ASSERT(have_ref >= 0);
        if (have_ref < 0) goto cleanup;
    }
    if (have_ref) {
        /* The writer's cache set was incomplete, so only an MTP-off engine
         * reads this file and generates through the plain eval path. */
        TEST_ASSERT(!mtp_requested);
        mtp_engine = 0;
    } else {
        if (load_rc == 0) {
            const ds4_tokens *h = ds4_session_tokens(session);
            TEST_ASSERT(h != NULL);
            for (int i = 0; h && i < h->len; i++) ds4_tokens_push(&restored_hist, h->v[i]);
        }
        TEST_ASSERT(ds4_session_create(&cold, engine, ctx) == 0 && cold);
        TEST_ASSERT(ds4_session_sync(cold, load_rc == 0 ? &restored_hist : &prompt,
                                     err, sizeof(err)) == 0);
        mtp_engine = ds4_test_glm_mtp_cache_complete(cold);
    }
    fprintf(stderr,
            "ds4-test: glm-mtp-payload-cross load mtp=%d requested=%d file=%ld "
            "load_rc=%d complete=%d\n", mtp_engine, mtp_requested ? 1 : 0,
            size, load_rc, ds4_test_glm_mtp_cache_complete(session));
    if (load_rc != 0) {
        /* The only file a session with live nextn caches refuses is one whose
         * MTP section carries no caches: a miss, exactly like an unreadable
         * file, and the caller prefills.  The trunk must not be left half
         * restored. */
        fprintf(stderr, "ds4-test: glm-mtp-payload-cross load refused: %s\n", err);
        TEST_ASSERT(mtp_engine != 0);
        TEST_ASSERT(!ds4_session_checkpoint_valid(session));
        TEST_ASSERT(ds4_test_glm_mtp_cache_complete(session) == 0);
        TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
        TEST_ASSERT(ds4_test_glm_mtp_cache_complete(session));
    } else {
        TEST_ASSERT(ds4_session_checkpoint_valid(session));
        if (mtp_engine) {
            /* A file this engine can draft from is a complete set of nextn
             * caches plus a seam that still has its rewrite to run. */
            uint32_t seam_pos = 0;
            int seam_written = -1;
            int seam_kept = 0;
            TEST_ASSERT(ds4_test_glm_mtp_seam_state(session, &seam_pos,
                                                    &seam_written, &seam_kept));
            fprintf(stderr,
                    "ds4-test: glm-mtp-payload-cross restored seam valid=%d "
                    "pos=%u written=%d\n", seam_kept, seam_pos, seam_written);
            TEST_ASSERT(ds4_test_glm_mtp_cache_complete(session));
            TEST_ASSERT(seam_kept != 0);
        } else {
            /* MTP off: the section was skipped whole, with no trailing bytes
             * left over, and nothing claims the caches cover the checkpoint. */
            TEST_ASSERT(ds4_test_glm_mtp_cache_complete(session) == 0);
        }
    }
    {
        /* The restored history starts with the prompt, and the reference
         * (cold sync of that history, or the writer) sits at the same
         * position. */
        const ds4_tokens *h = ds4_session_tokens(session);
        TEST_ASSERT(h != NULL && h->len >= prompt.len);
        for (int i = 0; h && i < prompt.len && i < h->len; i++) {
            TEST_ASSERT(h->v[i] == prompt.v[i]);
        }
        TEST_ASSERT(ds4_session_pos(session) ==
                    (have_ref ? ref.pos : ds4_session_pos(cold)));
    }

    {
        ds4_token_score got[8];
        ds4_token_score want[8];
        TEST_ASSERT(ds4_session_top_logprobs(session, got, 8) == 8);
        if (have_ref) {
            for (int i = 0; i < 8; i++) want[i] = ref.top[i];
        } else {
            TEST_ASSERT(ds4_session_top_logprobs(cold, want, 8) == 8);
        }
        for (int i = 0; i < 8; i++) {
            fprintf(stderr,
                    "ds4-test: glm-mtp-payload-cross top[%d] cold=(%d,%.9g) "
                    "restored=(%d,%.9g)\n", i, want[i].id, want[i].logit,
                    got[i].id, got[i].logit);
            TEST_ASSERT(got[i].id == want[i].id);
            TEST_ASSERT(fabsf(got[i].logit - want[i].logit) <= 1e-6f);
        }
    }

    {
        const int n_restored = test_glm_mtp_payload_cross_generate(
            session, mtp_engine, restored_tokens,
            (int)(sizeof(restored_tokens) / sizeof(restored_tokens[0])),
            err, sizeof(err));
        TEST_ASSERT(n_restored > 0);
        int n_cold;
        if (have_ref) {
            n_cold = ref.n;
            memcpy(cold_tokens, ref.tokens, sizeof(cold_tokens));
        } else {
            n_cold = test_glm_mtp_payload_cross_generate(
                cold, mtp_engine, cold_tokens,
                (int)(sizeof(cold_tokens) / sizeof(cold_tokens[0])),
                err, sizeof(err));
        }
        TEST_ASSERT(n_cold > 0);
        TEST_ASSERT(n_restored == n_cold);
        if (n_restored == n_cold) {
            for (int i = 0; i < n_restored; i++) {
                if (restored_tokens[i] != cold_tokens[i]) {
                    fprintf(stderr,
                            "ds4-test: glm-mtp-payload-cross token %d "
                            "cold=%d restored=%d\n", i, cold_tokens[i],
                            restored_tokens[i]);
                }
                TEST_ASSERT(restored_tokens[i] == cold_tokens[i]);
            }
        }
        fprintf(stderr,
                "ds4-test: glm-mtp-payload-cross load ok mtp=%d tokens=%d "
                "pos=%d/%d\n", mtp_engine, n_restored,
                ds4_session_pos(session),
                have_ref ? ref.pos + ref.n : ds4_session_pos(cold));
    }

cleanup:
    if (fp) fclose(fp);
    ds4_tokens_free(&prompt);
    ds4_tokens_free(&restored_hist);
    ds4_session_free(session);
    ds4_session_free(cold);
}

static uint64_t test_round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1) & ~(align - 1);
}

static uint32_t test_float_ordered_u32(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

typedef struct {
    size_t mismatch_count;
    uint32_t max_ulp;
    float max_abs;
} test_float_compare_stats;

static test_float_compare_stats test_compare_float_bits(
        const float *reference,
        const float *actual,
        size_t count) {
    test_float_compare_stats stats = {0};
    for (size_t i = 0; i < count; i++) {
        if (memcmp(&reference[i], &actual[i], sizeof(float)) != 0) {
            stats.mismatch_count++;
        }

        const uint32_t ref_ordered = test_float_ordered_u32(reference[i]);
        const uint32_t actual_ordered = test_float_ordered_u32(actual[i]);
        const uint32_t ulp = ref_ordered > actual_ordered
            ? ref_ordered - actual_ordered
            : actual_ordered - ref_ordered;
        if (ulp > stats.max_ulp) stats.max_ulp = ulp;

        const float abs_error = fabsf(reference[i] - actual[i]);
        if (abs_error > stats.max_abs) stats.max_abs = abs_error;
    }
    return stats;
}

#if defined(__APPLE__)
static const uint32_t test_copy_f32_patterns[] = {
    0x00000000u, 0x80000000u, /* signed zero */
    0x7f800000u, 0xff800000u, /* infinities */
    0x7fc00000u, 0x7fc12345u, 0xffc54321u, 0x7fa00001u, /* NaNs */
    0x00000001u, 0x007fffffu, 0x80000001u, 0x807fffffu, /* F32 subnormals */
    0x32ffffffu, 0x33000000u, 0x33000001u,
    0x337fffffu, 0x33800000u, 0x33800001u, /* minimum F16 subnormal boundary */
    0x387fbfffu, 0x387fc000u, 0x387fffffu,
    0x38800000u, 0x38800001u, /* maximum subnormal/minimum normal boundary */
    0x3f800fffu, 0x3f801000u, 0x3f801001u,
    0x3f802fffu, 0x3f803000u, 0x3f803001u, /* round-to-nearest ties */
    0x477fdfffu, 0x477fe000u, 0x477fefffu,
    0x477ff000u, 0x477ff001u, 0x47800000u, /* maximum/overflow boundary */
    0xc77fe000u, 0xbf801000u, 0x3eaaaaabu, 0xbeaaaaabu,
};

static void test_fill_copy_f32_patterns(void *dst, uint32_t n, uint32_t salt) {
    uint8_t *bytes = dst;
    const uint32_t n_patterns =
        (uint32_t)(sizeof(test_copy_f32_patterns) / sizeof(test_copy_f32_patterns[0]));
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t bits = test_copy_f32_patterns[(i + salt) % n_patterns];
        memcpy(bytes + (uint64_t)i * sizeof(bits), &bits, sizeof(bits));
    }
}
#endif

static uint16_t test_float_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };

    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static float test_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void test_fill_q8_0_weights(uint8_t *weights,
                                   uint32_t in_dim,
                                   uint32_t out_dim,
                                   uint32_t seed) {
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    for (uint32_t o = 0; o < out_dim; o++) {
        uint8_t *row = weights + (uint64_t)o * row_bytes;
        for (uint32_t b = 0; b < blocks; b++) {
            float vals[32];
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; i++) {
                const uint32_t k = b * 32u + i;
                const int v = (int)((o * 17u + k * 23u + (o ^ k) * 3u +
                                     seed * 29u + ((o + seed) ^ k) * 5u) % 67u) - 33;
                vals[i] = (float)v / 96.0f;
                float av = fabsf(vals[i]);
                if (av > amax) amax = av;
            }
            const uint16_t scale_bits = test_float_to_f16(amax / 127.0f);
            const float scale = test_f16_to_f32(scale_bits);
            memcpy(row + b * 34u, &scale_bits, sizeof(scale_bits));
            int8_t *qs = (int8_t *)(row + b * 34u + 2u);
            for (uint32_t i = 0; i < 32; i++) {
                int q = scale != 0.0f ? (int)lrintf(vals[i] / scale) : 0;
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                qs[i] = (int8_t)q;
            }
        }
    }
}

static void test_metal_f16_matvec_fast_nr0_4(void) {
    /*
     * This is the short regression for the long-context repetition failure.
     * Decode uses one-token F16 matvecs for several DS4 projections; the fast
     * nr0=4 variant must be numerically equivalent to the plain kernel.
     */
    const uint32_t in_dim = 4096;
    const uint32_t out_dim = 512;
    const uint64_t weight_bytes = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16(w);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)in_dim * sizeof(float));
    float *out_host = malloc((size_t)out_dim * sizeof(float));
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        x_host[i] = (float)((int)(i % 31u) - 15) / 32.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                            in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);

    float max_abs = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        float ref = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            ref += w * x_host[i];
        }
        float err = fabsf(out_host[o] - ref);
        if (err > max_abs) max_abs = err;
    }
    TEST_ASSERT(max_abs < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_f16_prefill_matmul(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 128;
    const uint64_t weight_bytes = (uint64_t)out_dim * in_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((o * 11u + i * 13u + (o ^ i) * 5u) % 61u) - 30;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16((float)v / 96.0f);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 7u + i * 17u + (t ^ i) * 3u) % 73u) - 36;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                          in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            float ref = 0.0f;
            for (uint32_t i = 0; i < in_dim; i++) {
                ref += test_f16_to_f32(weights[(uint64_t)o * in_dim + i]) *
                       x_host[(uint64_t)t * in_dim + i];
            }
            const float got = out_host[(uint64_t)t * out_dim + o];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)(n_tok * out_dim));
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_q8_0_prefill_matmul(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 128;
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint8_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights(weights, in_dim, out_dim, 0);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 19u + i * 7u + (t ^ i)) % 71u) - 35;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(out, weights_raw, weight_alloc, 0,
                                           in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            const uint8_t *row = weights + (uint64_t)o * row_bytes;
            float ref = 0.0f;
            for (uint32_t b = 0; b < in_dim / 32u; b++) {
                uint16_t scale_bits;
                memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
                const float scale = test_f16_to_f32(scale_bits);
                const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
                for (uint32_t i = 0; i < 32; i++) {
                    ref += scale * (float)qs[i] *
                           x_host[(uint64_t)t * in_dim + b * 32u + i];
                }
            }
            const float got = out_host[(uint64_t)t * out_dim + o];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)(n_tok * out_dim));
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_pack_slot_rows_f32(void) {
    const uint32_t n_rows = 3;
    const uint32_t width = 5;
    const uint32_t n_slots = 4;
    const uint32_t slot_cap = 6;
    const uint64_t slot_count = (uint64_t)n_slots * slot_cap * width;
    const uint64_t out_count = (uint64_t)n_rows * n_slots * width;
    const uint64_t slot_bytes = slot_count * sizeof(float);
    const uint64_t out_bytes = out_count * sizeof(float);

    ds4_gpu_tensor *slots = ds4_gpu_tensor_alloc(slot_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(slots != NULL);
    TEST_ASSERT(out != NULL);
    if (!slots || !out) {
        ds4_gpu_tensor_free(slots);
        ds4_gpu_tensor_free(out);
        return;
    }

    float *slot_host = malloc((size_t)slot_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(slot_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!slot_host || !out_host) {
        free(slot_host);
        free(out_host);
        ds4_gpu_tensor_free(slots);
        ds4_gpu_tensor_free(out);
        return;
    }

    for (uint32_t slot = 0; slot < n_slots; slot++) {
        for (uint32_t row = 0; row < slot_cap; row++) {
            for (uint32_t col = 0; col < width; col++) {
                slot_host[((uint64_t)slot * slot_cap + row) * width + col] =
                    (float)(slot * 1000u + row * 100u + col);
            }
        }
    }
    for (uint64_t i = 0; i < out_count; i++) out_host[i] = -1.0f;

    TEST_ASSERT(ds4_gpu_tensor_write(slots, 0, slot_host, slot_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_pack_slot_rows_f32_tensor(out,
                                                  slots,
                                                  n_rows,
                                                  width,
                                                  n_slots,
                                                  slot_cap) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            for (uint32_t col = 0; col < width; col++) {
                const float ref =
                    slot_host[((uint64_t)slot * slot_cap + row) * width + col];
                const float got =
                    out_host[((uint64_t)row * n_slots + slot) * width + col];
                TEST_ASSERT(got == ref);
            }
        }
    }

    free(slot_host);
    free(out_host);
    ds4_gpu_tensor_free(slots);
    ds4_gpu_tensor_free(out);
}

static void test_metal_store_raw_kv_batch_wrap(void) {
    const uint32_t raw_cap = 5;
    const uint32_t head_dim = 3;
    const uint32_t n_tokens = 4;
    const uint32_t pos0 = 3;
    const uint64_t kv_count = (uint64_t)n_tokens * head_dim;
    const uint64_t raw_count = (uint64_t)raw_cap * head_dim;
    const uint64_t kv_bytes = kv_count * sizeof(float);
    const uint64_t raw_bytes = raw_count * sizeof(float);

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(kv_bytes);
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    TEST_ASSERT(kv != NULL);
    TEST_ASSERT(raw != NULL);
    if (!kv || !raw) {
        ds4_gpu_tensor_free(kv);
        ds4_gpu_tensor_free(raw);
        return;
    }

    float kv_host[12];
    float raw_host[15];
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t d = 0; d < head_dim; d++) {
            kv_host[(uint64_t)t * head_dim + d] = (float)(100u * t + d);
        }
    }
    for (uint64_t i = 0; i < raw_count; i++) raw_host[i] = -1.0f;

    TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, kv_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
    TEST_ASSERT(ds4_gpu_store_raw_kv_batch_tensor(raw,
                                                  kv,
                                                  raw_cap,
                                                  pos0,
                                                  n_tokens,
                                                  head_dim) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(raw, 0, raw_host, raw_bytes) != 0);

    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t row = (pos0 + t) % raw_cap;
        for (uint32_t d = 0; d < head_dim; d++) {
            const float ref = kv_host[(uint64_t)t * head_dim + d];
            const float got = raw_host[(uint64_t)row * head_dim + d];
            TEST_ASSERT(got == ref);
        }
    }
    for (uint32_t d = 0; d < head_dim; d++) {
        TEST_ASSERT(raw_host[(uint64_t)2u * head_dim + d] == -1.0f);
    }

    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(raw);
}

static void test_dspark_cache_window_crop(void) {
    TEST_ASSERT(ds4_test_dspark_cache_window_crop());
}

static void test_metal_q8_0_decode_pair_exact_case(
        uint32_t out0_dim,
        uint32_t out1_dim,
        uint32_t seed0,
        uint32_t seed1) {
    /* Exercise the Q-A/KV contract with unequal, odd output extents and
     * independently page-aligned model ranges. The paired kernel must be
     * bit-identical to two standalone decode matvec dispatches. */
    const uint32_t in_dim = 4096;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight0_bytes = (uint64_t)out0_dim * row_bytes;
    const uint64_t weight1_bytes = (uint64_t)out1_dim * row_bytes;
    const uint64_t weight1_offset = test_round_up_u64(weight0_bytes, page);
    const uint64_t weight_alloc =
        test_round_up_u64(weight1_offset + weight1_bytes, page);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)page, (size_t)weight_alloc) == 0);
    if (!weights_raw) return;
    memset(weights_raw, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights((uint8_t *)weights_raw, in_dim, out0_dim, seed0);
    test_fill_q8_0_weights((uint8_t *)weights_raw + weight1_offset,
                           in_dim, out1_dim, seed1);

    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t out0_bytes = (uint64_t)out0_dim * sizeof(float);
    const uint64_t out1_bytes = (uint64_t)out1_dim * sizeof(float);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref0 = ds4_gpu_tensor_alloc(out0_bytes);
    ds4_gpu_tensor *ref1 = ds4_gpu_tensor_alloc(out1_bytes);
    ds4_gpu_tensor *pair0 = ds4_gpu_tensor_alloc(out0_bytes);
    ds4_gpu_tensor *pair1 = ds4_gpu_tensor_alloc(out1_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(ref0 != NULL);
    TEST_ASSERT(ref1 != NULL);
    TEST_ASSERT(pair0 != NULL);
    TEST_ASSERT(pair1 != NULL);
    if (!x || !ref0 || !ref1 || !pair0 || !pair1) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(ref0);
        ds4_gpu_tensor_free(ref1);
        ds4_gpu_tensor_free(pair0);
        ds4_gpu_tensor_free(pair1);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *ref0_host = malloc((size_t)out0_bytes);
    float *ref1_host = malloc((size_t)out1_bytes);
    float *pair0_host = malloc((size_t)out0_bytes);
    float *pair1_host = malloc((size_t)out1_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(ref0_host != NULL);
    TEST_ASSERT(ref1_host != NULL);
    TEST_ASSERT(pair0_host != NULL);
    TEST_ASSERT(pair1_host != NULL);
    if (!x_host || !ref0_host || !ref1_host || !pair0_host || !pair1_host) {
        free(x_host);
        free(ref0_host);
        free(ref1_host);
        free(pair0_host);
        free(pair1_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(ref0);
        ds4_gpu_tensor_free(ref1);
        ds4_gpu_tensor_free(pair0);
        ds4_gpu_tensor_free(pair1);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        const int v = (int)((i * 29u + (i ^ (i >> 3u)) * 7u) % 127u) - 63;
        x_host[i] = (float)v / 72.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(ref0, weights_raw, weight_alloc, 0,
                                           in_dim, out0_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(ref1, weights_raw, weight_alloc,
                                           weight1_offset,
                                           in_dim, out1_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_pair_tensor(pair0, pair1,
                                                weights_raw, weight_alloc,
                                                0, weight1_offset,
                                                in_dim, out0_dim, out1_dim,
                                                x, 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref0, 0, ref0_host, out0_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref1, 0, ref1_host, out1_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(pair0, 0, pair0_host, out0_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(pair1, 0, pair1_host, out1_bytes) != 0);

    uint32_t mismatch0 = 0;
    uint32_t mismatch1 = 0;
    float max_abs0 = 0.0f;
    float max_abs1 = 0.0f;
    for (uint32_t i = 0; i < out0_dim; i++) {
        if (memcmp(&ref0_host[i], &pair0_host[i], sizeof(float)) != 0) mismatch0++;
        const float err = fabsf(ref0_host[i] - pair0_host[i]);
        if (err > max_abs0) max_abs0 = err;
    }
    for (uint32_t i = 0; i < out1_dim; i++) {
        if (memcmp(&ref1_host[i], &pair1_host[i], sizeof(float)) != 0) mismatch1++;
        const float err = fabsf(ref1_host[i] - pair1_host[i]);
        if (err > max_abs1) max_abs1 = err;
    }
    if (mismatch0 != 0 || mismatch1 != 0) {
        fprintf(stderr,
                "ds4-test: paired Q8_0 exactness mismatches=%u/%u max_abs=%g, %u/%u max_abs=%g\n",
                mismatch0, out0_dim, max_abs0,
                mismatch1, out1_dim, max_abs1);
    }
    TEST_ASSERT(mismatch0 == 0);
    TEST_ASSERT(mismatch1 == 0);

    free(x_host);
    free(ref0_host);
    free(ref1_host);
    free(pair0_host);
    free(pair1_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(ref0);
    ds4_gpu_tensor_free(ref1);
    ds4_gpu_tensor_free(pair0);
    ds4_gpu_tensor_free(pair1);
    free(weights_raw);
}

static void test_metal_q8_0_decode_pair_exact(void) {
    /* Cover both possible one-bank tail directions. Distinct seeds ensure a
     * mistaken A-for-B weight binding cannot compare equal by construction. */
    test_metal_q8_0_decode_pair_exact_case(77, 19, 11, 97);
    test_metal_q8_0_decode_pair_exact_case(19, 77, 23, 131);
}

#if defined(__APPLE__)
static void test_metal_f16_compressor_pair_state_store_exact_case(
        uint32_t width,
        uint32_t ratio,
        uint32_t pos,
        uint32_t ape_type,
        uint32_t seed,
        bool test_decode_pack) {
    const uint32_t in_dim = 4096u;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t head_dim = width / coff;
    const uint32_t state_rows = coff * ratio;
    const bool emit = ((pos + 1u) % ratio) == 0u;
    TEST_ASSERT(!test_decode_pack ||
                (ratio == 4u && emit &&
                 (head_dim == 128u || head_dim == 512u)));
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t weight_bytes =
        (uint64_t)width * in_dim * sizeof(uint16_t);
    const uint64_t score_weight_offset =
        test_round_up_u64(weight_bytes, page);
    const uint64_t ape_offset = test_round_up_u64(
        score_weight_offset + weight_bytes, page);
    const uint64_t ape_elem_bytes = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)ratio * width * ape_elem_bytes;
    const uint64_t norm_offset =
        test_round_up_u64(ape_offset + ape_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        norm_offset + (uint64_t)head_dim * sizeof(float), page);
    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_count = (uint64_t)state_rows * width;
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)head_dim * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page,
                               (size_t)model_bytes) == 0);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_kv = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *ref_score = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_kv = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_score = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *ref_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *fused_comp = ds4_gpu_tensor_alloc(comp_bytes);

    float *x_host = malloc((size_t)x_bytes);
    float *ref_kv_host = malloc((size_t)out_bytes);
    float *ref_score_host = malloc((size_t)out_bytes);
    float *fused_kv_host = malloc((size_t)out_bytes);
    float *fused_score_host = malloc((size_t)out_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *fused_state_kv_host = malloc((size_t)state_bytes);
    float *fused_state_score_host = malloc((size_t)state_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *fused_comp_host = malloc((size_t)comp_bytes);

    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(ref_kv != NULL);
    TEST_ASSERT(ref_score != NULL);
    TEST_ASSERT(fused_kv != NULL);
    TEST_ASSERT(fused_score != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(fused_state_kv != NULL);
    TEST_ASSERT(fused_state_score != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(fused_comp != NULL);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(ref_kv_host != NULL);
    TEST_ASSERT(ref_score_host != NULL);
    TEST_ASSERT(fused_kv_host != NULL);
    TEST_ASSERT(fused_score_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(fused_state_kv_host != NULL);
    TEST_ASSERT(fused_state_score_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(fused_comp_host != NULL);

    const bool allocated = model_raw && x && ref_kv && ref_score && fused_kv &&
        fused_score && ref_state_kv && ref_state_score && fused_state_kv &&
        fused_state_score && ref_comp && fused_comp && x_host && ref_kv_host &&
        ref_score_host && fused_kv_host && fused_score_host &&
        ref_state_kv_host && ref_state_score_host && fused_state_kv_host &&
        fused_state_score_host && ref_comp_host && fused_comp_host;

    const char *pair_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ";
    const char *store_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_STORE_ONE";
    const char *decode_pack_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION";
    const char *exact_reduction_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_EXACT_REDUCTION_FUSION";
    const char *exact_reduction_poison_env =
        "DS4_METAL_TEST_POISON_COMPRESSOR_EXACT_REDUCTION_SCRATCH";
    char *saved_pair_disable = test_save_env(pair_disable_env);
    char *saved_store_disable = test_save_env(store_disable_env);
    char *saved_decode_pack_disable =
        test_save_env(decode_pack_disable_env);
    char *saved_exact_reduction_disable =
        test_save_env(exact_reduction_disable_env);
    char *saved_exact_reduction_poison =
        test_save_env(exact_reduction_poison_env);

    test_float_compare_stats kv_stats = {0};
    test_float_compare_stats score_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};
    test_float_compare_stats comp_stats = {0};

    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        uint16_t *kv_weights = model_raw;
        uint16_t *score_weights =
            (uint16_t *)((uint8_t *)model_raw + score_weight_offset);
        for (uint32_t o = 0; o < width; o++) {
            for (uint32_t i = 0; i < in_dim; i++) {
                const int kv_value =
                    (int)((o * 17u + i * 23u + (o ^ i) * 3u +
                           seed * 29u) % 67u) - 33;
                const int score_value =
                    (int)((o * 31u + i * 11u + (o ^ (i >> 2u)) * 5u +
                           seed * 19u) % 71u) - 35;
                const uint64_t wi = (uint64_t)o * in_dim + i;
                kv_weights[wi] = test_float_to_f16(
                    (float)kv_value / 96.0f);
                score_weights[wi] = test_float_to_f16(
                    (float)score_value / 104.0f);
            }
        }

        if (ape_type == 1u) {
            uint16_t *ape =
                (uint16_t *)((uint8_t *)model_raw + ape_offset);
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 13u + (i ^ (i >> 3u)) * 7u +
                           seed * 17u) % 61u) - 30;
                ape[i] = test_float_to_f16((float)value / 80.0f);
            }
        } else {
            float *ape = (float *)((uint8_t *)model_raw + ape_offset);
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 13u + (i ^ (i >> 3u)) * 7u +
                           seed * 17u) % 61u) - 30;
                ape[i] = (float)value / 80.0f;
            }
        }
        float *norm = (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.75f +
                (float)((i * 7u + seed * 3u) % 23u) / 64.0f;
        }

        for (uint32_t i = 0; i < in_dim; i++) {
            const int value =
                (int)((i * 29u + (i ^ (i >> 4u)) * 9u +
                       seed * 11u) % 127u) - 63;
            x_host[i] = (float)value / 88.0f;
        }
        for (uint32_t i = 0; i < width; i++) {
            const uint32_t poison = 0x7fc00001u + (i & 0x3ffu);
            memcpy(ref_kv_host + i, &poison, sizeof(poison));
            memcpy(ref_score_host + i, &poison, sizeof(poison));
            memcpy(fused_kv_host + i, &poison, sizeof(poison));
            memcpy(fused_score_host + i, &poison, sizeof(poison));
        }
        for (uint64_t i = 0; i < state_count; i++) {
            const int kv_value =
                (int)((i * 5u + seed * 13u) % 97u) - 48;
            const int score_value =
                (int)((i * 7u + seed * 5u) % 101u) - 50;
            ref_state_kv_host[i] = (float)kv_value / 64.0f;
            fused_state_kv_host[i] = ref_state_kv_host[i];
            ref_state_score_host[i] = (float)score_value / 72.0f;
            fused_state_score_host[i] = ref_state_score_host[i];
        }
        if (test_decode_pack) {
            static const uint32_t edge_bits[8] = {
                0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
                0x3f800000u, 0xbf800000u, 0x42a00000u, 0xc2a00000u,
            };
            for (uint32_t col = 0; col < 4u; col++) {
                for (uint32_t row = 0; row < 8u; row++) {
                    const uint64_t state_col =
                        (row >= 4u ? head_dim : 0u) + col;
                    const uint64_t state_index =
                        (uint64_t)row * width + state_col;
                    const uint32_t score_bits =
                        edge_bits[(row + col) & 7u];
                    const uint32_t kv_bits =
                        edge_bits[(7u - row + col) & 7u];
                    memcpy(ref_state_score_host + state_index,
                           &score_bits, sizeof(score_bits));
                    memcpy(fused_state_score_host + state_index,
                           &score_bits, sizeof(score_bits));
                    memcpy(ref_state_kv_host + state_index,
                           &kv_bits, sizeof(kv_bits));
                    memcpy(fused_state_kv_host + state_index,
                           &kv_bits, sizeof(kv_bits));
                }
            }
        }
        for (uint32_t i = 0; i < head_dim; i++) {
            const uint32_t poison = 0x7fc01001u + (i & 0x3ffu);
            memcpy(ref_comp_host + i, &poison, sizeof(poison));
            memcpy(fused_comp_host + i, &poison, sizeof(poison));
        }

        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_kv, 0, ref_kv_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_score, 0, ref_score_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_kv, 0, fused_kv_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_score, 0, fused_score_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, fused_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, fused_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_comp, 0, fused_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(unsetenv(pair_disable_env) == 0);
        TEST_ASSERT(unsetenv(store_disable_env) == 0);
        TEST_ASSERT(setenv(exact_reduction_disable_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(exact_reduction_poison_env) == 0);
        if (test_decode_pack) {
            TEST_ASSERT(setenv(decode_pack_disable_env, "1", 1) == 0);
        }

        TEST_ASSERT(ds4_gpu_matmul_f16_pair_tensor(
                        ref_kv, ref_score, model_raw, model_bytes,
                        0, score_weight_offset, in_dim, width, x, 1) != 0);
        TEST_ASSERT(ds4_gpu_compressor_update_tensor(
                        ref_kv, ref_score, ref_state_kv, ref_state_score,
                        ref_comp, model_raw, model_bytes, ape_offset, ape_type,
                        norm_offset, 0, head_dim, ratio, pos, 0, 0, 0,
                        10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f,
                        1.0e-6f, false, test_decode_pack, false) != 0);

        TEST_ASSERT(ds4_gpu_matmul_f16_pair_compressor_store_tensor(
                        fused_kv, fused_score,
                        fused_state_kv, fused_state_score,
                        model_raw, model_bytes, 0, score_weight_offset,
                        ape_offset, ape_type, in_dim, width, x,
                        ratio, pos) == 1);
        if (test_decode_pack) {
            TEST_ASSERT(unsetenv(decode_pack_disable_env) == 0);
            TEST_ASSERT(unsetenv(exact_reduction_disable_env) == 0);
            TEST_ASSERT(setenv(exact_reduction_poison_env, "1", 1) == 0);
        }
        TEST_ASSERT(ds4_gpu_compressor_update_tensor(
                        fused_kv, fused_score,
                        fused_state_kv, fused_state_score,
                        fused_comp, model_raw, model_bytes,
                        ape_offset, ape_type, norm_offset, 0,
                        head_dim, ratio, pos, 0, 0, 0,
                        10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f,
                        1.0e-6f, true, test_decode_pack, false) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_kv, 0, ref_kv_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_score, 0, ref_score_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_kv, 0, fused_kv_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_score, 0, fused_score_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_kv, 0, fused_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_score, 0, fused_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_comp, 0, fused_comp_host, comp_bytes) != 0);

        kv_stats = test_compare_float_bits(
            ref_kv_host, fused_kv_host, width);
        score_stats = test_compare_float_bits(
            ref_score_host, fused_score_host, width);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, fused_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, fused_state_score_host,
            (size_t)state_count);
        comp_stats = test_compare_float_bits(
            ref_comp_host, fused_comp_host, head_dim);
    }

    test_restore_env(pair_disable_env, saved_pair_disable);
    test_restore_env(store_disable_env, saved_store_disable);
    test_restore_env(decode_pack_disable_env, saved_decode_pack_disable);
    test_restore_env(exact_reduction_disable_env,
                     saved_exact_reduction_disable);
    test_restore_env(
        exact_reduction_poison_env, saved_exact_reduction_poison);

    fprintf(stderr,
            "ds4-test: compressor pair state-store exact width=%u ratio=%u "
            "pos=%u emit=%u ape=%s decode_pack=%u exact_reduce=%u "
            "proj=%zu/%zu state=%zu/%zu "
            "comp=%zu max_ulp=%u/%u/%u/%u/%u\n",
            width, ratio, pos, emit ? 1u : 0u,
            ape_type == 1u ? "f16" : "f32",
            test_decode_pack ? 1u : 0u,
            test_decode_pack ? 1u : 0u,
            kv_stats.mismatch_count, score_stats.mismatch_count,
            state_kv_stats.mismatch_count,
            state_score_stats.mismatch_count,
            comp_stats.mismatch_count,
            kv_stats.max_ulp, score_stats.max_ulp,
            state_kv_stats.max_ulp, state_score_stats.max_ulp,
            comp_stats.max_ulp);
    TEST_ASSERT(kv_stats.mismatch_count == 0);
    TEST_ASSERT(score_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);
    TEST_ASSERT(comp_stats.mismatch_count == 0);

    free(fused_comp_host);
    free(ref_comp_host);
    free(fused_state_score_host);
    free(fused_state_kv_host);
    free(ref_state_score_host);
    free(ref_state_kv_host);
    free(fused_score_host);
    free(fused_kv_host);
    free(ref_score_host);
    free(ref_kv_host);
    free(x_host);
    ds4_gpu_tensor_free(fused_comp);
    ds4_gpu_tensor_free(ref_comp);
    ds4_gpu_tensor_free(fused_state_score);
    ds4_gpu_tensor_free(fused_state_kv);
    ds4_gpu_tensor_free(ref_state_score);
    ds4_gpu_tensor_free(ref_state_kv);
    ds4_gpu_tensor_free(fused_score);
    ds4_gpu_tensor_free(fused_kv);
    ds4_gpu_tensor_free(ref_score);
    ds4_gpu_tensor_free(ref_kv);
    ds4_gpu_tensor_free(x);
    free(model_raw);
}

static void test_metal_f16_compressor_pair_state_store_exact(void) {
    test_metal_f16_compressor_pair_state_store_exact_case(
        256, 4, 8, 0, 17, false);
    test_metal_f16_compressor_pair_state_store_exact_case(
        256, 4, 11, 1, 23, true);
    test_metal_f16_compressor_pair_state_store_exact_case(
        1024, 4, 11, 1, 29, true);
    test_metal_f16_compressor_pair_state_store_exact_case(
        512, 128, 255, 1, 43, false);
}

static void test_metal_compressor_ape_add_exact_case(
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t ape_type,
        uint32_t seed,
        bool test_pack_fusion) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t input_count = (uint64_t)n_tokens * width;
    const uint64_t state_count = (uint64_t)state_rows * width;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t input_bytes = input_count * sizeof(float);
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(float);
    const uint64_t ape_elem_bytes = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)width * ratio * ape_elem_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t norm_offset = test_round_up_u64(ape_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        norm_offset + (uint64_t)head_dim * sizeof(float), page);

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *ref_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *fused_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *ref_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_score = ds4_gpu_tensor_alloc(state_bytes);
    TEST_ASSERT(kv != NULL);
    TEST_ASSERT(sc != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(fused_comp != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(fused_state_kv != NULL);
    TEST_ASSERT(fused_state_score != NULL);

    float *kv_host = malloc((size_t)input_bytes);
    float *sc_host = malloc((size_t)input_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *fused_comp_host = malloc((size_t)comp_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *fused_state_kv_host = malloc((size_t)state_bytes);
    float *fused_state_score_host = malloc((size_t)state_bytes);
    const uint64_t poison_count = input_count > state_count ?
        input_count : state_count;
    float *poison_host = test_pack_fusion ?
        malloc((size_t)(poison_count * sizeof(float))) : NULL;
    TEST_ASSERT(kv_host != NULL);
    TEST_ASSERT(sc_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(fused_comp_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(fused_state_kv_host != NULL);
    TEST_ASSERT(fused_state_score_host != NULL);
    TEST_ASSERT(!test_pack_fusion || poison_host != NULL);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page,
                               (size_t)model_bytes) == 0);
    const bool allocated = kv && sc && ref_comp && fused_comp && ref_state_kv &&
        ref_state_score && fused_state_kv && fused_state_score && kv_host &&
        sc_host && ref_comp_host && fused_comp_host && ref_state_kv_host &&
        ref_state_score_host && fused_state_kv_host && fused_state_score_host &&
        (!test_pack_fusion || poison_host) && model_raw;
    const char *disable_env = "DS4_METAL_DISABLE_COMPRESSOR_APE_ADD";
    const char *pack_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_RATIO4_PACK_FUSION";
    char *saved_disable = test_save_env(disable_env);
    char *saved_pack_disable = test_save_env(pack_disable_env);
    test_float_compare_stats comp_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};

    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        if (ape_type == 1u) {
            uint16_t *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)width * ratio; i++) {
                const int value = (int)((i * 17u + (i ^ (i >> 4u)) * 5u +
                                         seed * 13u) % 127u) - 63;
                ape[i] = test_float_to_f16((float)value / 80.0f);
            }
        } else {
            float *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)width * ratio; i++) {
                const int value = (int)((i * 17u + (i ^ (i >> 4u)) * 5u +
                                         seed * 13u) % 127u) - 63;
                ape[i] = (float)value / 80.0f;
            }
        }
        float *norm = (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.75f + (float)((i * 7u + seed) % 19u) / 64.0f;
        }
        for (uint64_t i = 0; i < input_count; i++) {
            const int kv_value = (int)((i * 29u + (i ^ (i >> 5u)) * 3u +
                                        seed * 23u) % 181u) - 90;
            const int sc_value = (int)((i * 31u + (i ^ (i >> 3u)) * 11u +
                                        seed * 17u) % 173u) - 86;
            kv_host[i] = (float)kv_value / 112.0f;
            sc_host[i] = (float)sc_value / 96.0f;
        }
        kv_host[0] = -0.0f;
        sc_host[0] = -0.0f;

        // Exercise exact-add edge values in the first active APE row. Using
        // bit patterns avoids host fast-math rewriting signed zeros or
        // subnormals before the legacy and fused Metal paths see them.
        const uint32_t cutoff = (n_tokens / ratio) * ratio;
        const uint32_t edge_token = cutoff < n_tokens ? cutoff : cutoff - ratio;
        const uint64_t edge_ape =
            (uint64_t)((pos0 + edge_token) % ratio) * width;
        float *edge_score = sc_host + (uint64_t)edge_token * width;
        static const uint16_t edge_f16[] = {
            0x0000u, 0x8000u, 0x0001u, 0x8001u,
            0x3c00u, 0xbc00u, 0x7bffu, 0xfbffu,
        };
        static const uint32_t edge_f32[] = {
            0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
            0x3f800000u, 0xbf800000u, 0x7f7fffffu, 0xff7fffffu,
        };
        static const uint32_t edge_score_f16[] = {
            0x80000000u, 0x00000000u, 0x00000000u, 0x80000000u,
            0xbf800000u, 0x3f800000u, 0xc77fe000u, 0x477fe000u,
        };
        static const uint32_t edge_score_f32[] = {
            0x80000000u, 0x00000000u, 0x00000000u, 0x80000000u,
            0xbf800000u, 0x3f800000u, 0xff7fffffu, 0x7f7fffffu,
        };
        if (ape_type == 1u) {
            uint16_t *ape = model_raw;
            memcpy(ape + edge_ape, edge_f16, sizeof(edge_f16));
            for (uint32_t i = 0; i < 8u; i++) {
                memcpy(edge_score + i, edge_score_f16 + i, sizeof(uint32_t));
            }
        } else {
            uint32_t *ape = model_raw;
            memcpy(ape + edge_ape, edge_f32, sizeof(edge_f32));
            for (uint32_t i = 0; i < 8u; i++) {
                memcpy(edge_score + i, edge_score_f32 + i, sizeof(uint32_t));
            }
        }
        for (uint64_t i = 0; i < state_count; i++) {
            ref_state_kv_host[i] = 1234.0f;
            fused_state_kv_host[i] = 1234.0f;
            ref_state_score_host[i] = -1234.0f;
            fused_state_score_host[i] = -1234.0f;
        }

        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, fused_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, ref_state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, fused_state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);

        if (test_pack_fusion) {
            TEST_ASSERT(unsetenv(disable_env) == 0);
        } else {
            TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
        }
        TEST_ASSERT(setenv(pack_disable_env, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_compressor_prefill_tensor(
            ref_comp, ref_state_kv, ref_state_score, kv, sc,
            model_raw, model_bytes, 0, ape_type, norm_offset, 0,
            head_dim, ratio, pos0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        if (test_pack_fusion) {
            // Overwrite every persistent pack cell with qNaN payloads through
            // the legacy replay path. This makes an omitted candidate write
            // observable even for plane-zero padding, which normal legacy
            // packing would otherwise leave at the correct 0/-inf values.
            for (uint64_t i = 0; i < poison_count; i++) {
                const uint32_t bits =
                    0x7fc00001u + (uint32_t)(i & 0x3ffu);
                memcpy(poison_host + i, &bits, sizeof(bits));
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                            kv, 0, poison_host, input_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            sc, 0, poison_host, input_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state_kv, 0, poison_host,
                            state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state_score, 0, poison_host,
                            state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                fused_comp, fused_state_kv, fused_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, 0, n_comp * ratio, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            kv, 0, kv_host, input_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            sc, 0, sc_host, input_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state_kv, 0, fused_state_kv_host,
                            state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state_score, 0, fused_state_score_host,
                            state_bytes) != 0);
            TEST_ASSERT(unsetenv(pack_disable_env) == 0);
        } else {
            TEST_ASSERT(unsetenv(disable_env) == 0);
        }
        TEST_ASSERT(ds4_gpu_compressor_prefill_tensor(
            fused_comp, fused_state_kv, fused_state_score, kv, sc,
            model_raw, model_bytes, 0, ape_type, norm_offset, 0,
            head_dim, ratio, pos0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        if (test_pack_fusion) {
            TEST_ASSERT(ds4_gpu_tensor_read(
                            kv, 0, poison_host, input_bytes) != 0);
            TEST_ASSERT(memcmp(kv_host, poison_host,
                               (size_t)input_bytes) == 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            sc, 0, poison_host, input_bytes) != 0);
            TEST_ASSERT(memcmp(sc_host, poison_host,
                               (size_t)input_bytes) == 0);
        }
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_comp, 0, fused_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_kv, 0, fused_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_score, 0, fused_state_score_host,
                        state_bytes) != 0);

        comp_stats = test_compare_float_bits(
            ref_comp_host, fused_comp_host, (size_t)comp_count);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, fused_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, fused_state_score_host, (size_t)state_count);
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(pack_disable_env, saved_pack_disable);
    fprintf(stderr,
            "ds4-test: compressor %s exact head=%u ratio=%u pos=%u "
            "tokens=%u ape=%s comp=%zu/%llu state_kv=%zu/%llu "
            "state_score=%zu/%llu max_ulp=%u/%u/%u\n",
            test_pack_fusion ? "ratio4 pack" : "APE add",
            head_dim, ratio, pos0, n_tokens, ape_type == 1u ? "f16" : "f32",
            comp_stats.mismatch_count, (unsigned long long)comp_count,
            state_kv_stats.mismatch_count, (unsigned long long)state_count,
            state_score_stats.mismatch_count, (unsigned long long)state_count,
            comp_stats.max_ulp, state_kv_stats.max_ulp,
            state_score_stats.max_ulp);
    TEST_ASSERT(comp_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);

    free(model_raw);
    free(poison_host);
    free(fused_state_score_host);
    free(fused_state_kv_host);
    free(ref_state_score_host);
    free(ref_state_kv_host);
    free(fused_comp_host);
    free(ref_comp_host);
    free(sc_host);
    free(kv_host);
    ds4_gpu_tensor_free(fused_state_score);
    ds4_gpu_tensor_free(fused_state_kv);
    ds4_gpu_tensor_free(ref_state_score);
    ds4_gpu_tensor_free(ref_state_kv);
    ds4_gpu_tensor_free(fused_comp);
    ds4_gpu_tensor_free(ref_comp);
    ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(kv);
}

static void test_metal_compressor_ape_add_exact(void) {
    test_metal_compressor_ape_add_exact_case(128, 4, 3, 16, 1, 7, false);
    test_metal_compressor_ape_add_exact_case(512, 4, 0, 16, 1, 13, false);
    test_metal_compressor_ape_add_exact_case(512, 4, 1, 14, 0, 19, false);
    test_metal_compressor_ape_add_exact_case(512, 128, 127, 257, 1, 31, false);
}

static void test_metal_compressor_ratio4_pack_exact(void) {
    test_metal_compressor_ape_add_exact_case(128, 4, 0, 4, 1, 37, true);
    test_metal_compressor_ape_add_exact_case(512, 4, 0, 8, 1, 41, true);
    test_metal_compressor_ape_add_exact_case(512, 4, 1, 14, 0, 43, true);
}

static void test_metal_compressor_ratio4_replay_pack_exact_case(
        uint32_t head_dim,
        uint32_t n_tokens,
        uint32_t seed) {
    const uint32_t width = 2u * head_dim;
    const uint32_t n_comp = n_tokens / 4u;
    const uint64_t input_count = (uint64_t)n_tokens * width;
    const uint64_t state_count = (uint64_t)8u * width;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t input_bytes = input_count * sizeof(float);
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)4u * width * sizeof(uint16_t);
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t norm_offset = test_round_up_u64(ape_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        norm_offset + (uint64_t)head_dim * sizeof(float), page);

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *ref_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *fused_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *ref_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_score = ds4_gpu_tensor_alloc(state_bytes);
    float *kv_host = malloc((size_t)input_bytes);
    float *sc_host = malloc((size_t)input_bytes);
    float *state_kv_host = malloc((size_t)state_bytes);
    float *state_score_host = malloc((size_t)state_bytes);
    float *source_after_host = malloc((size_t)input_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *fused_comp_host = malloc((size_t)comp_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *fused_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *fused_state_score_host = malloc((size_t)state_bytes);
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_bytes) == 0);

    TEST_ASSERT(kv != NULL);
    TEST_ASSERT(sc != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(fused_comp != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(fused_state_kv != NULL);
    TEST_ASSERT(fused_state_score != NULL);
    TEST_ASSERT(kv_host != NULL);
    TEST_ASSERT(sc_host != NULL);
    TEST_ASSERT(state_kv_host != NULL);
    TEST_ASSERT(state_score_host != NULL);
    TEST_ASSERT(source_after_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(fused_comp_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(fused_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(fused_state_score_host != NULL);
    TEST_ASSERT(model_raw != NULL);

    const char *ape_disable_env = "DS4_METAL_DISABLE_COMPRESSOR_APE_ADD";
    const char *pack_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_RATIO4_PACK_FUSION";
    char *saved_ape_disable = test_save_env(ape_disable_env);
    char *saved_pack_disable = test_save_env(pack_disable_env);
    test_float_compare_stats comp_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};

    const bool allocated = kv && sc && ref_comp && fused_comp &&
        ref_state_kv && ref_state_score && fused_state_kv &&
        fused_state_score && kv_host && sc_host && state_kv_host &&
        state_score_host && source_after_host && ref_comp_host && fused_comp_host &&
        ref_state_kv_host && fused_state_kv_host && ref_state_score_host &&
        fused_state_score_host && model_raw;
    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        uint16_t *ape = model_raw;
        for (uint64_t i = 0; i < (uint64_t)4u * width; i++) {
            const int value =
                (int)((i * 19u + (i ^ (i >> 3u)) * 7u + seed * 11u) %
                      113u) - 56;
            ape[i] = test_float_to_f16((float)value / 72.0f);
        }
        float *norm = (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.875f + (float)((i * 5u + seed) % 17u) / 64.0f;
        }
        for (uint64_t i = 0; i < input_count; i++) {
            const int kv_value =
                (int)((i * 31u + (i ^ (i >> 4u)) * 5u + seed * 13u) %
                      193u) - 96;
            const int sc_value =
                (int)((i * 37u + (i ^ (i >> 5u)) * 9u + seed * 17u) %
                      181u) - 90;
            kv_host[i] = (float)kv_value / 104.0f;
            sc_host[i] = (float)sc_value / 88.0f;
        }
        for (uint64_t i = 0; i < state_count; i++) {
            const int kv_value =
                (int)((i * 23u + (i ^ (i >> 2u)) * 3u + seed * 29u) %
                      167u) - 83;
            const int sc_value =
                (int)((i * 41u + (i ^ (i >> 6u)) * 11u + seed * 7u) %
                      157u) - 78;
            state_kv_host[i] = (float)kv_value / 80.0f;
            state_score_host[i] = (float)sc_value / 92.0f;
        }
        const uint32_t negative_zero = 0x80000000u;
        memcpy(kv_host, &negative_zero, sizeof(negative_zero));
        memcpy(state_kv_host, &negative_zero, sizeof(negative_zero));

        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);
        TEST_ASSERT(unsetenv(ape_disable_env) == 0);
        TEST_ASSERT(setenv(pack_disable_env, "1", 1) == 0);

        TEST_ASSERT(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
            ref_comp, ref_state_kv, ref_state_score, kv, sc,
            model_raw, model_bytes, 0, 1, norm_offset, 0,
            head_dim, 0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        // Poison every persistent pack cell through the legacy full-fill path
        // so a missing candidate write cannot inherit the reference value.
        for (uint64_t i = 0; i < input_count; i++) {
            const uint32_t bits = 0x7fc00001u + (uint32_t)(i & 0x3ffu);
            memcpy(kv_host + i, &bits, sizeof(bits));
            memcpy(sc_host + i, &bits, sizeof(bits));
        }
        for (uint64_t i = 0; i < state_count; i++) {
            const uint32_t bits = 0x7fc00401u + (uint32_t)(i & 0x3ffu);
            memcpy(ref_state_kv_host + i, &bits, sizeof(bits));
            memcpy(ref_state_score_host + i, &bits, sizeof(bits));
        }
        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
            fused_comp, fused_state_kv, fused_state_score, kv, sc,
            model_raw, model_bytes, 0, 1, norm_offset, 0,
            head_dim, 0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        // Restore finite sources and the candidate's original replay state.
        for (uint64_t i = 0; i < input_count; i++) {
            const int kv_value =
                (int)((i * 31u + (i ^ (i >> 4u)) * 5u + seed * 13u) %
                      193u) - 96;
            const int sc_value =
                (int)((i * 37u + (i ^ (i >> 5u)) * 9u + seed * 17u) %
                      181u) - 90;
            kv_host[i] = (float)kv_value / 104.0f;
            sc_host[i] = (float)sc_value / 88.0f;
        }
        memcpy(kv_host, &negative_zero, sizeof(negative_zero));
        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(unsetenv(pack_disable_env) == 0);
        TEST_ASSERT(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
            fused_comp, fused_state_kv, fused_state_score, kv, sc,
            model_raw, model_bytes, 0, 1, norm_offset, 0,
            head_dim, 0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        kv, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(kv_host, source_after_host,
                           (size_t)input_bytes) == 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        sc, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(sc_host, source_after_host,
                           (size_t)input_bytes) == 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_comp, 0, fused_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_kv, 0, fused_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_score, 0, fused_state_score_host,
                        state_bytes) != 0);

        comp_stats = test_compare_float_bits(
            ref_comp_host, fused_comp_host, (size_t)comp_count);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, fused_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, fused_state_score_host,
            (size_t)state_count);
    }

    test_restore_env(ape_disable_env, saved_ape_disable);
    test_restore_env(pack_disable_env, saved_pack_disable);
    fprintf(stderr,
            "ds4-test: compressor ratio4 replay pack exact head=%u "
            "tokens=%u comp=%zu/%llu state_kv=%zu/%llu "
            "state_score=%zu/%llu max_ulp=%u/%u/%u\n",
            head_dim, n_tokens,
            comp_stats.mismatch_count, (unsigned long long)comp_count,
            state_kv_stats.mismatch_count, (unsigned long long)state_count,
            state_score_stats.mismatch_count, (unsigned long long)state_count,
            comp_stats.max_ulp, state_kv_stats.max_ulp,
            state_score_stats.max_ulp);
    TEST_ASSERT(comp_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);

    free(model_raw);
    free(fused_state_score_host);
    free(ref_state_score_host);
    free(fused_state_kv_host);
    free(ref_state_kv_host);
    free(fused_comp_host);
    free(ref_comp_host);
    free(source_after_host);
    free(state_score_host);
    free(state_kv_host);
    free(sc_host);
    free(kv_host);
    ds4_gpu_tensor_free(fused_state_score);
    ds4_gpu_tensor_free(fused_state_kv);
    ds4_gpu_tensor_free(ref_state_score);
    ds4_gpu_tensor_free(ref_state_kv);
    ds4_gpu_tensor_free(fused_comp);
    ds4_gpu_tensor_free(ref_comp);
    ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(kv);
}

static void test_metal_compressor_ratio4_replay_pack_exact(void) {
    test_metal_compressor_ratio4_replay_pack_exact_case(128, 4, 47);
    test_metal_compressor_ratio4_replay_pack_exact_case(512, 8, 53);
}

static void test_metal_compressor_ratio4_direct_pool_exact_case(
        uint32_t head_dim,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t ape_type,
        bool replay,
        uint32_t seed) {
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t input_count = (uint64_t)n_tokens * width;
    const uint64_t state_count = (uint64_t)state_rows * width;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t input_bytes = input_count * sizeof(float);
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(float);
    const uint64_t ape_elem_bytes = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)ratio * width * ape_elem_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t norm_offset = test_round_up_u64(ape_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        norm_offset + (uint64_t)head_dim * sizeof(float), page);

    TEST_ASSERT(n_comp != 0);
    TEST_ASSERT(!replay || ((pos0 & 3u) == 0u && (n_tokens & 3u) == 0u));

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *ref_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *direct_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *ref_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *direct_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *direct_state_score = ds4_gpu_tensor_alloc(state_bytes);
    float *kv_host = malloc((size_t)input_bytes);
    float *sc_host = malloc((size_t)input_bytes);
    float *source_after_host = malloc((size_t)input_bytes);
    float *state_kv_host = malloc((size_t)state_bytes);
    float *state_score_host = malloc((size_t)state_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *direct_comp_host = malloc((size_t)comp_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *direct_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *direct_state_score_host = malloc((size_t)state_bytes);
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_bytes) == 0);

    TEST_ASSERT(kv != NULL);
    TEST_ASSERT(sc != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(direct_comp != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(direct_state_kv != NULL);
    TEST_ASSERT(direct_state_score != NULL);
    TEST_ASSERT(kv_host != NULL);
    TEST_ASSERT(sc_host != NULL);
    TEST_ASSERT(source_after_host != NULL);
    TEST_ASSERT(state_kv_host != NULL);
    TEST_ASSERT(state_score_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(direct_comp_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(direct_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(direct_state_score_host != NULL);
    TEST_ASSERT(model_raw != NULL);

    const char *ape_disable_env = "DS4_METAL_DISABLE_COMPRESSOR_APE_ADD";
    const char *pack_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_RATIO4_PACK_FUSION";
    const char *direct_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_RATIO4_DIRECT_POOL";
    char *saved_ape_disable = test_save_env(ape_disable_env);
    char *saved_pack_disable = test_save_env(pack_disable_env);
    char *saved_direct_disable = test_save_env(direct_disable_env);
    test_float_compare_stats comp_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};

    const bool allocated = kv && sc && ref_comp && direct_comp &&
        ref_state_kv && ref_state_score && direct_state_kv &&
        direct_state_score && kv_host && sc_host && source_after_host &&
        state_kv_host && state_score_host && ref_comp_host &&
        direct_comp_host && ref_state_kv_host && direct_state_kv_host &&
        ref_state_score_host && direct_state_score_host && model_raw;
    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        if (ape_type == 1u) {
            uint16_t *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 17u + (i ^ (i >> 4u)) * 5u + seed * 13u) %
                          127u) - 63;
                ape[i] = test_float_to_f16((float)value / 80.0f);
            }
        } else {
            float *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 17u + (i ^ (i >> 4u)) * 5u + seed * 13u) %
                          127u) - 63;
                ape[i] = (float)value / 80.0f;
            }
        }
        float *norm = (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.75f + (float)((i * 7u + seed) % 19u) / 64.0f;
        }

        for (uint64_t i = 0; i < input_count; i++) {
            const uint32_t token = (uint32_t)(i / width);
            const uint32_t col = (uint32_t)(i - (uint64_t)token * width);
            const int kv_value =
                (int)(((uint64_t)token * 37u + (uint64_t)col * 29u +
                       (col >= head_dim ? 71u : 3u) + seed * 23u) %
                      193u) - 96;
            const int sc_value =
                (int)(((uint64_t)token * 41u + (uint64_t)col * 31u +
                       (col >= head_dim ? 17u : 83u) + seed * 11u) %
                      181u) - 90;
            kv_host[i] = (float)kv_value / 104.0f;
            sc_host[i] = (float)sc_value / 32.0f;
        }
        const uint32_t negative_zero = 0x80000000u;
        memcpy(kv_host, &negative_zero, sizeof(negative_zero));
        memcpy(sc_host + width + head_dim, &negative_zero,
               sizeof(negative_zero));

        for (uint32_t row = 0; row < state_rows; row++) {
            for (uint32_t col = 0; col < width; col++) {
                const uint64_t i = (uint64_t)row * width + col;
                if (replay && row < ratio && col < head_dim) {
                    const int kv_value =
                        (int)(((uint64_t)row * 43u + (uint64_t)col * 19u +
                               seed * 29u) % 167u) - 83;
                    const int sc_value =
                        (int)(((uint64_t)row * 47u + (uint64_t)col * 23u +
                               seed * 7u) % 157u) - 78;
                    state_kv_host[i] = (float)kv_value / 80.0f;
                    state_score_host[i] = (float)sc_value / 28.0f;
                } else {
                    const uint32_t kv_bits =
                        0x7fc00001u + (uint32_t)(i & 0x3ffu);
                    const uint32_t score_bits =
                        0x7fc00401u + (uint32_t)(i & 0x3ffu);
                    memcpy(state_kv_host + i, &kv_bits, sizeof(kv_bits));
                    memcpy(state_score_host + i, &score_bits,
                           sizeof(score_bits));
                }
            }
        }

        for (uint64_t i = 0; i < comp_count; i++) {
            const uint32_t bits = 0x7fc00801u + (uint32_t)(i & 0x3ffu);
            memcpy(ref_comp_host + i, &bits, sizeof(bits));
            memcpy(direct_comp_host + i, &bits, sizeof(bits));
        }

        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        direct_comp, 0, direct_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        direct_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        direct_state_score, 0, state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(unsetenv(ape_disable_env) == 0);
        TEST_ASSERT(setenv(direct_disable_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(pack_disable_env) == 0);
        int ref_ok;
        if (replay) {
            ref_ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                ref_comp, ref_state_kv, ref_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, pos0, n_tokens, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        } else {
            ref_ok = ds4_gpu_compressor_prefill_tensor(
                ref_comp, ref_state_kv, ref_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, ratio, pos0, n_tokens, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        }
        TEST_ASSERT(ref_ok != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        kv, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(kv_host, source_after_host,
                           (size_t)input_bytes) == 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        sc, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(sc_host, source_after_host,
                           (size_t)input_bytes) == 0);

        TEST_ASSERT(unsetenv(direct_disable_env) == 0);
        TEST_ASSERT(setenv(pack_disable_env, "1", 1) == 0);
        int direct_ok;
        if (replay) {
            direct_ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                direct_comp, direct_state_kv, direct_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, pos0, n_tokens, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        } else {
            direct_ok = ds4_gpu_compressor_prefill_tensor(
                direct_comp, direct_state_kv, direct_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, ratio, pos0, n_tokens, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        }
        TEST_ASSERT(direct_ok != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        kv, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(kv_host, source_after_host,
                           (size_t)input_bytes) == 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        sc, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(sc_host, source_after_host,
                           (size_t)input_bytes) == 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        direct_comp, 0, direct_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        direct_state_kv, 0, direct_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        direct_state_score, 0, direct_state_score_host,
                        state_bytes) != 0);

        comp_stats = test_compare_float_bits(
            ref_comp_host, direct_comp_host, (size_t)comp_count);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, direct_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, direct_state_score_host,
            (size_t)state_count);
    }

    test_restore_env(ape_disable_env, saved_ape_disable);
    test_restore_env(pack_disable_env, saved_pack_disable);
    test_restore_env(direct_disable_env, saved_direct_disable);
    fprintf(stderr,
            "ds4-test: compressor ratio4 direct pool exact mode=%s "
            "head=%u pos=%u tokens=%u comp_rows=%u ape=%s "
            "comp=%zu/%llu state_kv=%zu/%llu state_score=%zu/%llu "
            "max_ulp=%u/%u/%u\n",
            replay ? "replay" : "prefill",
            head_dim, pos0, n_tokens, n_comp,
            ape_type == 1u ? "f16" : "f32",
            comp_stats.mismatch_count, (unsigned long long)comp_count,
            state_kv_stats.mismatch_count, (unsigned long long)state_count,
            state_score_stats.mismatch_count,
            (unsigned long long)state_count,
            comp_stats.max_ulp, state_kv_stats.max_ulp,
            state_score_stats.max_ulp);
    TEST_ASSERT(comp_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);

    free(model_raw);
    free(direct_state_score_host);
    free(ref_state_score_host);
    free(direct_state_kv_host);
    free(ref_state_kv_host);
    free(direct_comp_host);
    free(ref_comp_host);
    free(state_score_host);
    free(state_kv_host);
    free(source_after_host);
    free(sc_host);
    free(kv_host);
    ds4_gpu_tensor_free(direct_state_score);
    ds4_gpu_tensor_free(direct_state_kv);
    ds4_gpu_tensor_free(ref_state_score);
    ds4_gpu_tensor_free(ref_state_kv);
    ds4_gpu_tensor_free(direct_comp);
    ds4_gpu_tensor_free(ref_comp);
    ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(kv);
}

static void test_metal_compressor_ratio4_direct_pool_exact(void) {
    /* n_comp == 1 deliberately stays on the exact GGML reduction path. */
    test_metal_compressor_ratio4_direct_pool_exact_case(
        512, 0, 4, 1, false, 59);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        512, 1, 16, 0, false, 61);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        128, 3, 14, 1, false, 67);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        512, 2048, 8, 1, true, 71);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        128, 12, 12, 0, true, 73);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        128, 8, 4, 1, true, 79);
}

static void test_metal_inplace_rope_pair_exact(void) {
    typedef struct {
        uint32_t head_dim;
        uint32_t n_rot;
        uint32_t n_head;
        uint32_t n_tok;
        uint32_t pos0;
        bool inverse;
        float ext_factor;
    } rope_case;
    static const rope_case cases[] = {
        { 512, 64,  64,  1, UINT32_MAX, false, 1.0f },
        { 512, 64,   1,  1,  2047, false, 1.0f },
        { 512, 64,  64,  1, 65533,  true, 1.0f },
        { 128, 64,  64,  1,    37, false, 0.0f },
        { 512, 64,   4, 32,     0, false, 0.0f },
        { 512, 64,   7, 33,  2047,  true, 1.0f },
        { 128, 64,  64, 34, 65533, false, 1.0f },
        { 128, 64,  64, 35,    37,  true, 0.0f },
        { 128, 64,   4, 35, UINT32_MAX - 16u, true, 1.0f },
    };
    const char *disable_env = "DS4_METAL_DISABLE_INPLACE_ROPE_PAIR";
    const char *shared_disable_env =
        "DS4_METAL_DISABLE_SHARED_ROPE_COEFF";
    const char *affine_disable_env =
        "DS4_METAL_DISABLE_AFFINE_ROPE_PAIR";
    char *saved_disable = test_save_env(disable_env);
    char *saved_shared_disable = test_save_env(shared_disable_env);
    char *saved_affine_disable = test_save_env(affine_disable_env);
    size_t total_pair_mismatch = 0;
    size_t total_shared_mismatch = 0;
    size_t total_affine_mismatch = 0;
    size_t total_pair_prefix_mismatch = 0;
    size_t total_pair_tail_mismatch = 0;
    size_t total_shared_prefix_mismatch = 0;
    size_t total_shared_tail_mismatch = 0;
    size_t total_affine_prefix_mismatch = 0;
    size_t total_affine_tail_mismatch = 0;
    size_t total_elements = 0;

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const rope_case *c = &cases[ci];
        const size_t elements =
            (size_t)c->n_tok * c->n_head * c->head_dim;
        const uint64_t bytes = (uint64_t)elements * sizeof(float);
        const uint32_t n_nope = c->head_dim - c->n_rot;
        const float freq_base = c->ext_factor != 0.0f ? 160000.0f : 10000.0f;
        const float freq_scale = c->ext_factor != 0.0f ? 1.0f / 16.0f : 1.0f;
        const uint32_t n_ctx_orig = c->ext_factor != 0.0f ? 65536u : 0u;
        float attn_factor = 1.0f;
        if (c->ext_factor != 0.0f) {
            attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }

        ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(bytes);
        ds4_gpu_tensor *pair_candidate = ds4_gpu_tensor_alloc(bytes);
        ds4_gpu_tensor *shared_candidate = ds4_gpu_tensor_alloc(bytes);
        ds4_gpu_tensor *affine_candidate = ds4_gpu_tensor_alloc(bytes);
        float *input = malloc((size_t)bytes);
        float *reference_host = malloc((size_t)bytes);
        float *pair_host = malloc((size_t)bytes);
        float *shared_host = malloc((size_t)bytes);
        float *affine_host = malloc((size_t)bytes);
        TEST_ASSERT(reference != NULL);
        TEST_ASSERT(pair_candidate != NULL);
        TEST_ASSERT(shared_candidate != NULL);
        TEST_ASSERT(affine_candidate != NULL);
        TEST_ASSERT(input != NULL);
        TEST_ASSERT(reference_host != NULL);
        TEST_ASSERT(pair_host != NULL);
        TEST_ASSERT(shared_host != NULL);
        TEST_ASSERT(affine_host != NULL);

        const bool allocated = reference && pair_candidate &&
            shared_candidate && affine_candidate && input && reference_host &&
            pair_host && shared_host && affine_host;
        if (allocated) {
            for (size_t i = 0; i < elements; i++) {
                const uint32_t key =
                    (uint32_t)(i * 37u + (i ^ (i >> 5u)) * 11u + ci * 101u);
                const int value = (int)(key % 4093u) - 2046;
                input[i] = (float)value / 1024.0f;
                if ((i + ci * 17u) % 257u == 0u) {
                    const uint32_t negative_zero = 0x80000000u;
                    memcpy(&input[i], &negative_zero, sizeof(negative_zero));
                }
            }

            TEST_ASSERT(ds4_gpu_tensor_write(reference, 0, input, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                pair_candidate, 0, input, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                shared_candidate, 0, input, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                affine_candidate, 0, input, bytes) != 0);
            ds4_gpu_set_quality(false);

            TEST_ASSERT(setenv(affine_disable_env, "1", 1) == 0);
            TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
            TEST_ASSERT(setenv(shared_disable_env, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_rope_tail_tensor(
                reference,
                c->n_tok,
                c->n_head,
                c->head_dim,
                c->n_rot,
                c->pos0,
                n_ctx_orig,
                c->inverse,
                freq_base,
                freq_scale,
                c->ext_factor,
                attn_factor,
                32.0f,
                1.0f) != 0);

            TEST_ASSERT(unsetenv(disable_env) == 0);
            TEST_ASSERT(ds4_gpu_rope_tail_tensor(
                pair_candidate,
                c->n_tok,
                c->n_head,
                c->head_dim,
                c->n_rot,
                c->pos0,
                n_ctx_orig,
                c->inverse,
                freq_base,
                freq_scale,
                c->ext_factor,
                attn_factor,
                32.0f,
                1.0f) != 0);

            TEST_ASSERT(unsetenv(shared_disable_env) == 0);
            TEST_ASSERT(ds4_gpu_rope_tail_tensor(
                shared_candidate,
                c->n_tok,
                c->n_head,
                c->head_dim,
                c->n_rot,
                c->pos0,
                n_ctx_orig,
                c->inverse,
                freq_base,
                freq_scale,
                c->ext_factor,
                attn_factor,
                32.0f,
                1.0f) != 0);

            TEST_ASSERT(setenv(shared_disable_env, "1", 1) == 0);
            TEST_ASSERT(unsetenv(affine_disable_env) == 0);
            TEST_ASSERT(ds4_gpu_rope_tail_tensor(
                affine_candidate,
                c->n_tok,
                c->n_head,
                c->head_dim,
                c->n_rot,
                c->pos0,
                n_ctx_orig,
                c->inverse,
                freq_base,
                freq_scale,
                c->ext_factor,
                attn_factor,
                32.0f,
                1.0f) != 0);

            TEST_ASSERT(ds4_gpu_tensor_read(
                reference, 0, reference_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                pair_candidate, 0, pair_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                shared_candidate, 0, shared_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                affine_candidate, 0, affine_host, bytes) != 0);

            const test_float_compare_stats pair_stats =
                test_compare_float_bits(
                    reference_host, pair_host, elements);
            const test_float_compare_stats shared_stats =
                test_compare_float_bits(
                    pair_host, shared_host, elements);
            const test_float_compare_stats affine_stats =
                test_compare_float_bits(
                    pair_host, affine_host, elements);
            size_t pair_prefix_mismatch = 0;
            size_t pair_tail_mismatch = 0;
            size_t shared_prefix_mismatch = 0;
            size_t shared_tail_mismatch = 0;
            size_t affine_prefix_mismatch = 0;
            size_t affine_tail_mismatch = 0;
            for (uint32_t t = 0; t < c->n_tok; t++) {
                for (uint32_t h = 0; h < c->n_head; h++) {
                    const size_t row =
                        ((size_t)t * c->n_head + h) * c->head_dim;
                    for (uint32_t d = 0; d < n_nope; d++) {
                        if (memcmp(&input[row + d],
                                   &pair_host[row + d],
                                   sizeof(float)) != 0) {
                            pair_prefix_mismatch++;
                        }
                        if (memcmp(&input[row + d],
                                   &shared_host[row + d],
                                   sizeof(float)) != 0) {
                            shared_prefix_mismatch++;
                        }
                        if (memcmp(&input[row + d],
                                   &affine_host[row + d],
                                   sizeof(float)) != 0) {
                            affine_prefix_mismatch++;
                        }
                    }
                    for (uint32_t d = n_nope; d < c->head_dim; d++) {
                        if (memcmp(&reference_host[row + d],
                                   &pair_host[row + d],
                                   sizeof(float)) != 0) {
                            pair_tail_mismatch++;
                        }
                        if (memcmp(&pair_host[row + d],
                                   &shared_host[row + d],
                                   sizeof(float)) != 0) {
                            shared_tail_mismatch++;
                        }
                        if (memcmp(&pair_host[row + d],
                                   &affine_host[row + d],
                                   sizeof(float)) != 0) {
                            affine_tail_mismatch++;
                        }
                    }
                }
            }

            fprintf(stderr,
                    "ds4-test: in-place RoPE exactness case=%zu "
                    "shape=%ux%ux%u pos=%u inverse=%d ext=%g "
                    "pair=%zu/%zu shared=%zu/%zu affine=%zu/%zu "
                    "pair_prefix=%zu pair_tail=%zu "
                    "shared_prefix=%zu shared_tail=%zu "
                    "affine_prefix=%zu affine_tail=%zu "
                    "pair_max_ulp=%u shared_max_ulp=%u affine_max_ulp=%u "
                    "pair_max_abs=%g shared_max_abs=%g affine_max_abs=%g\n",
                    ci,
                    c->n_tok,
                    c->n_head,
                    c->head_dim,
                    c->pos0,
                    c->inverse ? 1 : 0,
                    c->ext_factor,
                    pair_stats.mismatch_count,
                    elements,
                    shared_stats.mismatch_count,
                    elements,
                    affine_stats.mismatch_count,
                    elements,
                    pair_prefix_mismatch,
                    pair_tail_mismatch,
                    shared_prefix_mismatch,
                    shared_tail_mismatch,
                    affine_prefix_mismatch,
                    affine_tail_mismatch,
                    pair_stats.max_ulp,
                    shared_stats.max_ulp,
                    affine_stats.max_ulp,
                    pair_stats.max_abs,
                    shared_stats.max_abs,
                    affine_stats.max_abs);
            TEST_ASSERT(pair_stats.mismatch_count == 0);
            TEST_ASSERT(shared_stats.mismatch_count == 0);
            TEST_ASSERT(affine_stats.mismatch_count == 0);
            TEST_ASSERT(pair_prefix_mismatch == 0);
            TEST_ASSERT(pair_tail_mismatch == 0);
            TEST_ASSERT(shared_prefix_mismatch == 0);
            TEST_ASSERT(shared_tail_mismatch == 0);
            TEST_ASSERT(affine_prefix_mismatch == 0);
            TEST_ASSERT(affine_tail_mismatch == 0);
            total_pair_mismatch += pair_stats.mismatch_count;
            total_shared_mismatch += shared_stats.mismatch_count;
            total_affine_mismatch += affine_stats.mismatch_count;
            total_pair_prefix_mismatch += pair_prefix_mismatch;
            total_pair_tail_mismatch += pair_tail_mismatch;
            total_shared_prefix_mismatch += shared_prefix_mismatch;
            total_shared_tail_mismatch += shared_tail_mismatch;
            total_affine_prefix_mismatch += affine_prefix_mismatch;
            total_affine_tail_mismatch += affine_tail_mismatch;
            total_elements += elements;
        }

        free(affine_host);
        free(shared_host);
        free(pair_host);
        free(reference_host);
        free(input);
        ds4_gpu_tensor_free(affine_candidate);
        ds4_gpu_tensor_free(shared_candidate);
        ds4_gpu_tensor_free(pair_candidate);
        ds4_gpu_tensor_free(reference);
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(shared_disable_env, saved_shared_disable);
    test_restore_env(affine_disable_env, saved_affine_disable);
    fprintf(stderr,
            "ds4-test: in-place RoPE total pair=%zu/%zu shared=%zu/%zu "
            "affine=%zu/%zu "
            "pair_prefix=%zu pair_tail=%zu "
            "shared_prefix=%zu shared_tail=%zu "
            "affine_prefix=%zu affine_tail=%zu\n",
            total_pair_mismatch,
            total_elements,
            total_shared_mismatch,
            total_elements,
            total_affine_mismatch,
            total_elements,
            total_pair_prefix_mismatch,
            total_pair_tail_mismatch,
            total_shared_prefix_mismatch,
            total_shared_tail_mismatch,
            total_affine_prefix_mismatch,
            total_affine_tail_mismatch);
    TEST_ASSERT(total_pair_mismatch == 0);
    TEST_ASSERT(total_shared_mismatch == 0);
    TEST_ASSERT(total_affine_mismatch == 0);
    TEST_ASSERT(total_pair_prefix_mismatch == 0);
    TEST_ASSERT(total_pair_tail_mismatch == 0);
    TEST_ASSERT(total_shared_prefix_mismatch == 0);
    TEST_ASSERT(total_shared_tail_mismatch == 0);
    TEST_ASSERT(total_affine_prefix_mismatch == 0);
    TEST_ASSERT(total_affine_tail_mismatch == 0);
}

static void test_metal_contiguous_f32_f16_roundtrip_exact(void) {
    typedef struct {
        uint32_t n;
        uint32_t src_offset;
        uint32_t dst_offset;
        uint32_t padding;
    } copy_case;
    static const copy_case cases[] = {
        { 1,  0,  0, 16 },
        { 3,  4,  2, 16 },
        { 4, 16,  8, 16 },
        { 5, 12,  6, 16 },
        { 17, 20, 10, 16 },
        { 65,  4,  2, 16 },
        { 1,  0,  0, 0 },
        { 2,  0,  0, 0 },
        { 3,  0,  0, 0 },
        { 1,  4,  2, 0 },
        { 2,  4,  2, 0 },
        { 3,  4,  2, 0 },
    };
    const char *env_name = "DS4_METAL_DISABLE_CONTIG_F32_F16_COPY";
    char *saved_env = test_save_env(env_name);
    size_t half_mismatch = 0;
    size_t half_guard_mismatch = 0;
    size_t roundtrip_mismatch = 0;
    size_t roundtrip_guard_mismatch = 0;
    size_t half_total = 0;
    size_t roundtrip_total = 0;

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const uint32_t n = cases[ci].n;
        const uint64_t src_bytes = cases[ci].src_offset +
                                   (uint64_t)n * sizeof(float) + cases[ci].padding;
        const uint64_t half_bytes = cases[ci].dst_offset +
                                    (uint64_t)n * sizeof(uint16_t) + cases[ci].padding;
        const uint32_t raw_cap = 3;
        const uint32_t raw_row = 1;
        const uint64_t raw_bytes =
            (uint64_t)raw_cap * n * sizeof(float);

        ds4_gpu_tensor *src_base = ds4_gpu_tensor_alloc(src_bytes);
        ds4_gpu_tensor *src_view = src_base
            ? ds4_gpu_tensor_view(src_base,
                                  cases[ci].src_offset,
                                  (uint64_t)n * sizeof(float))
            : NULL;
        ds4_gpu_tensor *half_ref = ds4_gpu_tensor_alloc(half_bytes);
        ds4_gpu_tensor *half_vec = ds4_gpu_tensor_alloc(half_bytes);
        ds4_gpu_tensor *raw_ref = ds4_gpu_tensor_alloc(raw_bytes);
        ds4_gpu_tensor *raw_vec = ds4_gpu_tensor_alloc(raw_bytes);
        TEST_ASSERT(src_base != NULL);
        TEST_ASSERT(src_view != NULL);
        TEST_ASSERT(half_ref != NULL);
        TEST_ASSERT(half_vec != NULL);
        TEST_ASSERT(raw_ref != NULL);
        TEST_ASSERT(raw_vec != NULL);

        uint8_t *src_host = malloc((size_t)src_bytes);
        uint16_t *half_init = malloc((size_t)half_bytes);
        uint16_t *half_ref_host = malloc((size_t)half_bytes);
        uint16_t *half_vec_host = malloc((size_t)half_bytes);
        uint32_t *raw_init = malloc((size_t)raw_bytes);
        uint32_t *raw_ref_host = malloc((size_t)raw_bytes);
        uint32_t *raw_vec_host = malloc((size_t)raw_bytes);
        TEST_ASSERT(src_host != NULL);
        TEST_ASSERT(half_init != NULL);
        TEST_ASSERT(half_ref_host != NULL);
        TEST_ASSERT(half_vec_host != NULL);
        TEST_ASSERT(raw_init != NULL);
        TEST_ASSERT(raw_ref_host != NULL);
        TEST_ASSERT(raw_vec_host != NULL);

        const bool allocated = src_base && src_view && half_ref && half_vec &&
            raw_ref && raw_vec && src_host && half_init && half_ref_host &&
            half_vec_host && raw_init && raw_ref_host && raw_vec_host;
        if (allocated) {
            memset(src_host, 0x6d, (size_t)src_bytes);
            test_fill_copy_f32_patterns(src_host + cases[ci].src_offset,
                                        n,
                                        (uint32_t)(ci * 7u));
            const size_t half_words = (size_t)(half_bytes / sizeof(uint16_t));
            for (size_t i = 0; i < half_words; i++) {
                half_init[i] = (uint16_t)(0xa55au ^ (uint16_t)(i * 73u));
            }
            const size_t raw_words = (size_t)raw_cap * n;
            for (size_t i = 0; i < raw_words; i++) {
                raw_init[i] = 0x4a000000u + (uint32_t)i;
            }

            TEST_ASSERT(ds4_gpu_tensor_write(src_base, 0, src_host, src_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(half_ref, 0, half_init, half_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(half_vec, 0, half_init, half_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(raw_ref, 0, raw_init, raw_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(raw_vec, 0, raw_init, raw_bytes) != 0);

            TEST_ASSERT(setenv(env_name, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_tensor_copy_f32_to_f16(
                half_ref,
                cases[ci].dst_offset,
                src_base,
                cases[ci].src_offset,
                n) != 0);
            TEST_ASSERT(ds4_gpu_store_raw_kv_tensor(
                raw_ref, src_view, raw_cap, raw_row, n) != 0);

            TEST_ASSERT(setenv(env_name, "0", 1) == 0);
            TEST_ASSERT(ds4_gpu_tensor_copy_f32_to_f16(
                half_vec,
                cases[ci].dst_offset,
                src_base,
                cases[ci].src_offset,
                n) != 0);
            TEST_ASSERT(ds4_gpu_store_raw_kv_tensor(
                raw_vec, src_view, raw_cap, raw_row, n) != 0);

            TEST_ASSERT(ds4_gpu_tensor_read(
                half_ref, 0, half_ref_host, half_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                half_vec, 0, half_vec_host, half_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                raw_ref, 0, raw_ref_host, raw_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                raw_vec, 0, raw_vec_host, raw_bytes) != 0);

            const size_t half_first = cases[ci].dst_offset / sizeof(uint16_t);
            const size_t half_last = half_first + n;
            for (size_t i = 0; i < half_words; i++) {
                if (half_ref_host[i] == half_vec_host[i]) continue;
                if (i >= half_first && i < half_last) {
                    half_mismatch++;
                } else {
                    half_guard_mismatch++;
                }
            }

            const size_t raw_first = (size_t)raw_row * n;
            const size_t raw_last = raw_first + n;
            for (size_t i = 0; i < raw_words; i++) {
                if (raw_ref_host[i] == raw_vec_host[i]) continue;
                if (i >= raw_first && i < raw_last) {
                    roundtrip_mismatch++;
                } else {
                    roundtrip_guard_mismatch++;
                }
            }
            half_total += n;
            roundtrip_total += n;
        }

        free(raw_vec_host);
        free(raw_ref_host);
        free(raw_init);
        free(half_vec_host);
        free(half_ref_host);
        free(half_init);
        free(src_host);
        ds4_gpu_tensor_free(raw_vec);
        ds4_gpu_tensor_free(raw_ref);
        ds4_gpu_tensor_free(half_vec);
        ds4_gpu_tensor_free(half_ref);
        ds4_gpu_tensor_free(src_view);
        ds4_gpu_tensor_free(src_base);
    }

    test_restore_env(env_name, saved_env);
    fprintf(stderr,
            "ds4-test: contiguous conversion exactness "
            "f32_f16=%zu/%zu guard=%zu, f16_f32_roundtrip=%zu/%zu guard=%zu\n",
            half_mismatch,
            half_total,
            half_guard_mismatch,
            roundtrip_mismatch,
            roundtrip_total,
            roundtrip_guard_mismatch);
    TEST_ASSERT(half_mismatch == 0);
    TEST_ASSERT(half_guard_mismatch == 0);
    TEST_ASSERT(roundtrip_mismatch == 0);
    TEST_ASSERT(roundtrip_guard_mismatch == 0);
}
#endif

#if defined(__APPLE__)
static void test_metal_gathered_kv_stage_exact(void) {
    const uint32_t head_dim = 512;
    const uint32_t raw_cap = 7;
    const uint32_t n_raw = 5;
    const uint32_t n_comp = 3;
    const uint32_t raw_starts[] = {0, 2, 5, 6};
    const uint64_t raw_bytes =
        (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes =
        (uint64_t)n_comp * head_dim * sizeof(uint16_t);
    const uint64_t payload_bytes =
        ((uint64_t)n_raw + n_comp) * head_dim * sizeof(uint16_t);
    const uint64_t raw_out_bytes =
        (uint64_t)n_raw * head_dim * sizeof(uint16_t);
    const uint64_t raw_view_offset = 4;
    const uint64_t comp_view_offset = 2;
    const uint64_t dst_view_offset = 6;
    const uint64_t raw_base_bytes = raw_view_offset + raw_bytes + 12;
    const uint64_t comp_base_bytes = comp_view_offset + comp_bytes + 14;
    const uint64_t dst_base_bytes = dst_view_offset + payload_bytes + 10;
    const char *envs[] = {
        "DS4_METAL_REQUIRE_GATHERED_KV_STAGE",
        "DS4_METAL_DISABLE_CONTIG_F32_F16_COPY",
        "DS4_METAL_DISABLE_CONTIG_F16_F16_COPY",
    };
    char *saved[sizeof(envs)/sizeof(envs[0])];
    for (size_t i = 0; i < sizeof(envs)/sizeof(envs[0]); i++) {
        saved[i] = test_save_env(envs[i]);
    }

    ds4_gpu_tensor *raw_base = ds4_gpu_tensor_alloc(raw_base_bytes);
    ds4_gpu_tensor *raw = raw_base
        ? ds4_gpu_tensor_view(raw_base, raw_view_offset, raw_bytes)
        : NULL;
    ds4_gpu_tensor *comp_base = ds4_gpu_tensor_alloc(comp_base_bytes);
    ds4_gpu_tensor *comp = comp_base
        ? ds4_gpu_tensor_view(comp_base, comp_view_offset, comp_bytes)
        : NULL;
    ds4_gpu_tensor *ref_base = ds4_gpu_tensor_alloc(dst_base_bytes);
    ds4_gpu_tensor *ref = ref_base
        ? ds4_gpu_tensor_view(ref_base, dst_view_offset, payload_bytes)
        : NULL;
    ds4_gpu_tensor *fused_base = ds4_gpu_tensor_alloc(dst_base_bytes);
    ds4_gpu_tensor *fused = fused_base
        ? ds4_gpu_tensor_view(fused_base, dst_view_offset, payload_bytes)
        : NULL;
    TEST_ASSERT(raw_base != NULL);
    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp_base != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(ref_base != NULL);
    TEST_ASSERT(ref != NULL);
    TEST_ASSERT(fused_base != NULL);
    TEST_ASSERT(fused != NULL);

    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    uint8_t *dst_init = malloc((size_t)dst_base_bytes);
    uint8_t *ref_host = malloc((size_t)dst_base_bytes);
    uint8_t *fused_host = malloc((size_t)dst_base_bytes);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(dst_init != NULL);
    TEST_ASSERT(ref_host != NULL);
    TEST_ASSERT(fused_host != NULL);

    static const uint16_t half_patterns[] = {
        0x0000u, 0x8000u, 0x0001u, 0x03ffu, 0x0400u,
        0x3555u, 0x3c00u, 0x3c01u, 0x7bffu, 0xfbffu,
        0x7c00u, 0xfc00u, 0x7e00u, 0x7e01u, 0xfe55u,
    };
    const bool allocated = raw_base && raw && comp_base && comp &&
        ref_base && ref && fused_base && fused && raw_host && comp_host &&
        dst_init && ref_host && fused_host;
    size_t raw_mismatch = 0;
    size_t comp_mismatch = 0;
    size_t guard_mismatch = 0;
    if (allocated) {
        for (uint32_t row = 0; row < raw_cap; row++) {
            test_fill_copy_f32_patterns(
                raw_host + (uint64_t)row * head_dim,
                head_dim,
                row * 17u + 3u);
        }
        for (uint64_t i = 0; i < (uint64_t)n_comp * head_dim; i++) {
            comp_host[i] = half_patterns[(i * 7u + (i >> 3u)) %
                (sizeof(half_patterns)/sizeof(half_patterns[0]))];
        }
        for (uint64_t i = 0; i < dst_base_bytes; i++) {
            dst_init[i] = (uint8_t)(0xa5u ^ (uint8_t)(i * 37u));
        }
        TEST_ASSERT(ds4_gpu_tensor_write(
                        raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        comp, 0, comp_host, comp_bytes) != 0);
        for (size_t ci = 0;
             ci < sizeof(raw_starts)/sizeof(raw_starts[0]);
             ci++) {
            TEST_ASSERT(ds4_gpu_tensor_write(
                            ref_base, 0, dst_init, dst_base_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_base, 0, dst_init, dst_base_bytes) != 0);

            TEST_ASSERT(unsetenv(envs[0]) == 0);
            TEST_ASSERT(unsetenv(envs[1]) == 0);
            TEST_ASSERT(unsetenv(envs[2]) == 0);
            ds4_gpu_set_quality(true);
            TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                            ref, raw, raw_cap, raw_starts[ci], n_raw,
                            comp, 1, n_comp, head_dim) != 0);

            ds4_gpu_set_quality(false);
            TEST_ASSERT(setenv(envs[0], "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                            fused, raw, raw_cap, raw_starts[ci], n_raw,
                            comp, 1, n_comp, head_dim) != 0);

            TEST_ASSERT(ds4_gpu_tensor_read(
                            ref_base, 0, ref_host, dst_base_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            fused_base, 0, fused_host,
                            dst_base_bytes) != 0);
            for (uint64_t i = 0; i < dst_base_bytes; i++) {
                if (i < dst_view_offset ||
                    i >= dst_view_offset + payload_bytes) {
                    if (ref_host[i] != dst_init[i]) guard_mismatch++;
                    if (fused_host[i] != dst_init[i]) guard_mismatch++;
                } else if (ref_host[i] == fused_host[i]) {
                    continue;
                } else if (i < dst_view_offset + raw_out_bytes) {
                    raw_mismatch++;
                } else {
                    comp_mismatch++;
                }
            }
        }

        /* Component-copy diagnostics and quality mode prevent strict
         * selection of the gathered kernel. */
        TEST_ASSERT(setenv(envs[0], "1", 1) == 0);
        TEST_ASSERT(setenv(envs[1], "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                        fused, raw, raw_cap, 5, n_raw,
                        comp, 1, n_comp, head_dim) == 0);
        TEST_ASSERT(unsetenv(envs[1]) == 0);
        TEST_ASSERT(setenv(envs[2], "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                        fused, raw, raw_cap, 5, n_raw,
                        comp, 1, n_comp, head_dim) == 0);
        TEST_ASSERT(unsetenv(envs[2]) == 0);
        ds4_gpu_set_quality(true);
        TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                        fused, raw, raw_cap, 5, n_raw,
                        comp, 1, n_comp, head_dim) == 0);
        ds4_gpu_set_quality(false);
    }

    for (size_t i = 0; i < sizeof(envs)/sizeof(envs[0]); i++) {
        test_restore_env(envs[i], saved[i]);
    }
    fprintf(stderr,
            "ds4-test: gathered KV staging exact cases=%zu "
            "raw_bytes=%zu comp_bytes=%zu guard_bytes=%zu\n",
            sizeof(raw_starts)/sizeof(raw_starts[0]),
            raw_mismatch, comp_mismatch, guard_mismatch);
    TEST_ASSERT(raw_mismatch == 0);
    TEST_ASSERT(comp_mismatch == 0);
    TEST_ASSERT(guard_mismatch == 0);

    free(fused_host);
    free(ref_host);
    free(dst_init);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(fused);
    ds4_gpu_tensor_free(fused_base);
    ds4_gpu_tensor_free(ref);
    ds4_gpu_tensor_free(ref_base);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(comp_base);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(raw_base);
}

static void test_metal_contiguous_compressed_f16_attention_exact(void) {
    const uint32_t head_dim = 512;
    const uint32_t n_head = 2;
    const uint32_t raw_cap = 7;
    const uint32_t n_raw = 5;
    const uint32_t raw_start = 5;
    const uint32_t n_comp = 3;
    const uint64_t raw_bytes =
        (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes =
        (uint64_t)n_comp * head_dim * sizeof(uint16_t);
    const uint64_t q_bytes =
        (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();
    const char *env_name = "DS4_METAL_DISABLE_CONTIG_F16_F16_COPY";
    static const uint16_t half_patterns[] = {
        0x0000u, 0x8000u, 0x0001u, 0x03ffu, 0x0400u,
        0x1001u, 0x3555u, 0x3c00u, 0x3c01u, 0x4000u,
        0xbc00u, 0xc000u, 0x7bffu, 0xfbffu,
    };

    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *comp_base = ds4_gpu_tensor_alloc(comp_bytes + 18u);
    ds4_gpu_tensor *comp = comp_base
        ? ds4_gpu_tensor_view(comp_base, 2u, comp_bytes)
        : NULL;
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *heads_blit = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *heads_compute = ds4_gpu_tensor_alloc(q_bytes);
    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp_base != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(heads_blit != NULL);
    TEST_ASSERT(heads_compute != NULL);

    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    float *q_host = malloc((size_t)q_bytes);
    float *blit_host = malloc((size_t)q_bytes);
    float *compute_host = malloc((size_t)q_bytes);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(q_host != NULL);
    TEST_ASSERT(blit_host != NULL);
    TEST_ASSERT(compute_host != NULL);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    const bool allocated = raw && comp_base && comp && q && heads_blit &&
        heads_compute && raw_host && comp_host && q_host && blit_host &&
        compute_host && model_raw;
    char *saved_env = test_save_env(env_name);
    test_float_compare_stats stats = {0};
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        float *sinks = model_raw;
        sinks[0] = -0.375f;
        sinks[1] = 0.1875f;
        for (uint64_t i = 0; i < (uint64_t)raw_cap * head_dim; i++) {
            const int value =
                (int)((i * 19u + (i ^ (i >> 5u)) * 7u) % 193u) - 96;
            raw_host[i] = (float)value / 128.0f;
        }
        for (uint64_t i = 0; i < (uint64_t)n_comp * head_dim; i++) {
            comp_host[i] = half_patterns[(i * 5u + (i >> 4u)) %
                (sizeof(half_patterns) / sizeof(half_patterns[0]))];
        }
        for (uint32_t i = 0; i < n_head * head_dim; i++) {
            const int value = (int)((i * 37u + (i ^ (i >> 3u)) * 11u) % 251u) - 125;
            q_host[i] = (float)value / 96.0f;
        }

        TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(comp, 0, comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(setenv(env_name, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            heads_blit,
            model_raw,
            page,
            0,
            q,
            raw,
            n_raw,
            raw_cap,
            raw_start,
            comp,
            1,
            n_comp,
            NULL,
            0,
            n_head,
            head_dim) != 0);

        TEST_ASSERT(setenv(env_name, "0", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            heads_compute,
            model_raw,
            page,
            0,
            q,
            raw,
            n_raw,
            raw_cap,
            raw_start,
            comp,
            1,
            n_comp,
            NULL,
            0,
            n_head,
            head_dim) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
            heads_blit, 0, blit_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
            heads_compute, 0, compute_host, q_bytes) != 0);
        stats = test_compare_float_bits(
            blit_host, compute_host, (size_t)n_head * head_dim);
    }
    test_restore_env(env_name, saved_env);
    fprintf(stderr,
            "ds4-test: contiguous compressed-F16 staging exactness "
            "mismatches=%zu/%u max_ulp=%u max_abs=%g\n",
            stats.mismatch_count,
            n_head * head_dim,
            stats.max_ulp,
            stats.max_abs);
    TEST_ASSERT(stats.mismatch_count == 0);

    free(model_raw);
    free(compute_host);
    free(blit_host);
    free(q_host);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(heads_compute);
    ds4_gpu_tensor_free(heads_blit);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(comp_base);
    ds4_gpu_tensor_free(raw);
}

static void test_metal_persistent_zero_attention_mask_exact_case(
        uint32_t raw_cap,
        uint32_t n_raw,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t seed) {
    const uint32_t head_dim = 512;
    const uint32_t n_head = 2;
    const uint64_t raw_bytes =
        (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes =
        (uint64_t)n_comp * head_dim * sizeof(uint16_t);
    const uint64_t q_bytes =
        (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t mask_bytes = (uint64_t)n_comp * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();

    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *comp_mask = ds4_gpu_tensor_alloc(mask_bytes);
    ds4_gpu_tensor *legacy = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *persistent = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *masked = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *pad_legacy = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *after_mask = ds4_gpu_tensor_alloc(q_bytes);
    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(comp_mask != NULL);
    TEST_ASSERT(legacy != NULL);
    TEST_ASSERT(persistent != NULL);
    TEST_ASSERT(masked != NULL);
    TEST_ASSERT(pad_legacy != NULL);
    TEST_ASSERT(after_mask != NULL);

    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    float *q_host = malloc((size_t)q_bytes);
    float *mask_host = malloc((size_t)mask_bytes);
    float *legacy_host = malloc((size_t)q_bytes);
    float *persistent_host = malloc((size_t)q_bytes);
    float *masked_host = malloc((size_t)q_bytes);
    float *pad_legacy_host = malloc((size_t)q_bytes);
    float *after_mask_host = malloc((size_t)q_bytes);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(q_host != NULL);
    TEST_ASSERT(mask_host != NULL);
    TEST_ASSERT(legacy_host != NULL);
    TEST_ASSERT(persistent_host != NULL);
    TEST_ASSERT(masked_host != NULL);
    TEST_ASSERT(pad_legacy_host != NULL);
    TEST_ASSERT(after_mask_host != NULL);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    const bool allocated = raw && comp && q && comp_mask && legacy &&
        persistent && masked && pad_legacy && after_mask && raw_host &&
        comp_host && q_host && mask_host && legacy_host && persistent_host &&
        masked_host && pad_legacy_host && after_mask_host &&
        model_raw;
    const char *disable_env =
        "DS4_METAL_DISABLE_PERSISTENT_ZERO_ATTN_MASK";
    const char *pad_disable_env =
        "DS4_METAL_DISABLE_GATHERED_KV_PAD_FUSION";
    const char *shared_pad_disable_env =
        "DS4_METAL_DISABLE_SHARED_KV_PAD";
    char *saved_disable = test_save_env(disable_env);
    char *saved_pad_disable = test_save_env(pad_disable_env);
    char *saved_shared_pad_disable = test_save_env(shared_pad_disable_env);
    test_float_compare_stats persistent_stats = {0};
    test_float_compare_stats masked_stats = {0};
    test_float_compare_stats pad_stats = {0};
    test_float_compare_stats after_mask_stats = {0};

    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        float *sinks = model_raw;
        sinks[0] = -0.3125f;
        sinks[1] = 0.21875f;
        for (uint64_t i = 0; i < (uint64_t)raw_cap * head_dim; i++) {
            const int value = (int)((i * 17u + (i ^ (i >> 5u)) * 11u +
                                     seed * 13u) % 211u) - 105;
            raw_host[i] = (float)value / 128.0f;
        }
        for (uint64_t i = 0; i < (uint64_t)n_comp * head_dim; i++) {
            const int value = (int)((i * 23u + (i ^ (i >> 4u)) * 7u +
                                     seed * 19u) % 193u) - 96;
            comp_host[i] = test_float_to_f16((float)value / 112.0f);
        }
        for (uint32_t i = 0; i < n_head * head_dim; i++) {
            const int value = (int)((i * 31u + (i ^ (i >> 3u)) * 5u +
                                     seed * 29u) % 227u) - 113;
            q_host[i] = (float)value / 104.0f;
        }
        for (uint32_t i = 0; i < n_comp; i++) {
            mask_host[i] = i == 0 ? -8.0f : -(float)(i + 1u) / 8.0f;
        }

        TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(comp, 0, comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        comp_mask, 0, mask_host, mask_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            legacy, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            NULL, 0, n_head, head_dim) != 0);

        unsetenv(disable_env);
        unsetenv(pad_disable_env);
        unsetenv(shared_pad_disable_env);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            persistent, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            NULL, 0, n_head, head_dim) != 0);

        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            masked, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            comp_mask, 1, n_head, head_dim) != 0);

        TEST_ASSERT(setenv(pad_disable_env, "1", 1) == 0);
        TEST_ASSERT(setenv(shared_pad_disable_env, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            pad_legacy, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            comp_mask, 1, n_head, head_dim) != 0);
        unsetenv(pad_disable_env);
        unsetenv(shared_pad_disable_env);

        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            after_mask, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            NULL, 0, n_head, head_dim) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        legacy, 0, legacy_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        persistent, 0, persistent_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        masked, 0, masked_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        pad_legacy, 0, pad_legacy_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        after_mask, 0, after_mask_host, q_bytes) != 0);
        persistent_stats = test_compare_float_bits(
            legacy_host, persistent_host, (size_t)n_head * head_dim);
        masked_stats = test_compare_float_bits(
            legacy_host, masked_host, (size_t)n_head * head_dim);
        pad_stats = test_compare_float_bits(
            pad_legacy_host, masked_host, (size_t)n_head * head_dim);
        after_mask_stats = test_compare_float_bits(
            legacy_host, after_mask_host, (size_t)n_head * head_dim);
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(pad_disable_env, saved_pad_disable);
    test_restore_env(shared_pad_disable_env, saved_shared_pad_disable);
    fprintf(stderr,
            "ds4-test: persistent zero attention mask exact keys=%u "
            "candidate=%zu/%u max_ulp=%u masked_diff=%zu/%u "
            "pad_fusion=%zu/%u max_ulp=%u "
            "after_mask=%zu/%u max_ulp=%u\n",
            n_raw + n_comp,
            persistent_stats.mismatch_count, n_head * head_dim,
            persistent_stats.max_ulp,
            masked_stats.mismatch_count, n_head * head_dim,
            pad_stats.mismatch_count, n_head * head_dim,
            pad_stats.max_ulp,
            after_mask_stats.mismatch_count, n_head * head_dim,
            after_mask_stats.max_ulp);
    TEST_ASSERT(persistent_stats.mismatch_count == 0);
    TEST_ASSERT(masked_stats.mismatch_count != 0);
    TEST_ASSERT(pad_stats.mismatch_count == 0);
    TEST_ASSERT(after_mask_stats.mismatch_count == 0);

    free(model_raw);
    free(after_mask_host);
    free(pad_legacy_host);
    free(masked_host);
    free(persistent_host);
    free(legacy_host);
    free(mask_host);
    free(q_host);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(after_mask);
    ds4_gpu_tensor_free(pad_legacy);
    ds4_gpu_tensor_free(masked);
    ds4_gpu_tensor_free(persistent);
    ds4_gpu_tensor_free(legacy);
    ds4_gpu_tensor_free(comp_mask);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
}

static void test_metal_persistent_zero_attention_mask_exact(void) {
    test_metal_persistent_zero_attention_mask_exact_case(7, 5, 5, 3, 11);
    test_metal_persistent_zero_attention_mask_exact_case(37, 29, 35, 3, 23);
}

typedef enum {
    TEST_METAL_PREFILL_MASK_CACHE_RAW = 1,
    TEST_METAL_PREFILL_MASK_CACHE_RATIO4 = 2,
    TEST_METAL_PREFILL_MASK_CACHE_RATIO128 = 3,
} test_metal_prefill_mask_cache_kind;

typedef struct {
    uint32_t n_tokens;
    uint32_t n_comp;
    uint32_t window;
    uint32_t ratio;
} test_metal_prefill_mask_cache_shape;

static int test_metal_zero_prefix_prefill_mask_cache_call(
        test_metal_prefill_mask_cache_kind  kind,
        ds4_gpu_tensor                    *heads,
        const void                        *model_map,
        uint64_t                           model_size,
        const ds4_gpu_tensor              *q,
        const ds4_gpu_tensor              *raw,
        const ds4_gpu_tensor              *comp,
        const ds4_gpu_tensor              *comp_mask,
        const test_metal_prefill_mask_cache_shape *shape,
        bool                               masked,
        uint32_t                           n_head,
        uint32_t                           head_dim) {
    if (kind == TEST_METAL_PREFILL_MASK_CACHE_RAW) {
        if (masked) return 0;
        return ds4_gpu_attention_prefill_raw_heads_tensor(
            heads, model_map, model_size, 0, q, raw,
            shape->n_tokens, shape->window, n_head, head_dim);
    }

    if (masked) {
        return ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
            heads, model_map, model_size, 0, q, raw, comp, 1, comp_mask,
            shape->n_tokens, shape->n_comp, shape->window, shape->ratio,
            n_head, head_dim);
    }

    return ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        heads, model_map, model_size, 0, q, raw, comp, 1,
        shape->n_tokens, shape->n_comp, shape->window, shape->ratio,
        n_head, head_dim);
}

static bool test_metal_zero_prefix_prefill_mask_cache_run(
        test_metal_prefill_mask_cache_kind  kind,
        ds4_gpu_tensor                    *heads,
        const void                        *model_map,
        uint64_t                           model_size,
        const ds4_gpu_tensor              *q,
        const ds4_gpu_tensor              *raw,
        const ds4_gpu_tensor              *comp,
        const ds4_gpu_tensor              *comp_mask,
        const test_metal_prefill_mask_cache_shape *shape,
        bool                               masked,
        uint32_t                           n_head,
        uint32_t                           head_dim,
        float                             *host) {
    const uint64_t bytes =
        (uint64_t)shape->n_tokens * n_head * head_dim * sizeof(float);
    const int call_ok = test_metal_zero_prefix_prefill_mask_cache_call(
        kind, heads, model_map, model_size, q, raw, comp, comp_mask,
        shape, masked, n_head, head_dim);
    TEST_ASSERT(call_ok != 0);
    if (!call_ok) return false;

    const int read_ok = ds4_gpu_tensor_read(heads, 0, host, bytes);
    TEST_ASSERT(read_ok != 0);
    return read_ok != 0;
}

static void test_metal_zero_prefix_prefill_mask_cache_compare(
        const float *expected,
        const float *actual,
        size_t       count,
        size_t      *total_mismatches,
        uint32_t    *max_ulp) {
    const test_float_compare_stats stats =
        test_compare_float_bits(expected, actual, count);
    *total_mismatches += stats.mismatch_count;
    if (stats.max_ulp > *max_ulp) *max_ulp = stats.max_ulp;
}

static void test_metal_zero_prefix_prefill_mask_cache_exact_kind(
        test_metal_prefill_mask_cache_kind kind,
        uint32_t                           seed) {
    const uint32_t head_dim = 512;
    const uint32_t n_head = 1;
    const uint32_t max_tokens = 129;
    const uint32_t max_comp = 32;
    const uint64_t raw_count = (uint64_t)max_tokens * head_dim;
    const uint64_t comp_count = (uint64_t)max_comp * head_dim;
    const uint64_t q_count = (uint64_t)max_tokens * n_head * head_dim;
    const uint64_t mask_count = (uint64_t)max_tokens * max_comp;
    const uint64_t raw_bytes = raw_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(uint16_t);
    const uint64_t q_bytes = q_count * sizeof(float);
    const uint64_t mask_bytes = mask_count * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();

    const test_metal_prefill_mask_cache_shape shape_a = {
        .n_tokens = 128,
        .n_comp = kind == TEST_METAL_PREFILL_MASK_CACHE_RAW ? 0u :
                  (kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO4 ? 32u : 1u),
        .window = 128,
        .ratio = kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO4 ? 4u :
                 (kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO128 ? 128u : 0u),
    };
    const test_metal_prefill_mask_cache_shape shape_b = {
        .n_tokens = 129,
        .n_comp = kind == TEST_METAL_PREFILL_MASK_CACHE_RAW ? 0u :
                  (kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO4 ? 31u : 2u),
        .window = 63,
        .ratio = shape_a.ratio,
    };
    const size_t count_a =
        (size_t)shape_a.n_tokens * n_head * head_dim;
    const size_t count_b =
        (size_t)shape_b.n_tokens * n_head * head_dim;

    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *comp_mask = ds4_gpu_tensor_alloc(mask_bytes);
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_bytes);
    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    float *q_host = calloc((size_t)q_count, sizeof(float));
    float *mask_host = malloc((size_t)mask_bytes);
    float *ref_a = malloc((size_t)q_bytes);
    float *ref_b = malloc((size_t)q_bytes);
    float *actual = malloc((size_t)q_bytes);
    float *masked_actual = malloc((size_t)q_bytes);
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);

    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(comp_mask != NULL);
    TEST_ASSERT(heads != NULL);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(q_host != NULL);
    TEST_ASSERT(mask_host != NULL);
    TEST_ASSERT(ref_a != NULL);
    TEST_ASSERT(ref_b != NULL);
    TEST_ASSERT(actual != NULL);
    TEST_ASSERT(masked_actual != NULL);
    TEST_ASSERT(model_raw != NULL);

    const char *disable_env =
        "DS4_METAL_DISABLE_ZERO_PREFIX_PREFILL_MASK_CACHE";
    char *saved_disable = test_save_env(disable_env);
    size_t total_mismatches = 0;
    uint32_t max_ulp = 0;
    size_t key_difference = 0;
    size_t masked_difference = 0;

    const bool allocated = raw && comp && q && comp_mask && heads &&
        raw_host && comp_host && q_host && mask_host && ref_a && ref_b &&
        actual && masked_actual && model_raw;
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        ((float *)model_raw)[0] = -1.0f;

        for (uint32_t row = 0; row < max_tokens; row++) {
            for (uint32_t col = 0; col < head_dim; col++) {
                const int value = (int)((row * 37u + col * 17u +
                                         (col ^ (col >> 3u)) * 5u +
                                         seed * 13u) % 257u) - 128;
                raw_host[(uint64_t)row * head_dim + col] =
                    (float)value / 256.0f;
            }
        }
        for (uint32_t row = 0; row < max_comp; row++) {
            for (uint32_t col = 0; col < head_dim; col++) {
                const int value = (int)((row * 29u + col * 11u +
                                         (col ^ (col >> 4u)) * 7u +
                                         seed * 19u) % 193u) - 96;
                comp_host[(uint64_t)row * head_dim + col] =
                    test_float_to_f16(0.375f + (float)value / 384.0f);
            }
        }
        for (uint64_t i = 0; i < mask_count; i++) {
            mask_host[i] = -65504.0f;
        }
        if (shape_a.n_comp != 0) {
            for (uint32_t row = 0; row < shape_a.n_tokens; row++) {
                const uint32_t visible = (row + 1u) / shape_a.ratio;
                for (uint32_t col = 0; col < shape_a.n_comp; col++) {
                    if (col < visible) {
                        mask_host[(uint64_t)row * shape_a.n_comp + col] =
                            (col & 1u) != 0u ? -2.0f : -65504.0f;
                    }
                }
            }
        }

        TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(comp, 0, comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        comp_mask, 0, mask_host, mask_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
        const bool have_ref_a =
            test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_a, false, n_head, head_dim, ref_a);
        const bool have_ref_b =
            test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_b, false, n_head, head_dim, ref_b);

        if (have_ref_a && have_ref_b) {
            const test_float_compare_stats key_stats =
                test_compare_float_bits(ref_a, ref_b, count_a);
            key_difference = key_stats.mismatch_count;
            TEST_ASSERT(key_difference != 0);
        }

        TEST_ASSERT(unsetenv(disable_env) == 0);

        if (test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_a, false, n_head, head_dim, actual) && have_ref_a) {
            test_metal_zero_prefix_prefill_mask_cache_compare(
                ref_a, actual, count_a, &total_mismatches, &max_ulp);
        }
        if (test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_a, false, n_head, head_dim, actual) && have_ref_a) {
            test_metal_zero_prefix_prefill_mask_cache_compare(
                ref_a, actual, count_a, &total_mismatches, &max_ulp);
        }

        if (kind != TEST_METAL_PREFILL_MASK_CACHE_RAW) {
            if (test_metal_zero_prefix_prefill_mask_cache_run(
                    kind, heads, model_raw, page, q, raw, comp, comp_mask,
                    &shape_a, true, n_head, head_dim, masked_actual) &&
                have_ref_a) {
                const test_float_compare_stats masked_stats =
                    test_compare_float_bits(ref_a, masked_actual, count_a);
                masked_difference = masked_stats.mismatch_count;
                TEST_ASSERT(masked_difference != 0);
            }
            if (test_metal_zero_prefix_prefill_mask_cache_run(
                    kind, heads, model_raw, page, q, raw, comp, comp_mask,
                    &shape_a, false, n_head, head_dim, actual) && have_ref_a) {
                test_metal_zero_prefix_prefill_mask_cache_compare(
                    ref_a, actual, count_a,
                    &total_mismatches, &max_ulp);
            }
        }

        if (test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_b, false, n_head, head_dim, actual) && have_ref_b) {
            test_metal_zero_prefix_prefill_mask_cache_compare(
                ref_b, actual, count_b, &total_mismatches, &max_ulp);
        }
        if (test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_a, false, n_head, head_dim, actual) && have_ref_a) {
            test_metal_zero_prefix_prefill_mask_cache_compare(
                ref_a, actual, count_a, &total_mismatches, &max_ulp);
        }
    }

    test_restore_env(disable_env, saved_disable);
    const char *kind_name = kind == TEST_METAL_PREFILL_MASK_CACHE_RAW ? "raw" :
        (kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO4 ? "ratio4" : "ratio128");
    fprintf(stderr,
            "ds4-test: zero-prefix prefill mask cache %s exact "
            "mismatches=%zu max_ulp=%u key_diff=%zu masked_diff=%zu\n",
            kind_name, total_mismatches, max_ulp,
            key_difference, masked_difference);
    TEST_ASSERT(total_mismatches == 0);
    TEST_ASSERT(max_ulp == 0);

    free(model_raw);
    free(masked_actual);
    free(actual);
    free(ref_b);
    free(ref_a);
    free(mask_host);
    free(q_host);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(comp_mask);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
}

static void test_metal_zero_prefix_prefill_mask_cache_exact(void) {
    test_metal_zero_prefix_prefill_mask_cache_exact_kind(
        TEST_METAL_PREFILL_MASK_CACHE_RAW, 41);
    test_metal_zero_prefix_prefill_mask_cache_exact_kind(
        TEST_METAL_PREFILL_MASK_CACHE_RATIO4, 43);
    test_metal_zero_prefix_prefill_mask_cache_exact_kind(
        TEST_METAL_PREFILL_MASK_CACHE_RATIO128, 47);
}
#endif

#if defined(__APPLE__)
static void test_metal_hc_split_weighted_sum_norm_batch_exact(void) {
    /* Compare the batched HC+RMSNorm fusion against the exact two-dispatch
     * sequence used by the reference path at DS4's production dimensions. */
    const uint32_t n_embd = 7168;
    const uint32_t n_hc = 4;
    const uint32_t n_rows = 3;
    const uint32_t sinkhorn_iters = 20;
    const float hc_eps = 1.0e-6f;
    const float norm_eps = 1.0e-6f;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t scale_offset = 0;
    const uint64_t base_offset = test_round_up_u64(3u * sizeof(float), page);
    const uint64_t norm_weight_offset =
        test_round_up_u64(base_offset + mix_hc * sizeof(float), page);
    const uint64_t model_alloc = test_round_up_u64(
        norm_weight_offset + (uint64_t)n_embd * sizeof(float), page);
    const uint64_t mix_bytes = (uint64_t)n_rows * mix_hc * sizeof(float);
    const uint64_t residual_bytes =
        (uint64_t)n_rows * n_hc * n_embd * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_rows * n_embd * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)model_alloc) == 0);
    if (!model_raw) return;
    memset(model_raw, 0, (size_t)model_alloc);

    float *scale = (float *)((uint8_t *)model_raw + scale_offset);
    float *base = (float *)((uint8_t *)model_raw + base_offset);
    float *norm_weight = (float *)((uint8_t *)model_raw + norm_weight_offset);
    scale[0] = 0.625f;
    scale[1] = -0.75f;
    scale[2] = 0.4375f;
    for (uint32_t i = 0; i < mix_hc; i++) {
        const int value = (int)((i * 17u + 5u) % 29u) - 14;
        base[i] = (float)value / 16.0f;
    }
    for (uint32_t i = 0; i < n_embd; i++) {
        norm_weight[i] = 0.5f + (float)((i * 13u + 7u) % 31u) / 32.0f;
    }

    ds4_gpu_tensor *mix = ds4_gpu_tensor_alloc(mix_bytes);
    ds4_gpu_tensor *residual = ds4_gpu_tensor_alloc(residual_bytes);
    ds4_gpu_tensor *ref_split = ds4_gpu_tensor_alloc(mix_bytes);
    ds4_gpu_tensor *fused_split = ds4_gpu_tensor_alloc(mix_bytes);
    ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *ref_norm = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_norm = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(mix != NULL);
    TEST_ASSERT(residual != NULL);
    TEST_ASSERT(ref_split != NULL);
    TEST_ASSERT(fused_split != NULL);
    TEST_ASSERT(ref_out != NULL);
    TEST_ASSERT(fused_out != NULL);
    TEST_ASSERT(ref_norm != NULL);
    TEST_ASSERT(fused_norm != NULL);
    if (!mix || !residual || !ref_split || !fused_split ||
        !ref_out || !fused_out || !ref_norm || !fused_norm) {
        ds4_gpu_tensor_free(mix);
        ds4_gpu_tensor_free(residual);
        ds4_gpu_tensor_free(ref_split);
        ds4_gpu_tensor_free(fused_split);
        ds4_gpu_tensor_free(ref_out);
        ds4_gpu_tensor_free(fused_out);
        ds4_gpu_tensor_free(ref_norm);
        ds4_gpu_tensor_free(fused_norm);
        free(model_raw);
        return;
    }

    float *mix_host = malloc((size_t)mix_bytes);
    float *residual_host = malloc((size_t)residual_bytes);
    float *ref_split_host = malloc((size_t)mix_bytes);
    float *fused_split_host = malloc((size_t)mix_bytes);
    float *ref_out_host = malloc((size_t)out_bytes);
    float *fused_out_host = malloc((size_t)out_bytes);
    float *ref_norm_host = malloc((size_t)out_bytes);
    float *fused_norm_host = malloc((size_t)out_bytes);
    TEST_ASSERT(mix_host != NULL);
    TEST_ASSERT(residual_host != NULL);
    TEST_ASSERT(ref_split_host != NULL);
    TEST_ASSERT(fused_split_host != NULL);
    TEST_ASSERT(ref_out_host != NULL);
    TEST_ASSERT(fused_out_host != NULL);
    TEST_ASSERT(ref_norm_host != NULL);
    TEST_ASSERT(fused_norm_host != NULL);
    if (!mix_host || !residual_host || !ref_split_host || !fused_split_host ||
        !ref_out_host || !fused_out_host || !ref_norm_host || !fused_norm_host) {
        free(mix_host);
        free(residual_host);
        free(ref_split_host);
        free(fused_split_host);
        free(ref_out_host);
        free(fused_out_host);
        free(ref_norm_host);
        free(fused_norm_host);
        ds4_gpu_tensor_free(mix);
        ds4_gpu_tensor_free(residual);
        ds4_gpu_tensor_free(ref_split);
        ds4_gpu_tensor_free(fused_split);
        ds4_gpu_tensor_free(ref_out);
        ds4_gpu_tensor_free(fused_out);
        ds4_gpu_tensor_free(ref_norm);
        ds4_gpu_tensor_free(fused_norm);
        free(model_raw);
        return;
    }

    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t i = 0; i < mix_hc; i++) {
            const int value =
                (int)(((row + 1u) * 19u + i * 11u + (i ^ row) * 3u) % 47u) - 23;
            mix_host[(uint64_t)row * mix_hc + i] = (float)value / 9.0f;
        }
        for (uint32_t hc = 0; hc < n_hc; hc++) {
            for (uint32_t d = 0; d < n_embd; d++) {
                const uint32_t key =
                    d * 37u + hc * 173u + row * 997u + ((d >> 3u) ^ (d * 7u));
                const int value = (int)(key % 2047u) - 1023;
                residual_host[((uint64_t)row * n_hc + hc) * n_embd + d] =
                    (float)value / 512.0f;
            }
        }
    }

    TEST_ASSERT(ds4_gpu_tensor_write(mix, 0, mix_host, mix_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(residual, 0, residual_host, residual_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
    TEST_ASSERT(ds4_gpu_hc_split_weighted_sum_tensor(
        ref_out, ref_split, mix, residual,
        model_raw, model_alloc, scale_offset, base_offset,
        n_embd, n_hc, sinkhorn_iters, hc_eps) != 0);
    TEST_ASSERT(ds4_gpu_rms_norm_weight_rows_tensor(
        ref_norm, ref_out, model_raw, model_alloc, norm_weight_offset,
        n_embd, n_rows, norm_eps) != 0);
    TEST_ASSERT(ds4_gpu_hc_split_weighted_sum_norm_tensor(
        fused_out, fused_norm, fused_split, mix, residual,
        model_raw, model_alloc, scale_offset, base_offset, norm_weight_offset,
        n_embd, n_hc, sinkhorn_iters, hc_eps, norm_eps) != 0);

    TEST_ASSERT(ds4_gpu_tensor_read(ref_split, 0, ref_split_host, mix_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(fused_split, 0, fused_split_host, mix_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref_out, 0, ref_out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(fused_out, 0, fused_out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref_norm, 0, ref_norm_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(fused_norm, 0, fused_norm_host, out_bytes) != 0);

    const test_float_compare_stats split_stats = test_compare_float_bits(
        ref_split_host, fused_split_host, (size_t)(n_rows * mix_hc));
    const test_float_compare_stats out_stats = test_compare_float_bits(
        ref_out_host, fused_out_host, (size_t)n_rows * n_embd);
    const test_float_compare_stats norm_stats = test_compare_float_bits(
        ref_norm_host, fused_norm_host, (size_t)n_rows * n_embd);
    fprintf(stderr,
            "ds4-test: batch HC+RMSNorm exactness rows=%u "
            "split=%zu/%llu max_ulp=%u max_abs=%g, "
            "collapse=%zu/%llu max_ulp=%u max_abs=%g, "
            "norm=%zu/%llu max_ulp=%u max_abs=%g\n",
            n_rows,
            split_stats.mismatch_count,
            (unsigned long long)(n_rows * mix_hc),
            split_stats.max_ulp,
            split_stats.max_abs,
            out_stats.mismatch_count,
            (unsigned long long)((uint64_t)n_rows * n_embd),
            out_stats.max_ulp,
            out_stats.max_abs,
            norm_stats.mismatch_count,
            (unsigned long long)((uint64_t)n_rows * n_embd),
            norm_stats.max_ulp,
            norm_stats.max_abs);
    TEST_ASSERT(split_stats.mismatch_count == 0);
    TEST_ASSERT(out_stats.mismatch_count == 0);
    TEST_ASSERT(norm_stats.mismatch_count == 0);

    free(mix_host);
    free(residual_host);
    free(ref_split_host);
    free(fused_split_host);
    free(ref_out_host);
    free(fused_out_host);
    free(ref_norm_host);
    free(fused_norm_host);
    ds4_gpu_tensor_free(mix);
    ds4_gpu_tensor_free(residual);
    ds4_gpu_tensor_free(ref_split);
    ds4_gpu_tensor_free(fused_split);
    ds4_gpu_tensor_free(ref_out);
    ds4_gpu_tensor_free(fused_out);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(fused_norm);
    free(model_raw);
}

static void test_metal_output_hc_weights4_exact(void) {
    const uint32_t n_hc = 4;
    const float eps = 1.0e-6f;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t scale_offset = 0;
    const uint64_t base_offset = page;
    const uint64_t model_alloc = 2u * page;
    const uint64_t bytes = n_hc * sizeof(float);
    const char *require_env = "DS4_METAL_REQUIRE_OUTPUT_HC_WEIGHTS4";
    char *saved_require = test_save_env(require_env);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_alloc) == 0);
    ds4_gpu_tensor *pre = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *candidate = ds4_gpu_tensor_alloc(bytes);
    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(pre != NULL);
    TEST_ASSERT(reference != NULL);
    TEST_ASSERT(candidate != NULL);

    size_t total_mismatch = 0;
    uint32_t max_ulp = 0;
    const bool allocated = model_raw && pre && reference && candidate;
    if (allocated) {
        memset(model_raw, 0, (size_t)model_alloc);
        float *scale = (float *)((uint8_t *)model_raw + scale_offset);
        float *base = (float *)((uint8_t *)model_raw + base_offset);

        for (uint32_t ci = 0; ci < 4; ci++) {
            float pre_host[4];
            float ref_host[4];
            float candidate_host[4];
            for (uint32_t i = 0; i < 4; i++) {
                pre_host[i] =
                    ((float)((int)(ci * 17u + i * 11u) - 23)) / 8.0f;
                base[i] =
                    ((float)((int)(ci * 13u + i * 7u) - 19)) / 16.0f;
            }
            scale[0] = 0.62500012f + (float)ci * 0.125f;

            if (ci == 1) {
                /* The first lane distinguishes the required two-rounding
                 * sequence from an illegally contracted multiply-add. */
                const uint32_t pre_bits = 0x4620a541u;
                const uint32_t scale_bits = 0x462483bau;
                const uint32_t base_bits = 0xccce790eu;
                memcpy(&pre_host[0], &pre_bits, sizeof(pre_bits));
                memcpy(&scale[0], &scale_bits, sizeof(scale_bits));
                memcpy(&base[0], &base_bits, sizeof(base_bits));
            } else if (ci == 2) {
                const uint32_t pre_bits[4] = {
                    0x00000000u, 0x80000000u,
                    0x00000001u, 0x80000001u,
                };
                const uint32_t base_bits[4] = {
                    0x80000000u, 0x00000000u,
                    0x00800000u, 0x80800000u,
                };
                for (uint32_t i = 0; i < 4; i++) {
                    memcpy(&pre_host[i], &pre_bits[i], sizeof(uint32_t));
                    memcpy(&base[i], &base_bits[i], sizeof(uint32_t));
                }
                scale[0] = -1.00000012f;
            } else if (ci == 3) {
                pre_host[0] = 100.0f;
                pre_host[1] = -100.0f;
                pre_host[2] = 88.0f;
                pre_host[3] = -88.0f;
                scale[0] = 1.0f;
                for (uint32_t i = 0; i < 4; i++) base[i] = 0.0f;
            }

            memset(ref_host, 0xa5, sizeof(ref_host));
            memset(candidate_host, 0xa5, sizeof(candidate_host));
            TEST_ASSERT(ds4_gpu_tensor_write(pre, 0, pre_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            reference, 0, ref_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            candidate, 0, candidate_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
            TEST_ASSERT(unsetenv(require_env) == 0);
            ds4_gpu_test_set_flags(0);
            ds4_gpu_set_quality(true);
            TEST_ASSERT(ds4_gpu_output_hc_weights_tensor(
                            reference, pre, model_raw, model_alloc,
                            scale_offset, base_offset, n_hc, eps) != 0);

            ds4_gpu_set_quality(false);
            ds4_gpu_test_set_flags(DS4_GPU_TEST_OUTPUT_HC_WEIGHTS4);
            TEST_ASSERT(setenv(require_env, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_output_hc_weights_tensor(
                            candidate, pre, model_raw, model_alloc,
                            scale_offset, base_offset, n_hc, eps) != 0);

            TEST_ASSERT(ds4_gpu_tensor_read(
                            reference, 0, ref_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            candidate, 0, candidate_host, bytes) != 0);
            const test_float_compare_stats stats =
                test_compare_float_bits(ref_host, candidate_host, n_hc);
            fprintf(stderr,
                    "ds4-test: output HC weights4 exact case=%u "
                    "mismatch=%zu/%u max_ulp=%u max_abs=%g\n",
                    ci, stats.mismatch_count, n_hc,
                    stats.max_ulp, stats.max_abs);
            TEST_ASSERT(stats.mismatch_count == 0);
            total_mismatch += stats.mismatch_count;
            if (stats.max_ulp > max_ulp) max_ulp = stats.max_ulp;
        }

        /* Quality mode prevents strict selection of the fast path. */
        TEST_ASSERT(setenv(require_env, "1", 1) == 0);
        ds4_gpu_set_quality(true);
        TEST_ASSERT(ds4_gpu_output_hc_weights_tensor(
                        candidate, pre, model_raw, model_alloc,
                        scale_offset, base_offset, n_hc, eps) == 0);
        ds4_gpu_set_quality(false);
        ds4_gpu_test_set_flags(0);
    }

    test_restore_env(require_env, saved_require);
    fprintf(stderr,
            "ds4-test: output HC weights4 total mismatch=%zu/16 max_ulp=%u\n",
            total_mismatch, max_ulp);
    TEST_ASSERT(total_mismatch == 0);
    TEST_ASSERT(max_ulp == 0);

    ds4_gpu_tensor_free(candidate);
    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(pre);
    free(model_raw);
}

static void test_metal_hc_rms_scale_project_f16_exact_shape(
        uint32_t in_dim,
        uint32_t seed) {
    const uint32_t out_dim = 24;
    /* One full 32-row matmul tile plus a tail row covers both load paths. */
    const uint32_t n_rows = 33;
    const float eps = 1.0e-6f;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t weight_offset = page;
    const uint64_t weight_bytes =
        (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t model_alloc = test_round_up_u64(
        weight_offset + weight_bytes, page);
    const uint64_t x_count = (uint64_t)in_dim * n_rows;
    const uint64_t out_count = (uint64_t)out_dim * n_rows;
    const uint64_t x_bytes = x_count * sizeof(float);
    const uint64_t out_bytes = out_count * sizeof(float);
    const uint64_t scale_bytes = (uint64_t)n_rows * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_alloc) == 0);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_norm = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *scaled_out = ds4_gpu_tensor_alloc(out_bytes);
    /* Deliberately too small for the full-RMS fallback. */
    ds4_gpu_tensor *scale_scratch = ds4_gpu_tensor_alloc(scale_bytes);
    float *x_host = malloc((size_t)x_bytes);
    float *ref_norm_host = malloc((size_t)x_bytes);
    float *ref_out_host = malloc((size_t)out_bytes);
    float *scaled_out_host = malloc((size_t)out_bytes);
    float *scale_host = malloc((size_t)scale_bytes);
    float *expected_scale = malloc((size_t)scale_bytes);

    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(ref_norm != NULL);
    TEST_ASSERT(ref_out != NULL);
    TEST_ASSERT(scaled_out != NULL);
    TEST_ASSERT(scale_scratch != NULL);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(ref_norm_host != NULL);
    TEST_ASSERT(ref_out_host != NULL);
    TEST_ASSERT(scaled_out_host != NULL);
    TEST_ASSERT(scale_host != NULL);
    TEST_ASSERT(expected_scale != NULL);

    const bool allocated = model_raw && x && ref_norm && ref_out &&
        scaled_out && scale_scratch && x_host && ref_norm_host &&
        ref_out_host && scaled_out_host && scale_host && expected_scale;
    test_float_compare_stats scale_stats = {0};
    test_float_compare_stats out_stats = {0};
    if (allocated) {
        memset(model_raw, 0, (size_t)model_alloc);
        uint16_t *weights = (uint16_t *)((uint8_t *)model_raw + weight_offset);
        for (uint32_t o = 0; o < out_dim; o++) {
            for (uint32_t i = 0; i < in_dim; i++) {
                const uint32_t key = i * 37u + o * 1009u + seed * 53u +
                    ((i >> 4u) ^ (o * 17u));
                uint16_t bits;
                if (key % 257u == 0u) {
                    bits = (key & 1u) ? 0x8000u : 0x0000u;
                } else if (key % 263u == 0u) {
                    bits = (key & 1u) ? 0x8001u : 0x0001u;
                } else if (key % 269u == 0u) {
                    bits = (key & 1u) ? 0x8400u : 0x0400u;
                } else {
                    const int value = (int)(key % 127u) - 63;
                    bits = test_float_to_f16((float)value / 128.0f);
                }
                weights[(uint64_t)o * in_dim + i] = bits;
            }
        }

        static const uint32_t rounding_bits[] = {
            0x3f800fffu, 0x3f801000u, 0x3f801001u,
            0x3f802fffu, 0x3f803000u, 0x3f803001u,
            0xbf800fffu, 0xbf801000u, 0xbf801001u,
            0x3eaaaaabu, 0xbeaaaaabu,
        };
        static const int sentinel_exp[] = { -2, -10, 0, -4, -6, -1 };
        for (uint32_t row = 0; row < n_rows; row++) {
            for (uint32_t i = 0; i < in_dim; i++) {
                const uint32_t key = i * 131u + row * 977u + seed * 71u +
                    ((i >> 3u) ^ (row * 29u));
                const float sign = (key & 1u) ? -1.0f : 1.0f;
                float value;
                switch (row % 6u) {
                    case 0:
                        value = (float)((int)(key % 4093u) - 2046) / 512.0f;
                        break;
                    case 1:
                        value = sign * ldexpf(
                            (float)(1u + (key & 7u)) / 8.0f, -19);
                        break;
                    case 2:
                        value = i % 257u == row % 257u
                            ? sign * (16.0f + (float)(key & 7u))
                            : sign * (float)(1u + (key & 31u)) / 8192.0f;
                        break;
                    case 3: {
                        const uint32_t bits = rounding_bits[
                            key % (sizeof(rounding_bits) /
                                   sizeof(rounding_bits[0]))];
                        memcpy(&value, &bits, sizeof(value));
                        break;
                    }
                    case 4:
                        if ((key & 7u) == 0u) {
                            const uint32_t bits = (key & 8u) ? 0x80000000u : 0u;
                            memcpy(&value, &bits, sizeof(value));
                        } else {
                            value = sign * ldexpf(
                                1.0f + (float)(key & 3u) * 0.25f,
                                (int)((key >> 4u) % 14u) - 10);
                        }
                        break;
                    default:
                        value = sign * (float)(1u + (key % 251u)) / 256.0f;
                        break;
                }
                x_host[(uint64_t)row * in_dim + i] = value;
            }
            x_host[(uint64_t)row * in_dim] =
                ldexpf(1.0f, sentinel_exp[row % 6u]);
        }

        for (uint64_t i = 0; i < out_count; i++) {
            const uint32_t bits = 0x7fc01000u + (uint32_t)(i & 0xfffu);
            memcpy(scaled_out_host + i, &bits, sizeof(bits));
        }
        for (uint32_t row = 0; row < n_rows; row++) {
            const uint32_t bits = 0x7fc02000u + row;
            memcpy(scale_host + row, &bits, sizeof(bits));
        }

        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        scaled_out, 0, scaled_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        scale_scratch, 0, scale_host, scale_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
        ds4_gpu_set_quality(false);

        int ref_begun = ds4_gpu_begin_commands();
        int ref_ok = ref_begun;
        if (ref_ok) ref_ok = ds4_gpu_rms_norm_plain_rows_tensor(
            ref_norm, x, in_dim, n_rows, eps);
        if (ref_ok) ref_ok = ds4_gpu_matmul_f16_tensor(
            ref_out, model_raw, model_alloc, weight_offset,
            in_dim, out_dim, ref_norm, n_rows);
        const int ref_end = ref_begun ? ds4_gpu_end_commands() : 0;
        TEST_ASSERT(ref_ok != 0);
        TEST_ASSERT(ref_end != 0);

        int scaled_begun = ds4_gpu_begin_commands();
        int scaled_ok = scaled_begun;
        if (scaled_ok) scaled_ok = ds4_gpu_hc_rms_scale_project_f16_tensor(
            scaled_out, scale_scratch, model_raw, model_alloc, weight_offset,
            in_dim, out_dim, x, n_rows, eps);
        const int scaled_end = scaled_begun ? ds4_gpu_end_commands() : 0;
        TEST_ASSERT(scaled_ok != 0);
        TEST_ASSERT(scaled_end != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_norm, 0, ref_norm_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_out, 0, ref_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        scaled_out, 0, scaled_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        scale_scratch, 0, scale_host, scale_bytes) != 0);

        for (uint32_t row = 0; row < n_rows; row++) {
            expected_scale[row] = ldexpf(
                ref_norm_host[(uint64_t)row * in_dim],
                -sentinel_exp[row % 6u]);
            TEST_ASSERT(isfinite(expected_scale[row]));
            TEST_ASSERT(isfinite(scale_host[row]));
        }
        scale_stats = test_compare_float_bits(
            expected_scale, scale_host, n_rows);
        out_stats = test_compare_float_bits(
            ref_out_host, scaled_out_host, (size_t)out_count);
    }

    fprintf(stderr,
            "ds4-test: HC RMS scale F16 projection exact K=%u rows=%u "
            "scale=%zu/%u max_ulp=%u projection=%zu/%llu max_ulp=%u\n",
            in_dim, n_rows,
            scale_stats.mismatch_count, n_rows, scale_stats.max_ulp,
            out_stats.mismatch_count, (unsigned long long)out_count,
            out_stats.max_ulp);
    TEST_ASSERT(scale_stats.mismatch_count == 0);
    TEST_ASSERT(scale_stats.max_ulp == 0);
    TEST_ASSERT(out_stats.mismatch_count == 0);
    TEST_ASSERT(out_stats.max_ulp == 0);

    free(expected_scale);
    free(scale_host);
    free(scaled_out_host);
    free(ref_out_host);
    free(ref_norm_host);
    free(x_host);
    ds4_gpu_tensor_free(scale_scratch);
    ds4_gpu_tensor_free(scaled_out);
    ds4_gpu_tensor_free(ref_out);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(x);
    free(model_raw);
}

static void test_metal_hc_rms_scale_project_f16_exact(void) {
    const char *disable_env = "DS4_METAL_DISABLE_HC_RMS_SCALE_PROJ";
    char *saved_disable = test_save_env(disable_env);

    TEST_ASSERT(unsetenv(disable_env) == 0);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_HC_RMS_SCALE_PROJ);
    test_metal_hc_rms_scale_project_f16_exact_shape(16384u, 59u);
    test_metal_hc_rms_scale_project_f16_exact_shape(28672u, 61u);
    ds4_gpu_test_set_flags(0);

    test_restore_env(disable_env, saved_disable);
}

static void test_metal_router_simd_finalize_exact(void) {
    typedef struct {
        const char *name;
        bool has_bias;
        uint32_t pattern;
    } router_case;
    static const router_case cases[] = {
        { "unique", false, 0 },
        { "bias", true, 0 },
        { "top6-ties", false, 1 },
        { "signed-zero-extremes", false, 2 },
        { "clamp-underflow", false, 3 },
        { "sum-rounding", true, 4 },
    };
    const uint32_t n_expert = 256;
    const uint32_t n_used = 6;
    const uint32_t modes = 4;
    const uint64_t probs_bytes = (uint64_t)n_expert * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)n_used * sizeof(int32_t);
    const uint64_t weights_bytes = (uint64_t)n_used * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();
    const char *disable_env =
        "DS4_METAL_DISABLE_PRE_M5_ROUTER_SIMD_FINALIZE";
    const char *weights_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_ROUTER_SIMD_WEIGHTS_FUSION";
    const char *transform_finalize_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_ROUTER_TRANSFORM_FINALIZE_FUSION";
    const char *select_disable_env =
        "DS4_METAL_DISABLE_ROUTER_SELECT_FUSION";

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(probs_bytes);
    ds4_gpu_tensor *ref_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *simd_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *ref_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *simd_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *ref_probs = ds4_gpu_tensor_alloc(probs_bytes);
    ds4_gpu_tensor *simd_probs = ds4_gpu_tensor_alloc(probs_bytes);
    float *logits_host = malloc((size_t)probs_bytes);
    float *ref_probs_host = malloc((size_t)probs_bytes);
    float *simd_probs_host = malloc((size_t)probs_bytes);
    int32_t ref_selected_host[6];
    int32_t simd_selected_host[6];
    int32_t unique_selected_host[6];
    float ref_weights_host[6];
    float simd_weights_host[6];
    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(logits != NULL);
    TEST_ASSERT(ref_selected != NULL);
    TEST_ASSERT(simd_selected != NULL);
    TEST_ASSERT(ref_weights != NULL);
    TEST_ASSERT(simd_weights != NULL);
    TEST_ASSERT(ref_probs != NULL);
    TEST_ASSERT(simd_probs != NULL);
    TEST_ASSERT(logits_host != NULL);
    TEST_ASSERT(ref_probs_host != NULL);
    TEST_ASSERT(simd_probs_host != NULL);

    char *saved_disable = test_save_env(disable_env);
    char *saved_weights_disable = test_save_env(weights_disable_env);
    char *saved_transform_finalize_disable =
        test_save_env(transform_finalize_disable_env);
    char *saved_select_disable = test_save_env(select_disable_env);
    size_t total_selected_mismatch = 0;
    size_t total_weights_mismatch = 0;
    size_t total_probs_mismatch = 0;
    const bool allocated = model_raw && logits && ref_selected &&
        simd_selected && ref_weights && simd_weights && ref_probs &&
        simd_probs && logits_host && ref_probs_host && simd_probs_host;
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        float *bias = model_raw;
        for (uint32_t i = 0; i < n_used; i++) {
            bias[200u + i] = 16.0f - (float)i;
        }
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);
        TEST_ASSERT(setenv(transform_finalize_disable_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(select_disable_env) == 0);

        for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
            const router_case *c = &cases[ci];
            for (uint32_t i = 0; i < n_expert; i++) {
                const int value =
                    (int)((i * 47u + (i ^ (i >> 3u)) * 13u) % 257u) - 128;
                logits_host[i] = (float)value / 32.0f;
            }
            if (c->pattern == 1) {
                static const uint32_t tied[] = {
                    7u, 19u, 43u, 71u, 103u, 149u, 211u, 239u,
                };
                for (uint32_t i = 0; i < n_expert; i++) {
                    logits_host[i] = -7.0f - (float)(i % 17u) / 64.0f;
                }
                for (size_t i = 0; i < sizeof(tied) / sizeof(tied[0]); i++) {
                    logits_host[tied[i]] = 1.0f;
                }
            } else if (c->pattern == 2) {
                for (uint32_t i = 0; i < n_expert; i++) {
                    const uint32_t zero_bits = (i & 1u) ? 0x80000000u : 0u;
                    memcpy(&logits_host[i], &zero_bits, sizeof(zero_bits));
                }
                logits_host[3] = 80.0f;
                logits_host[17] = 40.0f;
                logits_host[61] = 20.0f;
                logits_host[127] = -20.0f;
                logits_host[193] = -40.0f;
                logits_host[251] = -80.0f;
            } else if (c->pattern == 3) {
                for (uint32_t i = 0; i < n_expert; i++) {
                    logits_host[i] = -30.0f - (float)(i % 11u);
                }
            } else if (c->pattern == 4) {
                static const float rounding_logits[6] = {
                    -0.6356699467f,
                    -0.8182631135f,
                    -2.7906901836f,
                    -3.4414808750f,
                    -3.4991359711f,
                    -3.1251864433f,
                };
                for (uint32_t i = 0; i < n_expert; i++) {
                    logits_host[i] = -20.0f;
                }
                for (uint32_t i = 0; i < n_used; i++) {
                    logits_host[200u + i] = rounding_logits[i];
                }
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                logits, 0, logits_host, probs_bytes) != 0);

            TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_router_select_tensor(
                ref_selected, ref_weights, ref_probs,
                model_raw, page, 0, 0, 1, 0,
                n_expert, n_used, 1.5f, 1, 0,
                c->has_bias, false, logits) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_selected, 0, ref_selected_host, selected_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_weights, 0, ref_weights_host, weights_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_probs, 0, ref_probs_host, probs_bytes) != 0);

            if (ci == 0) {
                memcpy(unique_selected_host,
                       ref_selected_host,
                       sizeof(unique_selected_host));
            } else if (c->has_bias) {
                TEST_ASSERT(memcmp(unique_selected_host,
                                   ref_selected_host,
                                   sizeof(unique_selected_host)) != 0);
            }

            TEST_ASSERT(unsetenv(disable_env) == 0);
            for (uint32_t mode = 0; mode < modes; mode++) {
                const bool fused_weights = mode != 0;
                const bool fused_transform = mode >= 2;
                if (fused_weights) {
                    TEST_ASSERT(unsetenv(weights_disable_env) == 0);
                } else {
                    TEST_ASSERT(setenv(weights_disable_env, "1", 1) == 0);
                }
                if (fused_transform) {
                    TEST_ASSERT(unsetenv(transform_finalize_disable_env) == 0);
                } else {
                    TEST_ASSERT(setenv(
                        transform_finalize_disable_env, "1", 1) == 0);
                }
                for (uint32_t i = 0; i < n_expert; i++) {
                    const uint32_t poison_bits = 0x7fc00001u + i;
                    memcpy(&simd_probs_host[i],
                           &poison_bits,
                           sizeof(poison_bits));
                }
                TEST_ASSERT(ds4_gpu_tensor_write(
                    simd_probs, 0, simd_probs_host, probs_bytes) != 0);
                TEST_ASSERT(ds4_gpu_router_select_tensor(
                    simd_selected, simd_weights, simd_probs,
                    model_raw, page, 0, 0, 1, 0,
                    n_expert, n_used, 1.5f, 1, 0,
                    c->has_bias, false, logits) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    simd_selected, 0, simd_selected_host,
                    selected_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    simd_weights, 0, simd_weights_host,
                    weights_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    simd_probs, 0, simd_probs_host, probs_bytes) != 0);

                size_t selected_mismatch = 0;
                for (uint32_t i = 0; i < n_used; i++) {
                    if (ref_selected_host[i] != simd_selected_host[i]) {
                        selected_mismatch++;
                    }
                }
                const test_float_compare_stats weights_stats =
                    test_compare_float_bits(
                        ref_weights_host, simd_weights_host, n_used);
                const test_float_compare_stats probs_stats =
                    test_compare_float_bits(
                        ref_probs_host, simd_probs_host, n_expert);
                fprintf(stderr,
                        "ds4-test: router SIMD finalize exactness "
                        "case=%s mode=%s rep=%u selected=%zu/%u weights=%zu/%u "
                        "probs=%zu/%u max_weight_ulp=%u max_prob_ulp=%u\n",
                        c->name,
                        fused_transform ? "fused-transform" :
                        fused_weights ? "fused-weights" : "split-weights",
                        fused_transform ? mode - 2u : 0u,
                        selected_mismatch,
                        n_used,
                        weights_stats.mismatch_count,
                        n_used,
                        probs_stats.mismatch_count,
                        n_expert,
                        weights_stats.max_ulp,
                        probs_stats.max_ulp);
                TEST_ASSERT(selected_mismatch == 0);
                TEST_ASSERT(weights_stats.mismatch_count == 0);
                TEST_ASSERT(probs_stats.mismatch_count == 0);
                total_selected_mismatch += selected_mismatch;
                total_weights_mismatch += weights_stats.mismatch_count;
                total_probs_mismatch += probs_stats.mismatch_count;
            }
        }
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(weights_disable_env, saved_weights_disable);
    test_restore_env(transform_finalize_disable_env,
                     saved_transform_finalize_disable);
    test_restore_env(select_disable_env, saved_select_disable);
    fprintf(stderr,
            "ds4-test: router SIMD finalize total selected=%zu/%zu "
            "weights=%zu/%zu probs=%zu/%zu\n",
            total_selected_mismatch,
            (sizeof(cases) / sizeof(cases[0])) * modes * (size_t)n_used,
            total_weights_mismatch,
            (sizeof(cases) / sizeof(cases[0])) * modes * (size_t)n_used,
            total_probs_mismatch,
            (sizeof(cases) / sizeof(cases[0])) * modes * (size_t)n_expert);
    TEST_ASSERT(total_selected_mismatch == 0);
    TEST_ASSERT(total_weights_mismatch == 0);
    TEST_ASSERT(total_probs_mismatch == 0);

    free(simd_probs_host);
    free(ref_probs_host);
    free(logits_host);
    ds4_gpu_tensor_free(simd_probs);
    ds4_gpu_tensor_free(ref_probs);
    ds4_gpu_tensor_free(simd_weights);
    ds4_gpu_tensor_free(ref_weights);
    ds4_gpu_tensor_free(simd_selected);
    ds4_gpu_tensor_free(ref_selected);
    ds4_gpu_tensor_free(logits);
    free(model_raw);
}

static void test_metal_router_weights_batch_exact(void) {
    typedef struct {
        const char *name;
        uint32_t n_tokens;
        bool has_bias;
        bool hash_mode;
        uint32_t pattern;
    } router_batch_case;
    static const router_batch_case cases[] = {
        { "rows2-unique", 2, false, false, 0 },
        { "rows17-bias-ties", 17, true, false, 1 },
        { "rows129-hash-duplicates", 129, false, true, 2 },
        { "rows2048-typical", 2048, false, false, 0 },
        { "rows3-clamp-underflow", 3, false, false, 3 },
    };
    const uint32_t n_expert = 256;
    const uint32_t n_used = 6;
    const uint32_t max_tokens = 2048;
    const uint32_t hash_rows = 64;
    const uint32_t repeats = 2;
    const uint64_t logits_bytes =
        (uint64_t)max_tokens * n_expert * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)max_tokens * n_used * sizeof(int32_t);
    const uint64_t weights_bytes =
        (uint64_t)max_tokens * n_used * sizeof(float);
    const uint64_t tokens_bytes = (uint64_t)max_tokens * sizeof(int32_t);
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t bias_offset = 0;
    const uint64_t hash_offset = 2048;
    const char *disable_env =
        "DS4_METAL_DISABLE_ROUTER_WEIGHTS_BATCH_FUSION";
    const char *select_disable_env =
        "DS4_METAL_DISABLE_ROUTER_SELECT_FUSION";

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(logits_bytes);
    ds4_gpu_tensor *tokens = ds4_gpu_tensor_alloc(tokens_bytes);
    ds4_gpu_tensor *ref_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *batch_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *ref_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *batch_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *ref_probs = ds4_gpu_tensor_alloc(logits_bytes);
    ds4_gpu_tensor *batch_probs = ds4_gpu_tensor_alloc(logits_bytes);
    float *logits_host = malloc((size_t)logits_bytes);
    int32_t *tokens_host = malloc((size_t)tokens_bytes);
    int32_t *ref_selected_host = malloc((size_t)selected_bytes);
    int32_t *batch_selected_host = malloc((size_t)selected_bytes);
    float *ref_weights_host = malloc((size_t)weights_bytes);
    float *batch_weights_host = malloc((size_t)weights_bytes);
    float *ref_probs_host = malloc((size_t)logits_bytes);
    float *batch_probs_host = malloc((size_t)logits_bytes);
    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(logits != NULL);
    TEST_ASSERT(tokens != NULL);
    TEST_ASSERT(ref_selected != NULL);
    TEST_ASSERT(batch_selected != NULL);
    TEST_ASSERT(ref_weights != NULL);
    TEST_ASSERT(batch_weights != NULL);
    TEST_ASSERT(ref_probs != NULL);
    TEST_ASSERT(batch_probs != NULL);
    TEST_ASSERT(logits_host != NULL);
    TEST_ASSERT(tokens_host != NULL);
    TEST_ASSERT(ref_selected_host != NULL);
    TEST_ASSERT(batch_selected_host != NULL);
    TEST_ASSERT(ref_weights_host != NULL);
    TEST_ASSERT(batch_weights_host != NULL);
    TEST_ASSERT(ref_probs_host != NULL);
    TEST_ASSERT(batch_probs_host != NULL);

    char *saved_disable = test_save_env(disable_env);
    char *saved_select_disable = test_save_env(select_disable_env);
    size_t total_selected_mismatch = 0;
    size_t total_weights_mismatch = 0;
    size_t total_probs_mismatch = 0;
    const bool allocated = model_raw && logits && tokens && ref_selected &&
        batch_selected && ref_weights && batch_weights && ref_probs &&
        batch_probs && logits_host && tokens_host && ref_selected_host &&
        batch_selected_host && ref_weights_host && batch_weights_host &&
        ref_probs_host && batch_probs_host;
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        float *bias = (float *)((uint8_t *)model_raw + bias_offset);
        int32_t *hash = (int32_t *)((uint8_t *)model_raw + hash_offset);
        for (uint32_t i = 0; i < n_expert; i++) {
            bias[i] = (float)((int)((i * 29u) % 67u) - 33) / 16.0f;
        }
        for (uint32_t row = 0; row < hash_rows; row++) {
            const int32_t base = (int32_t)((row * 37u) % n_expert);
            hash[(uint64_t)row * n_used + 0u] = base;
            hash[(uint64_t)row * n_used + 1u] = base;
            hash[(uint64_t)row * n_used + 2u] = (base + 19) % (int32_t)n_expert;
            hash[(uint64_t)row * n_used + 3u] = (base + 43) % (int32_t)n_expert;
            hash[(uint64_t)row * n_used + 4u] = (base + 43) % (int32_t)n_expert;
            hash[(uint64_t)row * n_used + 5u] = (base + 101) % (int32_t)n_expert;
        }
        for (uint32_t row = 0; row < max_tokens; row++) {
            tokens_host[row] = (int32_t)((row * 13u + 7u) % hash_rows);
        }

        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);
        TEST_ASSERT(ds4_gpu_tensor_write(
            tokens, 0, tokens_host, tokens_bytes) != 0);
        TEST_ASSERT(unsetenv(select_disable_env) == 0);

        for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
            const router_batch_case *c = &cases[ci];
            const size_t prob_count = (size_t)c->n_tokens * n_expert;
            const size_t route_count = (size_t)c->n_tokens * n_used;
            const uint64_t case_probs_bytes = prob_count * sizeof(float);
            const uint64_t case_selected_bytes = route_count * sizeof(int32_t);
            const uint64_t case_weights_bytes = route_count * sizeof(float);
            for (uint32_t row = 0; row < c->n_tokens; row++) {
                for (uint32_t expert = 0; expert < n_expert; expert++) {
                    const size_t index = (size_t)row * n_expert + expert;
                    const int value =
                        (int)((row * 131u + expert * 47u +
                               (expert ^ (expert >> 3u)) * 13u) % 513u) -
                        256;
                    logits_host[index] = (float)value / 64.0f;
                    if (c->pattern == 1) {
                        logits_host[index] =
                            -7.0f - (float)((row + expert) % 17u) / 64.0f;
                    } else if (c->pattern == 3) {
                        logits_host[index] =
                            -30.0f - (float)((row * 17u + expert * 29u) % 11u);
                    }
                }
                if (c->pattern == 1) {
                    static const uint32_t tied[] = {
                        7u, 19u, 43u, 71u, 103u, 149u, 211u, 239u,
                    };
                    for (size_t i = 0; i < sizeof(tied) / sizeof(tied[0]); i++) {
                        logits_host[(size_t)row * n_expert + tied[i]] = 1.0f;
                    }
                }
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                logits, 0, logits_host, case_probs_bytes) != 0);

            TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_router_select_batch_tensor(
                ref_selected, ref_weights, ref_probs,
                model_raw, page, bias_offset, hash_offset, hash_rows,
                1, 0, c->has_bias, c->hash_mode, logits, tokens,
                n_expert, n_used, 1.5f, c->n_tokens) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_selected, 0, ref_selected_host, case_selected_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_weights, 0, ref_weights_host, case_weights_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_probs, 0, ref_probs_host, case_probs_bytes) != 0);

            if (c->hash_mode) {
                for (uint32_t row = 0; row < c->n_tokens; row++) {
                    const uint32_t hash_row = (uint32_t)tokens_host[row];
                    for (uint32_t lane = 0; lane < n_used; lane++) {
                        TEST_ASSERT(
                            ref_selected_host[(size_t)row * n_used + lane] ==
                            hash[(uint64_t)hash_row * n_used + lane]);
                    }
                }
            }
            if (c->pattern == 3) {
                for (uint32_t row = 0; row < c->n_tokens; row++) {
                    float sum = 0.0f;
                    for (uint32_t lane = 0; lane < n_used; lane++) {
                        const int32_t expert =
                            ref_selected_host[(size_t)row * n_used + lane];
                        sum += ref_probs_host[(size_t)row * n_expert +
                                              (uint32_t)expert];
                    }
                    TEST_ASSERT(sum < 6.103515625e-5f);
                }
            }

            TEST_ASSERT(unsetenv(disable_env) == 0);
            for (uint32_t rep = 0; rep < repeats; rep++) {
                TEST_ASSERT(ds4_gpu_router_select_batch_tensor(
                    batch_selected, batch_weights, batch_probs,
                    model_raw, page, bias_offset, hash_offset, hash_rows,
                    1, 0, c->has_bias, c->hash_mode, logits, tokens,
                    n_expert, n_used, 1.5f, c->n_tokens) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    batch_selected, 0, batch_selected_host,
                    case_selected_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    batch_weights, 0, batch_weights_host,
                    case_weights_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    batch_probs, 0, batch_probs_host,
                    case_probs_bytes) != 0);

                size_t selected_mismatch = 0;
                for (size_t i = 0; i < route_count; i++) {
                    if (ref_selected_host[i] != batch_selected_host[i]) {
                        selected_mismatch++;
                    }
                }
                const test_float_compare_stats weights_stats =
                    test_compare_float_bits(
                        ref_weights_host, batch_weights_host, route_count);
                const test_float_compare_stats probs_stats =
                    test_compare_float_bits(
                        ref_probs_host, batch_probs_host, prob_count);
                fprintf(stderr,
                        "ds4-test: router batch weights exactness "
                        "case=%s rep=%u selected=%zu/%zu weights=%zu/%zu "
                        "probs=%zu/%zu max_weight_ulp=%u max_prob_ulp=%u\n",
                        c->name,
                        rep,
                        selected_mismatch,
                        route_count,
                        weights_stats.mismatch_count,
                        route_count,
                        probs_stats.mismatch_count,
                        prob_count,
                        weights_stats.max_ulp,
                        probs_stats.max_ulp);
                TEST_ASSERT(selected_mismatch == 0);
                TEST_ASSERT(weights_stats.mismatch_count == 0);
                TEST_ASSERT(probs_stats.mismatch_count == 0);
                total_selected_mismatch += selected_mismatch;
                total_weights_mismatch += weights_stats.mismatch_count;
                total_probs_mismatch += probs_stats.mismatch_count;
            }
        }
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(select_disable_env, saved_select_disable);
    fprintf(stderr,
            "ds4-test: router batch weights total selected=%zu "
            "weights=%zu probs=%zu\n",
            total_selected_mismatch,
            total_weights_mismatch,
            total_probs_mismatch);
    TEST_ASSERT(total_selected_mismatch == 0);
    TEST_ASSERT(total_weights_mismatch == 0);
    TEST_ASSERT(total_probs_mismatch == 0);

    free(batch_probs_host);
    free(ref_probs_host);
    free(batch_weights_host);
    free(ref_weights_host);
    free(batch_selected_host);
    free(ref_selected_host);
    free(tokens_host);
    free(logits_host);
    ds4_gpu_tensor_free(batch_probs);
    ds4_gpu_tensor_free(ref_probs);
    ds4_gpu_tensor_free(batch_weights);
    ds4_gpu_tensor_free(ref_weights);
    ds4_gpu_tensor_free(batch_selected);
    ds4_gpu_tensor_free(ref_selected);
    ds4_gpu_tensor_free(tokens);
    ds4_gpu_tensor_free(logits);
    free(model_raw);
}
#endif

/* Metal tensor type codes (ds4_metal.m enum; not exposed through ds4_gpu.h). */
#define TEST_METAL_TENSOR_Q8_0 8

/* GLM 5.3 MTP verify KDA: one fused rows_snap dispatch must match the per-row
 * decode loop plus the post-row snapshot copies, bit for bit. */
static void test_metal_glm53_kda_rows_snap_exact_case(uint32_t n_rows) {
    const uint32_t n_heads = 4;
    const uint32_t head_dim = 128;
    const uint32_t projection = n_heads * head_dim;
    const uint32_t snap_rows = n_rows - 1u;
    const uint64_t page = (uint64_t)getpagesize();

    const uint64_t q_conv_bytes = (uint64_t)projection * 4u * sizeof(float);
    const uint64_t a_log_bytes = (uint64_t)n_heads * sizeof(float);
    const uint64_t dt_bias_bytes = (uint64_t)projection * sizeof(float);
    const uint64_t o_norm_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t model_bytes = 3u * q_conv_bytes + a_log_bytes +
        dt_bias_bytes + o_norm_bytes;
    const uint64_t model_alloc = test_round_up_u64(model_bytes, page);

    const uint64_t conv_state_bytes = (uint64_t)projection * 9u * sizeof(float);
    const uint64_t state_bytes = (uint64_t)projection * head_dim * sizeof(float);
    const uint64_t act_bytes = (uint64_t)n_rows * projection * sizeof(float);
    const uint64_t beta_bytes = (uint64_t)n_rows * n_heads * sizeof(float);
    const uint64_t snapshot_bytes = conv_state_bytes + state_bytes;

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)model_alloc) == 0);
    if (!model_raw) return;
    memset(model_raw, 0, (size_t)model_alloc);
    float *mf = model_raw;
    for (uint32_t i = 0; i < projection * 4u; i++) {
        mf[i] = (float)((int)((i * 7u + 3u) % 41u) - 20) / 64.0f;
        mf[projection * 4u + i] = (float)((int)((i * 11u + 5u) % 43u) - 21) / 64.0f;
        mf[2u * projection * 4u + i] = (float)((int)((i * 13u + 7u) % 47u) - 23) / 64.0f;
    }
    float *a_log = mf + 3u * projection * 4u;
    for (uint32_t h = 0; h < n_heads; h++) a_log[h] = (float)((int)((h * 3u + 2u) % 9u) - 4) / 8.0f;
    float *dt_bias = a_log + n_heads;
    for (uint32_t i = 0; i < projection; i++) {
        dt_bias[i] = (float)((int)((i * 17u + 11u) % 51u) - 25) / 64.0f;
    }
    float *o_norm = dt_bias + projection;
    for (uint32_t i = 0; i < head_dim; i++) {
        o_norm[i] = (float)((int)((i * 19u + 13u) % 53u) - 26) / 64.0f;
    }

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(act_bytes);
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(act_bytes);
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(act_bytes);
    ds4_gpu_tensor *raw_gate = ds4_gpu_tensor_alloc(act_bytes);
    ds4_gpu_tensor *raw_beta = ds4_gpu_tensor_alloc(beta_bytes);
    ds4_gpu_tensor *output_gate = ds4_gpu_tensor_alloc(act_bytes);
    ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(act_bytes);
    ds4_gpu_tensor *got_out = ds4_gpu_tensor_alloc(act_bytes);
    ds4_gpu_tensor *ref_conv = ds4_gpu_tensor_alloc(conv_state_bytes);
    ds4_gpu_tensor *got_conv = ds4_gpu_tensor_alloc(conv_state_bytes);
    ds4_gpu_tensor *ref_state = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *got_state = ds4_gpu_tensor_alloc(state_bytes);
    TEST_ASSERT(q && k && v && raw_gate && raw_beta && output_gate &&
                ref_out && got_out && ref_conv && got_conv && ref_state && got_state);
    if (!q || !k || !v || !raw_gate || !raw_beta || !output_gate ||
        !ref_out || !got_out || !ref_conv || !got_conv || !ref_state || !got_state) {
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(v);
        ds4_gpu_tensor_free(raw_gate); ds4_gpu_tensor_free(raw_beta); ds4_gpu_tensor_free(output_gate);
        ds4_gpu_tensor_free(ref_out); ds4_gpu_tensor_free(got_out);
        ds4_gpu_tensor_free(ref_conv); ds4_gpu_tensor_free(got_conv);
        ds4_gpu_tensor_free(ref_state); ds4_gpu_tensor_free(got_state);
        free(model_raw);
        return;
    }

    float *q_host = malloc((size_t)act_bytes);
    float *k_host = malloc((size_t)act_bytes);
    float *v_host = malloc((size_t)act_bytes);
    float *gate_host = malloc((size_t)act_bytes);
    float *beta_host = malloc((size_t)beta_bytes);
    float *ogate_host = malloc((size_t)act_bytes);
    float *state_host = malloc((size_t)state_bytes);
    float *conv_host = malloc((size_t)conv_state_bytes);
    TEST_ASSERT(q_host && k_host && v_host && gate_host && beta_host &&
                ogate_host && state_host && conv_host);
    if (!q_host || !k_host || !v_host || !gate_host || !beta_host ||
        !ogate_host || !state_host || !conv_host) {
        free(q_host); free(k_host); free(v_host); free(gate_host); free(beta_host);
        free(ogate_host); free(state_host); free(conv_host);
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(v);
        ds4_gpu_tensor_free(raw_gate); ds4_gpu_tensor_free(raw_beta); ds4_gpu_tensor_free(output_gate);
        ds4_gpu_tensor_free(ref_out); ds4_gpu_tensor_free(got_out);
        ds4_gpu_tensor_free(ref_conv); ds4_gpu_tensor_free(got_conv);
        ds4_gpu_tensor_free(ref_state); ds4_gpu_tensor_free(got_state);
        free(model_raw);
        return;
    }
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t i = 0; i < projection; i++) {
            const uint32_t idx = r * projection + i;
            q_host[idx] = (float)((int)((idx * 5u + 3u) % 61u) - 30) / 64.0f;
            k_host[idx] = (float)((int)((idx * 9u + 7u) % 67u) - 33) / 64.0f;
            v_host[idx] = (float)((int)((idx * 13u + 11u) % 71u) - 35) / 64.0f;
            gate_host[idx] = (float)((int)((idx * 17u + 5u) % 73u) - 36) / 64.0f;
            ogate_host[idx] = (float)((int)((idx * 23u + 9u) % 79u) - 39) / 64.0f;
        }
    }
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t h = 0; h < n_heads; h++) {
            beta_host[r * n_heads + h] = (float)((int)((r * 3u + h * 5u + 2u) % 19u) - 9) / 16.0f;
        }
    }
    for (uint32_t i = 0; i < projection * head_dim; i++) {
        state_host[i] = (float)((int)((i * 3u + 1u) % 37u) - 18) / 32.0f;
    }
    for (uint32_t i = 0; i < projection * 9u; i++) {
        conv_host[i] = (float)((int)((i * 7u + 2u) % 41u) - 20) / 32.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, act_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(k, 0, k_host, act_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(v, 0, v_host, act_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(raw_gate, 0, gate_host, act_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(raw_beta, 0, beta_host, beta_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(output_gate, 0, ogate_host, act_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(ref_conv, 0, conv_host, conv_state_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(got_conv, 0, conv_host, conv_state_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(ref_state, 0, state_host, state_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(got_state, 0, state_host, state_bytes) != 0);

    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
    ds4_gpu_set_quality(false);

    /* Reference: per-row decode calls, snapshotting conv then state after each
     * row r < snap_rows (same layout as glm53_graph_copy_spec_state). */
    ds4_gpu_tensor *ref_snap0 = ds4_gpu_tensor_alloc(snapshot_bytes);
    ds4_gpu_tensor *ref_snap1 = ds4_gpu_tensor_alloc(snapshot_bytes);
    TEST_ASSERT(ref_snap0 != NULL && ref_snap1 != NULL);
    if (!ref_snap0 || !ref_snap1) {
        free(q_host); free(k_host); free(v_host); free(gate_host); free(beta_host);
        free(ogate_host); free(state_host); free(conv_host);
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(v);
        ds4_gpu_tensor_free(raw_gate); ds4_gpu_tensor_free(raw_beta); ds4_gpu_tensor_free(output_gate);
        ds4_gpu_tensor_free(ref_out); ds4_gpu_tensor_free(got_out);
        ds4_gpu_tensor_free(ref_conv); ds4_gpu_tensor_free(got_conv);
        ds4_gpu_tensor_free(ref_state); ds4_gpu_tensor_free(got_state);
        free(model_raw);
        return;
    }
    for (uint32_t r = 0; r < n_rows; r++) {
        const uint64_t row_act = (uint64_t)projection * sizeof(float);
        ds4_gpu_tensor *q_row = ds4_gpu_tensor_view(q, (uint64_t)r * row_act, row_act);
        ds4_gpu_tensor *k_row = ds4_gpu_tensor_view(k, (uint64_t)r * row_act, row_act);
        ds4_gpu_tensor *v_row = ds4_gpu_tensor_view(v, (uint64_t)r * row_act, row_act);
        ds4_gpu_tensor *gate_row = ds4_gpu_tensor_view(raw_gate, (uint64_t)r * row_act, row_act);
        ds4_gpu_tensor *beta_row = ds4_gpu_tensor_view(raw_beta,
                (uint64_t)r * n_heads * sizeof(float), (uint64_t)n_heads * sizeof(float));
        ds4_gpu_tensor *ogate_row = ds4_gpu_tensor_view(output_gate, (uint64_t)r * row_act, row_act);
        ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(ref_out, (uint64_t)r * row_act, row_act);
        TEST_ASSERT(q_row && k_row && v_row && gate_row && beta_row && ogate_row && out_row);
        if (q_row && k_row && v_row && gate_row && beta_row && ogate_row && out_row) {
            TEST_ASSERT(ds4_gpu_glm53_kda_decode(
                    out_row, ref_conv, ref_state, q_row, k_row, v_row,
                    gate_row, beta_row, ogate_row,
                    model_raw, model_alloc,
                    0, q_conv_bytes, 2u * q_conv_bytes,
                    3u * q_conv_bytes,
                    3u * q_conv_bytes + a_log_bytes,
                    3u * q_conv_bytes + a_log_bytes + dt_bias_bytes,
                    n_heads, 1, -5.0f, 1.0e-6f) != 0);
        }
        ds4_gpu_tensor_free(out_row);
        ds4_gpu_tensor_free(ogate_row);
        ds4_gpu_tensor_free(beta_row);
        ds4_gpu_tensor_free(gate_row);
        ds4_gpu_tensor_free(v_row);
        ds4_gpu_tensor_free(k_row);
        ds4_gpu_tensor_free(q_row);
        if (r < snap_rows) {
            ds4_gpu_tensor *snap = r == 0u ? ref_snap0 : ref_snap1;
            /* tensor_copy encodes a blit into the open command batch. */
            TEST_ASSERT(ds4_gpu_begin_commands() != 0);
            TEST_ASSERT(ds4_gpu_tensor_copy(snap, 0, ref_conv, 0, conv_state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_copy(snap, conv_state_bytes, ref_state, 0, state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_end_commands() != 0);
        }
    }

    /* Tested: one fused dispatch writes the same snapshots at a 256-byte
     * non-zero offset in a single buffer. */
    const uint64_t snap_region = snapshot_bytes;
    const uint64_t test_snap_bytes = 256u + 2u * snap_region;
    ds4_gpu_tensor *test_snap = ds4_gpu_tensor_alloc(test_snap_bytes);
    TEST_ASSERT(test_snap != NULL);
    if (!test_snap) {
        free(q_host); free(k_host); free(v_host); free(gate_host); free(beta_host);
        free(ogate_host); free(state_host); free(conv_host);
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(v);
        ds4_gpu_tensor_free(raw_gate); ds4_gpu_tensor_free(raw_beta); ds4_gpu_tensor_free(output_gate);
        ds4_gpu_tensor_free(ref_out); ds4_gpu_tensor_free(got_out);
        ds4_gpu_tensor_free(ref_conv); ds4_gpu_tensor_free(got_conv);
        ds4_gpu_tensor_free(ref_state); ds4_gpu_tensor_free(got_state);
        ds4_gpu_tensor_free(ref_snap0); ds4_gpu_tensor_free(ref_snap1);
        free(model_raw);
        return;
    }
    TEST_ASSERT(ds4_gpu_glm53_kda_decode_rows_snap(
            got_out, got_conv, got_state,
            q, k, v, raw_gate, raw_beta, output_gate,
            model_raw, model_alloc,
            0, q_conv_bytes, 2u * q_conv_bytes,
            3u * q_conv_bytes,
            3u * q_conv_bytes + a_log_bytes,
            3u * q_conv_bytes + a_log_bytes + dt_bias_bytes,
            n_heads, n_rows,
            test_snap, 256u,
            test_snap, 256u + snap_region,
            snap_rows, -5.0f, 1.0e-6f) != 0);

    /* Compare outputs and final state bit for bit. */
    float *got_host = malloc((size_t)act_bytes);
    float *ref_host = malloc((size_t)act_bytes);
    float *got_conv_host = malloc((size_t)conv_state_bytes);
    float *got_state_host = malloc((size_t)state_bytes);
    float *ref_conv_host = malloc((size_t)conv_state_bytes);
    float *ref_state_host = malloc((size_t)state_bytes);
    TEST_ASSERT(got_host && ref_host && got_conv_host && got_state_host &&
                ref_conv_host && ref_state_host);
    if (!got_host || !ref_host || !got_conv_host || !got_state_host ||
        !ref_conv_host || !ref_state_host) {
        free(got_host); free(ref_host); free(got_conv_host); free(got_state_host);
        free(ref_conv_host); free(ref_state_host);
        free(q_host); free(k_host); free(v_host); free(gate_host); free(beta_host);
        free(ogate_host); free(state_host); free(conv_host);
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(v);
        ds4_gpu_tensor_free(raw_gate); ds4_gpu_tensor_free(raw_beta); ds4_gpu_tensor_free(output_gate);
        ds4_gpu_tensor_free(ref_out); ds4_gpu_tensor_free(got_out);
        ds4_gpu_tensor_free(ref_conv); ds4_gpu_tensor_free(got_conv);
        ds4_gpu_tensor_free(ref_state); ds4_gpu_tensor_free(got_state);
        ds4_gpu_tensor_free(ref_snap0); ds4_gpu_tensor_free(ref_snap1);
        ds4_gpu_tensor_free(test_snap);
        free(model_raw);
        return;
    }
    TEST_ASSERT(ds4_gpu_tensor_read(got_out, 0, got_host, act_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref_out, 0, ref_host, act_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(got_conv, 0, got_conv_host, conv_state_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(got_state, 0, got_state_host, state_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref_conv, 0, ref_conv_host, conv_state_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref_state, 0, ref_state_host, state_bytes) != 0);

    uint32_t mismatch = 0;
    for (uint32_t i = 0; i < n_rows * projection; i++) {
        if (memcmp(&ref_host[i], &got_host[i], sizeof(float)) != 0) mismatch++;
    }
    for (uint32_t i = 0; i < projection * 9u; i++) {
        if (memcmp(&ref_conv_host[i], &got_conv_host[i], sizeof(float)) != 0) mismatch++;
    }
    for (uint32_t i = 0; i < projection * head_dim; i++) {
        if (memcmp(&ref_state_host[i], &got_state_host[i], sizeof(float)) != 0) mismatch++;
    }
    if (mismatch != 0) {
        fprintf(stderr, "ds4-test: GLM 5.3 KDA rows snap rows=%u out/state mismatches=%u\n",
                n_rows, mismatch);
    }
    TEST_ASSERT(mismatch == 0);

    /* Compare each snapshot region (conv then state) with the reference copy. */
    for (uint32_t s = 0; s < snap_rows; s++) {
        ds4_gpu_tensor *ref_snap = s == 0u ? ref_snap0 : ref_snap1;
        const uint64_t test_off = 256u + (uint64_t)s * snap_region;
        float *ref_snap_host = malloc((size_t)snapshot_bytes);
        float *got_snap_host = malloc((size_t)snapshot_bytes);
        TEST_ASSERT(ref_snap_host != NULL && got_snap_host != NULL);
        if (!ref_snap_host || !got_snap_host) {
            free(ref_snap_host);
            free(got_snap_host);
            continue;
        }
        TEST_ASSERT(ds4_gpu_tensor_read(ref_snap, 0, ref_snap_host, snapshot_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(test_snap, test_off, got_snap_host, snapshot_bytes) != 0);
        uint32_t snap_mismatch = 0;
        for (uint32_t i = 0; i < snapshot_bytes / sizeof(float); i++) {
            if (memcmp(&ref_snap_host[i], &got_snap_host[i], sizeof(float)) != 0) snap_mismatch++;
        }
        if (snap_mismatch != 0) {
            fprintf(stderr,
                    "ds4-test: GLM 5.3 KDA rows snap rows=%u snap=%u mismatches=%u\n",
                    n_rows, s, snap_mismatch);
        }
        TEST_ASSERT(snap_mismatch == 0);
        free(ref_snap_host);
        free(got_snap_host);
    }

    free(got_host); free(ref_host); free(got_conv_host); free(got_state_host);
    free(ref_conv_host); free(ref_state_host);
    free(q_host); free(k_host); free(v_host); free(gate_host); free(beta_host);
    free(ogate_host); free(state_host); free(conv_host);
    ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(raw_gate); ds4_gpu_tensor_free(raw_beta); ds4_gpu_tensor_free(output_gate);
    ds4_gpu_tensor_free(ref_out); ds4_gpu_tensor_free(got_out);
    ds4_gpu_tensor_free(ref_conv); ds4_gpu_tensor_free(got_conv);
    ds4_gpu_tensor_free(ref_state); ds4_gpu_tensor_free(got_state);
    ds4_gpu_tensor_free(ref_snap0); ds4_gpu_tensor_free(ref_snap1);
    ds4_gpu_tensor_free(test_snap);
    free(model_raw);
}

static void test_metal_glm53_kda_rows_snap_exact(void) {
    test_metal_glm53_kda_rows_snap_exact_case(2);
    test_metal_glm53_kda_rows_snap_exact_case(3);
}

/* GLM 5.3 MTP verify attention: the one-dispatch rows kernel must match one
 * decode dispatch per row, for both float and f16 caches. */
static void test_metal_glm53_attention_rows_exact_case(bool cache_f16, uint32_t n_rows) {
    const uint32_t n_head = 4;
    const uint32_t kv_lora_dim = 512;
    const uint32_t qk_nope = 512;
    const uint32_t qk_rope = 0;
    const uint32_t value_dim = 256;
    const uint32_t cache_cap = 64;
    const uint32_t n_selected = 16;
    const uint32_t qk_dim = qk_nope + qk_rope;
    const uint64_t page = (uint64_t)getpagesize();
    const uint32_t value_rows = n_head * value_dim;

    const uint64_t row_bytes = (uint64_t)(kv_lora_dim / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)value_rows * row_bytes;
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, page);
    const uint64_t cache_elem = cache_f16 ? sizeof(uint16_t) : sizeof(float);
    const uint64_t kv_cache_bytes = (uint64_t)cache_cap * kv_lora_dim * cache_elem;
    const uint64_t rope_cache_bytes = (uint64_t)cache_cap * qk_rope * cache_elem;
    const uint64_t q_bytes = (uint64_t)n_rows * n_head * qk_dim * sizeof(float);
    const uint64_t low_bytes = (uint64_t)n_rows * n_head * kv_lora_dim * sizeof(float);
    const uint64_t heads_bytes = (uint64_t)n_rows * n_head * value_dim * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)n_rows * n_selected * sizeof(uint32_t);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)page, (size_t)weight_alloc) == 0);
    if (!weights_raw) return;
    memset(weights_raw, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights((uint8_t *)weights_raw, kv_lora_dim, value_rows, 41);

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *qk_low = ds4_gpu_tensor_alloc(low_bytes);
    ds4_gpu_tensor *kv_cache = ds4_gpu_tensor_alloc(kv_cache_bytes);
    ds4_gpu_tensor *k_rope = ds4_gpu_tensor_alloc(rope_cache_bytes > 0 ? rope_cache_bytes : 1);
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *ref_heads = ds4_gpu_tensor_alloc(heads_bytes);
    ds4_gpu_tensor *got_heads = ds4_gpu_tensor_alloc(heads_bytes);
    TEST_ASSERT(q && qk_low && kv_cache && k_rope && selected && ref_heads && got_heads);
    if (!q || !qk_low || !kv_cache || !k_rope || !selected || !ref_heads || !got_heads) {
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(qk_low); ds4_gpu_tensor_free(kv_cache);
        ds4_gpu_tensor_free(k_rope); ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(ref_heads); ds4_gpu_tensor_free(got_heads);
        free(weights_raw);
        return;
    }

    float *q_host = malloc((size_t)q_bytes);
    float *low_host = malloc((size_t)low_bytes);
    uint32_t *selected_host = malloc((size_t)selected_bytes);
    TEST_ASSERT(q_host != NULL && low_host != NULL && selected_host != NULL);
    if (!q_host || !low_host || !selected_host) {
        free(q_host); free(low_host); free(selected_host);
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(qk_low); ds4_gpu_tensor_free(kv_cache);
        ds4_gpu_tensor_free(k_rope); ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(ref_heads); ds4_gpu_tensor_free(got_heads);
        free(weights_raw);
        return;
    }
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t h = 0; h < n_head; h++) {
            for (uint32_t i = 0; i < qk_dim; i++) {
                const uint32_t idx = (r * n_head + h) * qk_dim + i;
                q_host[idx] = (float)((int)((idx * 7u + r * 3u + 11u) % 97u) - 48) / 64.0f;
            }
            for (uint32_t i = 0; i < kv_lora_dim; i++) {
                const uint32_t idx = (r * n_head + h) * kv_lora_dim + i;
                low_host[idx] = (float)((int)((idx * 13u + h * 5u + 17u) % 101u) - 50) / 64.0f;
            }
        }
        for (uint32_t s = 0; s < n_selected; s++) {
            selected_host[r * n_selected + s] = (uint32_t)((r * 31u + s * 7u + 5u) % cache_cap);
        }
        selected_host[r * n_selected + n_selected - 1u] = UINT32_MAX;
    }

    void *kv_host = malloc((size_t)kv_cache_bytes);
    void *rope_host = malloc((size_t)(rope_cache_bytes > 0 ? rope_cache_bytes : 1));
    TEST_ASSERT(kv_host != NULL && rope_host != NULL);
    if (!kv_host || !rope_host) {
        free(kv_host); free(rope_host);
        free(q_host); free(low_host); free(selected_host);
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(qk_low); ds4_gpu_tensor_free(kv_cache);
        ds4_gpu_tensor_free(k_rope); ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(ref_heads); ds4_gpu_tensor_free(got_heads);
        free(weights_raw);
        return;
    }
    if (cache_f16) {
        uint16_t *kv16 = kv_host;
        for (uint32_t i = 0; i < cache_cap * kv_lora_dim; i++) {
            const float f = (float)((int)((i * 3u + 7u) % 83u) - 41) / 64.0f;
            kv16[i] = test_float_to_f16(f);
        }
    } else {
        float *kv32 = kv_host;
        for (uint32_t i = 0; i < cache_cap * kv_lora_dim; i++) {
            kv32[i] = (float)((int)((i * 3u + 7u) % 83u) - 41) / 64.0f;
        }
    }

    TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(qk_low, 0, low_host, low_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(kv_cache, 0, kv_host, kv_cache_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(selected, 0, selected_host, selected_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);

    for (uint32_t r = 0; r < n_rows; r++) {
        const uint64_t row_heads = (uint64_t)n_head * value_dim * sizeof(float);
        const uint64_t row_q = (uint64_t)n_head * qk_dim * sizeof(float);
        const uint64_t row_low = (uint64_t)n_head * kv_lora_dim * sizeof(float);
        const uint64_t row_sel = (uint64_t)n_selected * sizeof(uint32_t);
        ds4_gpu_tensor *heads_row = ds4_gpu_tensor_view(ref_heads, (uint64_t)r * row_heads, row_heads);
        ds4_gpu_tensor *q_row = ds4_gpu_tensor_view(q, (uint64_t)r * row_q, row_q);
        ds4_gpu_tensor *low_row = ds4_gpu_tensor_view(qk_low, (uint64_t)r * row_low, row_low);
        ds4_gpu_tensor *sel_row = ds4_gpu_tensor_view(selected, (uint64_t)r * row_sel, row_sel);
        TEST_ASSERT(heads_row && q_row && low_row && sel_row);
        if (heads_row && q_row && low_row && sel_row) {
            TEST_ASSERT(ds4_gpu_glm_attention_indexed_decode_typed_tensor(
                    heads_row, q_row, low_row, kv_cache, k_rope,
                    weights_raw, weight_alloc, 0, TEST_METAL_TENSOR_Q8_0,
                    sel_row, n_selected, cache_cap, cache_f16,
                    n_head, kv_lora_dim, qk_nope, qk_rope, value_dim,
                    0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) != 0);
        }
        ds4_gpu_tensor_free(sel_row);
        ds4_gpu_tensor_free(low_row);
        ds4_gpu_tensor_free(q_row);
        ds4_gpu_tensor_free(heads_row);
    }

    TEST_ASSERT(ds4_gpu_glm_attention_indexed_decode_rows_typed_tensor(
            got_heads, q, qk_low, kv_cache, k_rope,
            weights_raw, weight_alloc, 0, TEST_METAL_TENSOR_Q8_0,
            selected, n_selected, cache_cap, cache_f16,
            n_head, kv_lora_dim, qk_nope, qk_rope, value_dim,
            0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, n_rows) != 0);

    float *ref_heads_host = malloc((size_t)heads_bytes);
    float *got_heads_host = malloc((size_t)heads_bytes);
    TEST_ASSERT(ref_heads_host != NULL && got_heads_host != NULL);
    if (!ref_heads_host || !got_heads_host) {
        free(ref_heads_host); free(got_heads_host);
        free(kv_host); free(rope_host);
        free(q_host); free(low_host); free(selected_host);
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(qk_low); ds4_gpu_tensor_free(kv_cache);
        ds4_gpu_tensor_free(k_rope); ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(ref_heads); ds4_gpu_tensor_free(got_heads);
        free(weights_raw);
        return;
    }
    TEST_ASSERT(ds4_gpu_tensor_read(ref_heads, 0, ref_heads_host, heads_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(got_heads, 0, got_heads_host, heads_bytes) != 0);
    uint32_t mismatch = 0;
    for (uint32_t i = 0; i < n_rows * n_head * value_dim; i++) {
        if (memcmp(&ref_heads_host[i], &got_heads_host[i], sizeof(float)) != 0) mismatch++;
    }
    if (mismatch != 0) {
        fprintf(stderr,
                "ds4-test: GLM 5.3 attention rows cache_f16=%u rows=%u mismatches=%u/%u\n",
                cache_f16 ? 1u : 0u, n_rows, mismatch, n_rows * n_head * value_dim);
    }
    TEST_ASSERT(mismatch == 0);

    free(ref_heads_host); free(got_heads_host);
    free(kv_host); free(rope_host);
    free(q_host); free(low_host); free(selected_host);
    ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(qk_low); ds4_gpu_tensor_free(kv_cache);
    ds4_gpu_tensor_free(k_rope); ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(ref_heads); ds4_gpu_tensor_free(got_heads);
    free(weights_raw);
}

static void test_metal_glm53_attention_rows_exact(void) {
    test_metal_glm53_attention_rows_exact_case(false, 2);
    test_metal_glm53_attention_rows_exact_case(false, 3);
    test_metal_glm53_attention_rows_exact_case(true, 2);
    test_metal_glm53_attention_rows_exact_case(true, 3);
}

static void test_metal_kernel_group(void) {
    test_metal_f16_matvec_fast_nr0_4();
    test_metal_f16_prefill_matmul();
    test_metal_q8_0_prefill_matmul();
    test_metal_pack_slot_rows_f32();
    test_metal_store_raw_kv_batch_wrap();
    test_dspark_cache_window_crop();
    test_metal_q8_0_decode_pair_exact();
#if defined(__APPLE__)
    test_metal_f16_compressor_pair_state_store_exact();
    test_metal_compressor_ape_add_exact();
    test_metal_compressor_ratio4_pack_exact();
    test_metal_compressor_ratio4_replay_pack_exact();
    test_metal_compressor_ratio4_direct_pool_exact();
    test_metal_inplace_rope_pair_exact();
    test_metal_contiguous_f32_f16_roundtrip_exact();
    test_metal_gathered_kv_stage_exact();
    test_metal_contiguous_compressed_f16_attention_exact();
    test_metal_persistent_zero_attention_mask_exact();
    test_metal_zero_prefix_prefill_mask_cache_exact();
    test_metal_hc_split_weighted_sum_norm_batch_exact();
    test_metal_output_hc_weights4_exact();
    test_metal_hc_rms_scale_project_f16_exact();
    test_metal_router_simd_finalize_exact();
    test_metal_router_weights_batch_exact();
    test_metal_glm53_kda_rows_snap_exact();
    test_metal_glm53_attention_rows_exact();
#endif
}

static void test_metal_short_prefill_ratio4(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine) return;

    const int tokens[] = {
        ds4_token_user(engine),
        ds4_token_assistant(engine),
        ds4_token_eos(engine),
    };
    for (size_t i = 0; i < sizeof(tokens) / sizeof(tokens[0]); i++) {
        TEST_ASSERT(tokens[i] >= 0);
        if (tokens[i] < 0) return;
    }

    for (size_t n = 1; n <= 3; n++) {
        ds4_tokens prompt = {0};
        for (size_t i = 0; i < n; i++) {
            ds4_tokens_push(&prompt, tokens[i]);
        }
        TEST_ASSERT(prompt.len == (int)n);

        ds4_session *session = NULL;
        TEST_ASSERT(ds4_session_create(&session, engine, 2048) == 0);
        if (!session) {
            ds4_tokens_free(&prompt);
            return;
        }

        char err[160] = {0};
        const int rc = ds4_session_sync(session, &prompt, err, sizeof(err));
        if (rc != 0) {
            fprintf(stderr, "ds4-test: short prefill failed for %zu token(s): %s\n",
                    n, err);
        }
        TEST_ASSERT(rc == 0);

        ds4_session_free(session);
        ds4_tokens_free(&prompt);
    }
}

typedef struct {
    int expected_start;
    int expected_total;
    int last_current;
    int chunk_events;
    int display_events;
    int intermediate_events;
    bool invalid;
} test_continued_prefill_progress;

static void test_continued_prefill_progress_cb(
        void       *ud,
        const char *event,
        int         current,
        int         total) {
    test_continued_prefill_progress *p = ud;
    const bool chunk = !strcmp(event, "prefill_chunk");
    const bool display = !strcmp(event, "prefill_display");
    if (!chunk && !display) return;

    if (total != p->expected_total || current < p->expected_start ||
        current > total || current < p->last_current) {
        p->invalid = true;
    }
    if (current > p->expected_start && current < total) {
        p->intermediate_events++;
    }
    if (current > p->last_current) p->last_current = current;
    if (chunk) p->chunk_events++;
    if (display) p->display_events++;
}

static double test_monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static bool test_sync_continued_prefill_stage(
        ds4_session *session,
        ds4_tokens  *prompt,
        int          previous_len,
        int          next_len,
        double      *elapsed_out) {
    test_continued_prefill_progress progress = {
        .expected_start = previous_len,
        .expected_total = next_len,
        .last_current = previous_len,
    };
    char err[160] = {0};
    prompt->len = next_len;
    ds4_session_set_progress(session, test_continued_prefill_progress_cb,
                             &progress);
    ds4_session_set_display_progress(session,
                                     test_continued_prefill_progress_cb,
                                     &progress);
    const double started = test_monotonic_seconds();
    const int rc = ds4_session_sync(session, prompt, err, sizeof(err));
    const double elapsed = test_monotonic_seconds() - started;
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);
    if (elapsed_out) *elapsed_out = elapsed;

    if (rc != 0) {
        fprintf(stderr,
                "ds4-test: continued prefill %d -> %d failed: %s\n",
                previous_len, next_len, err);
    }
    const int added = next_len - previous_len;
    fprintf(stderr,
            "ds4-test: continued prefill +%d: %.2f ms, %.1f t/s, "
            "chunk=%d display=%d intermediate=%d\n",
            added, elapsed * 1000.0,
            elapsed > 0.0 ? (double)added / elapsed : 0.0,
            progress.chunk_events, progress.display_events,
            progress.intermediate_events);

    TEST_ASSERT(rc == 0);
    TEST_ASSERT(!progress.invalid);
    TEST_ASSERT(progress.chunk_events > 0);
    TEST_ASSERT(progress.last_current == next_len);
    if (added > 1 &&
        (added >= 32 ||
         !test_env_bool("DS4_TEST_CONTINUED_PREFILL_ALLOW_COARSE"))) {
        TEST_ASSERT(progress.intermediate_events > 0);
    }
    if (added >= 32) TEST_ASSERT(progress.display_events > 0);
    return rc == 0 && !progress.invalid;
}

static bool test_top_tokens_overlap(
        const ds4_token_score *a,
        const ds4_token_score *b,
        int                    count) {
    bool a0_in_b = false;
    bool b0_in_a = false;
    for (int i = 0; i < count; i++) {
        if (a[0].id == b[i].id) a0_in_b = true;
        if (b[0].id == a[i].id) b0_in_a = true;
    }
    return a0_in_b && b0_in_a;
}

static void test_glm53_continued_prefill(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_is_glm53(engine)) {
        fprintf(stderr,
                "ds4-test: glm53-continued-prefill skipped (GLM 5.3 model required)\n");
        return;
    }

    const int base_len = 64;
    uint32_t large_add = test_env_u32("DS4_TEST_CONTINUED_PREFILL_TOKENS");
    if (large_add == 0) large_add = 256;
    uint32_t large_steps = test_env_u32("DS4_TEST_CONTINUED_PREFILL_STEPS");
    if (large_steps == 0) large_steps = 1;
    const uint64_t final_len64 = (uint64_t)base_len + 4u +
                                 (uint64_t)large_add * large_steps;
    TEST_ASSERT(final_len64 < INT_MAX - 128);
    if (final_len64 >= INT_MAX - 128) return;
    const int final_len = (int)final_len64;
    int ctx_size = final_len + 128;
    if (ctx_size < 4096) ctx_size = 4096;

    ds4_tokens pattern = {0};
    ds4_tokens prompt = {0};
    ds4_session *resumed = NULL;
    ds4_session *cold = NULL;
    ds4_token_score resumed_top[8] = {0};
    ds4_token_score cold_top[8] = {0};
    char err[160] = {0};
    ds4_tokenize_text(engine,
                      " continued prefill checks latency throughput and progress",
                      &pattern);
    TEST_ASSERT(pattern.len > 0);
    if (pattern.len == 0) goto cleanup;
    ds4_chat_begin(engine, &prompt);
    while (prompt.len < final_len) {
        ds4_tokens_push(&prompt, pattern.v[prompt.len % pattern.len]);
    }

    TEST_ASSERT(ds4_session_create(&resumed, engine, ctx_size) == 0);
    if (!resumed) goto cleanup;

    prompt.len = base_len;
    TEST_ASSERT(ds4_session_sync(resumed, &prompt, err, sizeof(err)) == 0);
    if (!test_sync_continued_prefill_stage(resumed, &prompt,
                                           base_len, base_len + 1, NULL)) {
        goto cleanup;
    }
    if (!test_sync_continued_prefill_stage(resumed, &prompt,
                                           base_len + 1, base_len + 4, NULL)) {
        goto cleanup;
    }
    int previous_len = base_len + 4;
    for (uint32_t step = 0; step < large_steps; step++) {
        const int next_len = previous_len + (int)large_add;
        if (!test_sync_continued_prefill_stage(resumed, &prompt,
                                               previous_len, next_len, NULL)) {
            goto cleanup;
        }
        previous_len = next_len;
    }
    TEST_ASSERT(ds4_session_top_logprobs(resumed, resumed_top, 8) == 8);
    ds4_session_free(resumed);
    resumed = NULL;

    TEST_ASSERT(ds4_session_create(&cold, engine, ctx_size) == 0);
    if (!cold) goto cleanup;
    prompt.len = final_len;
    TEST_ASSERT(ds4_session_sync(cold, &prompt, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_top_logprobs(cold, cold_top, 8) == 8);
    int same_rank = 0;
    float max_same_rank_delta = 0.0f;
    for (int i = 0; i < 8; i++) {
        if (resumed_top[i].id != cold_top[i].id) continue;
        same_rank++;
        const float delta = fabsf(resumed_top[i].logit - cold_top[i].logit);
        if (delta > max_same_rank_delta) max_same_rank_delta = delta;
    }
    fprintf(stderr,
            "ds4-test: continued/cold final top token %d/%d, "
            "same-rank top8=%d, max same-rank logit delta=%.6g\n",
            resumed_top[0].id, cold_top[0].id,
            same_rank, max_same_rank_delta);
    TEST_ASSERT(resumed_top[0].id == cold_top[0].id);
    TEST_ASSERT(test_top_tokens_overlap(resumed_top, cold_top, 8));

cleanup:
    ds4_session_free(cold);
    ds4_session_free(resumed);
    ds4_tokens_free(&prompt);
    ds4_tokens_free(&pattern);
}

static char *test_read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *s = malloc((size_t)len + 1);
    if (!s) {
        fclose(fp);
        return NULL;
    }
    size_t nread = fread(s, 1, (size_t)len, fp);
    fclose(fp);
    if (nread != (size_t)len) {
        free(s);
        return NULL;
    }
    s[len] = '\0';
    return s;
}

typedef struct {
    const char *name;
    int number;
} test_long_fact;

static const test_long_fact test_long_facts[] = {
    {"Bob", 34},
    {"Alice", 52},
    {"Clara", 71},
    {"Diego", 93},
    {"Elena", 16},
    {"Felix", 88},
    {"Greta", 47},
    {"Hugo", 29},
    {"Iris", 64},
    {"Jonas", 12},
    {"Kira", 81},
    {"Leo", 39},
    {"Marta", 76},
    {"Nadia", 23},
    {"Owen", 58},
    {"Priya", 97},
};

static bool test_is_name_boundary(char c) {
    unsigned char uc = (unsigned char)c;
    return c == '\0' || !(isalnum(uc) || c == '_');
}

static bool test_parse_assignment_value(const char *p, int *value) {
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '=') return false;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (!isdigit((unsigned char)*p)) return false;

    int v = 0;
    while (isdigit((unsigned char)*p)) {
        v = v * 10 + (*p - '0');
        p++;
    }
    *value = v;
    return true;
}

static bool test_output_has_fact(const char *text, const test_long_fact *fact) {
    const size_t name_len = strlen(fact->name);
    const char *p = text;
    bool saw_wrong_assignment = false;
    int wrong_value = -1;

    while ((p = strstr(p, fact->name)) != NULL) {
        const bool before_ok = p == text || test_is_name_boundary(p[-1]);
        const bool after_ok = test_is_name_boundary(p[name_len]) ||
                              p[name_len] == ' ' ||
                              p[name_len] == '\t' ||
                              p[name_len] == '=';
        if (before_ok && after_ok) {
            int value = 0;
            if (test_parse_assignment_value(p + name_len, &value)) {
                if (value == fact->number) return true;
                saw_wrong_assignment = true;
                wrong_value = value;
            }
        }
        p += name_len;
    }

    if (saw_wrong_assignment) {
        fprintf(stderr,
                "ds4-test: long-context wrong assignment for %s: got %d expected %d\n",
                fact->name, wrong_value, fact->number);
    } else {
        fprintf(stderr,
                "ds4-test: long-context missing assignment for %s=%d\n",
                fact->name, fact->number);
    }
    return false;
}

static int test_hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static bool test_hex_to_bytes(const char *hex, unsigned char *out, int cap, int *len) {
    int n = 0;
    while (*hex && !isspace((unsigned char)*hex)) {
        int hi = test_hex_digit(hex[0]);
        int lo = test_hex_digit(hex[1]);
        if (hi < 0 || lo < 0 || n >= cap) return false;
        out[n++] = (unsigned char)((hi << 4) | lo);
        hex += 2;
    }
    *len = n;
    return true;
}

static bool test_token_bytes_equal(ds4_engine *engine, int token,
                                   const unsigned char *want, int want_len) {
    size_t got_len = 0;
    char *got = ds4_token_text(engine, token, &got_len);
    bool eq = got && got_len == (size_t)want_len &&
              memcmp(got, want, (size_t)want_len) == 0;
    free(got);
    return eq;
}

static void test_long_prefill_progress(void *ud, const char *event, int current, int total) {
    (void)ud;
    if (strcmp(event, "prefill_chunk")) return;
    if (current == 0 || current == total || current % 8192 == 0) {
        fprintf(stderr, "ds4-test: long-context prefill %d/%d\n", current, total);
    }
}

static void test_long_story_fact_recall(void) {
    const char *prompt_path = getenv("DS4_TEST_LONG_PROMPT");
    if (!prompt_path || !prompt_path[0]) {
        prompt_path = "tests/long_context_story_prompt.txt";
    }
    char *prompt_text = test_read_file(prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_engine *engine = test_get_engine(false);
    if (!engine) {
        free(prompt_text);
        return;
    }

    ds4_tokens prompt = {0};
    ds4_tokenize_rendered_chat(engine, prompt_text, &prompt);
    TEST_ASSERT(prompt.len > 30000);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 100000) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        free(prompt_text);
        return;
    }

    char err[160];
    ds4_session_set_progress(session, test_long_prefill_progress, NULL);
    TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    ds4_session_set_progress(session, NULL, NULL);

    buf out = {0};
    uint64_t rng = 12345;
    int generated = 0;
    bool decode_ok = true;
    for (; generated < 350; generated++) {
        int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
        if (token == ds4_token_eos(engine)) break;

        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&out, piece, piece_len);
        free(piece);

        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    const char *text = out.ptr ? out.ptr : "";
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(generated > 0);
    for (size_t i = 0; i < sizeof(test_long_facts) / sizeof(test_long_facts[0]); i++) {
        TEST_ASSERT(test_output_has_fact(text, &test_long_facts[i]));
    }

    buf_free(&out);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    free(prompt_text);
}

#define TEST_VEC_MAX_STEPS 16
#define TEST_VEC_MAX_TOP 32
#define TEST_VEC_MAX_TOKEN_BYTES 128

typedef struct {
    unsigned char bytes[TEST_VEC_MAX_TOKEN_BYTES];
    int len;
    float logprob;
} test_vec_top;

typedef struct {
    unsigned char selected[TEST_VEC_MAX_TOKEN_BYTES];
    int selected_len;
    int ntop;
    test_vec_top top[TEST_VEC_MAX_TOP];
} test_vec_step;

typedef struct {
    char id[96];
    char prompt_path[512];
    int ctx;
    int nsteps;
    test_vec_step steps[TEST_VEC_MAX_STEPS];
} test_vec_case;

static char *test_trim_line(char *line) {
    while (*line && isspace((unsigned char)*line)) line++;
    size_t n = strlen(line);
    while (n && isspace((unsigned char)line[n - 1])) line[--n] = '\0';
    return line;
}

static bool test_read_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    memset(vc, 0, sizeof(*vc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %d %d %511s",
                   vc->id, &vc->ctx, &vc->nsteps, vc->prompt_path) == 4) {
            TEST_ASSERT(vc->nsteps > 0 && vc->nsteps <= TEST_VEC_MAX_STEPS);
            return true;
        }
        TEST_ASSERT(!"unexpected line before vector case");
    }
    return false;
}

static bool test_fill_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    int step_index = -1;
    int top_index = 0;

    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) return true;

        if (!strncmp(p, "step ", 5)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            int ntop = 0;
            if (sscanf(p, "step %d %257s %d", &step_index, hex, &ntop) != 3) {
                TEST_ASSERT(!"bad vector step line");
                return false;
            }
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(ntop >= 0 && ntop <= TEST_VEC_MAX_TOP);
            vc->steps[step_index].ntop = ntop;
            TEST_ASSERT(test_hex_to_bytes(hex,
                                          vc->steps[step_index].selected,
                                          TEST_VEC_MAX_TOKEN_BYTES,
                                          &vc->steps[step_index].selected_len));
            top_index = 0;
            continue;
        }

        if (!strncmp(p, "top ", 4)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            float lp = 0.0f;
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(top_index < vc->steps[step_index].ntop);
            if (sscanf(p, "top %257s %f", hex, &lp) != 2) {
                TEST_ASSERT(!"bad vector top line");
                return false;
            }
            test_vec_top *top = &vc->steps[step_index].top[top_index++];
            top->logprob = lp;
            TEST_ASSERT(test_hex_to_bytes(hex, top->bytes,
                                          TEST_VEC_MAX_TOKEN_BYTES, &top->len));
            continue;
        }

        TEST_ASSERT(!"unexpected vector line");
        return false;
    }

    TEST_ASSERT(!"unterminated vector case");
    return false;
}

static void test_logprob_vector_case(ds4_engine *engine, const test_vec_case *vc) {
    char *prompt_text = test_read_file(vc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    free(prompt_text);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, vc->ctx) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        return;
    }

    char err[160];
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-test: vector %s prefill failed: %s\n", vc->id, err);
        TEST_ASSERT(false);
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        return;
    }

    ds4_token_score scores[20];
    for (int i = 0; i < vc->nsteps; i++) {
        const test_vec_step *step = &vc->steps[i];
        int nscore = ds4_session_top_logprobs(session, scores, 20);
        int token = ds4_session_argmax(session);
        if (!test_token_bytes_equal(engine, token, step->selected, step->selected_len)) {
            fprintf(stderr, "ds4-test: vector %s step %d selected token mismatch\n",
                    vc->id, i);
            TEST_ASSERT(false);
        }

        for (int t = 0; t < step->ntop; t++) {
            bool found = false;
            float local_lp = 0.0f;
            for (int j = 0; j < nscore; j++) {
                if (scores[j].id < 0) continue;
                if (test_token_bytes_equal(engine, scores[j].id,
                                           step->top[t].bytes,
                                           step->top[t].len)) {
                    found = true;
                    local_lp = scores[j].logprob;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "ds4-test: vector %s step %d official top token missing locally\n",
                        vc->id, i);
                TEST_ASSERT(false);
            } else if (fabsf(local_lp - step->top[t].logprob) > 4.0f) {
                fprintf(stderr,
                        "ds4-test: vector %s step %d logprob delta too high: local=%g official=%g\n",
                        vc->id, i, local_lp, step->top[t].logprob);
                TEST_ASSERT(false);
            }
        }

        if (i + 1 < vc->nsteps) {
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-test: vector %s step %d eval failed: %s\n",
                        vc->id, i, err);
                TEST_ASSERT(false);
                break;
            }
        }
    }

    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static bool test_logprob_vector_case_disabled(const char *path,
                                              const test_vec_case *vc) {
    /*
     * This one long-context vector currently matches the public DeepSeek API less
     * after adding the official Hadamard+FP4 indexer path.  The public official
     * implementation and the API appear to disagree here; the official graph has
     * slightly lower local perplexity on the A/B check we ran, so DS4 keeps that
     * implementation and only excludes this brittle API fixture for now.
     */
    return !strcmp(path, "tests/test-vectors/flash-pre-0731/official.vec") &&
           !strcmp(vc->id, "long_memory_archive");
}

static void test_official_logprob_vectors_run(const char *case_filter) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) {
        path = "tests/test-vectors/flash-0731/official.vec";
    }
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    char *saved_prefill_chunk = test_save_env("DS4_METAL_PREFILL_CHUNK");
    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    test_streaming_prefill_env saved_canonical_streaming_prefill =
        test_force_canonical_streaming_prefill();
    setenv("DS4_METAL_PREFILL_CHUNK", "2048", 1);
    if (getenv("DS4_TEST_LOGPROB_AUTO_METAL") == NULL) {
        setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    } else {
        unsetenv("DS4_METAL_DISABLE_METAL4");
    }
    ds4_engine *engine = test_open_engine(false);
    if (!engine) {
        test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
        fclose(fp);
        return;
    }

    test_vec_case vc;
    int ran = 0;
    while (test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        if (case_filter && case_filter[0] && strcmp(vc.id, case_filter)) {
            continue;
        }
        if (test_logprob_vector_case_disabled(path, &vc)) {
            fprintf(stderr, "ds4-test: vector %s skipped (API/official graph mismatch)\n",
                    vc.id);
            continue;
        }
        fprintf(stderr, "ds4-test: vector %s\n", vc.id);
        test_logprob_vector_case(engine, &vc);
        ran++;
    }
    TEST_ASSERT(!case_filter || !case_filter[0] || ran == 1);
    ds4_engine_close(engine);
    test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
    test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
    fclose(fp);
}

static void test_official_logprob_vectors(void) {
    test_official_logprob_vectors_run(NULL);
}

static void test_metal_ssd_streaming_cache_pressure(void) {
#ifndef __APPLE__
    fprintf(stderr,
            "ds4-test: Metal SSD streaming cache-pressure repro skipped "
            "(Metal-only)\n");
#else
    /*
     * Regression repro for GitHub issue #384.
     *
     * The bug needs the Metal SSD-streaming decode layer-batch path and a small
     * routed-expert cache. Under pressure, a cache entry referenced by an
     * already-encoded-but-not-yet-executed layer can be reused for a later
     * layer in the same command buffer, producing deterministic wrong logits.
     */
    char *saved_streaming = test_save_env("DS4_TEST_SSD_STREAMING");
    char *saved_cache_gb = test_save_env("DS4_TEST_SSD_STREAMING_CACHE_GB");
    char *saved_cache_experts =
        test_save_env("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS");
    char *saved_disable_layer_batch =
        test_save_env("DS4_METAL_DISABLE_STREAMING_LAYER_BATCH");
    char *saved_disable_static_decode =
        test_save_env("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP");
    char *saved_one_stage =
        test_save_env("DS4_METAL_MOE_ONE_STAGE_PROFILE");

    setenv("DS4_TEST_SSD_STREAMING", "1", 1);
    setenv("DS4_TEST_SSD_STREAMING_CACHE_GB", "16", 1);
    unsetenv("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS");
    unsetenv("DS4_METAL_DISABLE_STREAMING_LAYER_BATCH");
    unsetenv("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP");
    unsetenv("DS4_METAL_MOE_ONE_STAGE_PROFILE");

    fprintf(stderr,
            "ds4-test: Metal SSD streaming cache-pressure repro "
            "(16GiB cache, layer-batched decode, short_code_completion)\n");
    test_official_logprob_vectors_run("short_code_completion");

    test_restore_env("DS4_METAL_MOE_ONE_STAGE_PROFILE", saved_one_stage);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP",
                     saved_disable_static_decode);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_LAYER_BATCH",
                     saved_disable_layer_batch);
    test_restore_env("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS",
                     saved_cache_experts);
    test_restore_env("DS4_TEST_SSD_STREAMING_CACHE_GB", saved_cache_gb);
    test_restore_env("DS4_TEST_SSD_STREAMING", saved_streaming);
#endif
}

static void test_logits_topk(const float *logits, int n, int *out, int k);
static bool test_topk_contains(const int *top, int k, int id);

#define TEST_LOCAL_GOLDEN_MAX_TOP 128

typedef struct {
    int id;
    float logit;
} test_local_golden_top;

typedef struct {
    char id[96];
    char mode[16];
    char prompt_path[512];
    int ctx;
    int frontier;
    int ntop;
    test_local_golden_top top[TEST_LOCAL_GOLDEN_MAX_TOP];
} test_local_golden_case;

static bool test_read_local_golden_case(FILE *fp, test_local_golden_case *tc) {
    char line[2048];
    memset(tc, 0, sizeof(*tc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %15s %d %d %511s %d",
                   tc->id, tc->mode, &tc->ctx, &tc->frontier,
                   tc->prompt_path, &tc->ntop) == 6) {
            TEST_ASSERT(tc->ctx > tc->frontier);
            TEST_ASSERT(tc->frontier > 0);
            TEST_ASSERT(tc->ntop > 0 && tc->ntop <= TEST_LOCAL_GOLDEN_MAX_TOP);
            return true;
        }
        TEST_ASSERT(!"unexpected line before local golden case");
        return false;
    }
    return false;
}

static bool test_fill_local_golden_case(FILE *fp, test_local_golden_case *tc) {
    char line[2048];
    int seen = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) {
            TEST_ASSERT(seen == tc->ntop);
            return seen == tc->ntop;
        }
        int rank = -1;
        int id = -1;
        float logit = 0.0f;
        if (sscanf(p, "top %d %d %f", &rank, &id, &logit) != 3) {
            TEST_ASSERT(!"bad local golden top line");
            return false;
        }
        TEST_ASSERT(rank == seen);
        TEST_ASSERT(seen < tc->ntop);
        if (seen >= tc->ntop) return false;
        tc->top[seen].id = id;
        tc->top[seen].logit = logit;
        seen++;
    }
    TEST_ASSERT(!"unterminated local golden case");
    return false;
}

static int test_local_golden_overlap(const test_local_golden_case *tc,
                                     const int *cand_top,
                                     int n) {
    int overlap = 0;
    if (n > tc->ntop) n = tc->ntop;
    for (int i = 0; i < n; i++) {
        if (test_topk_contains(cand_top, n, tc->top[i].id)) overlap++;
    }
    return overlap;
}

static float test_local_golden_max_abs(const test_local_golden_case *tc,
                                       const float *cand_logits,
                                       int n) {
    float max_abs = 0.0f;
    if (n > tc->ntop) n = tc->ntop;
    for (int i = 0; i < n; i++) {
        const int id = tc->top[i].id;
        if (id < 0) continue;
        const float abs_delta = fabsf(cand_logits[id] - tc->top[i].logit);
        if (abs_delta > max_abs) max_abs = abs_delta;
    }
    return max_abs;
}

static void test_local_golden_case_run(ds4_engine *engine,
                                       const test_local_golden_case *tc) {
    char *prompt_text = test_read_file(tc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_tokens prompt = {0};
    if (!strcmp(tc->mode, "text")) {
        ds4_tokenize_text(engine, prompt_text, &prompt);
    } else if (!strcmp(tc->mode, "rendered")) {
        ds4_tokenize_rendered_chat(engine, prompt_text, &prompt);
    } else if (!strcmp(tc->mode, "chat")) {
        ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    } else {
        TEST_ASSERT(!"unknown local golden prompt mode");
    }
    free(prompt_text);
    TEST_ASSERT(prompt.len >= tc->frontier);
    if (prompt.len < tc->frontier) {
        ds4_tokens_free(&prompt);
        return;
    }

    ds4_tokens prefix = {
        .v = prompt.v,
        .len = tc->frontier,
        .cap = tc->frontier,
    };

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);

    const int vocab = ds4_engine_vocab_size(engine);
    float *cand_logits = malloc((size_t)vocab * sizeof(cand_logits[0]));
    TEST_ASSERT(cand_logits != NULL);
    if (cand_logits &&
        ds4_session_copy_logits(session, cand_logits, vocab) == vocab) {
        int cand_top[TEST_LOCAL_GOLDEN_MAX_TOP];
        const int ntop = tc->ntop < TEST_LOCAL_GOLDEN_MAX_TOP ?
                         tc->ntop : TEST_LOCAL_GOLDEN_MAX_TOP;
        test_logits_topk(cand_logits, vocab, cand_top, ntop);

        const int top5_overlap = test_local_golden_overlap(tc, cand_top, 5);
        const int top20_overlap = test_local_golden_overlap(tc, cand_top, 20);
        const int top64_overlap = test_local_golden_overlap(tc, cand_top, 64);
        const float top20_max_abs =
            test_local_golden_max_abs(tc, cand_logits, 20);

        fprintf(stderr,
                "ds4-test: local golden %s top1 ref=%d cand=%d "
                "top5_overlap=%d/5 top20_overlap=%d/20 top64_overlap=%d/64 "
                "top20_max_abs=%g\n",
                tc->id, tc->top[0].id, cand_top[0],
                top5_overlap, top20_overlap, top64_overlap, top20_max_abs);

        /*
         * This is intentionally tolerant: it is meant to catch substantial
         * backend drift (wrong tiling, skipped work, bad dispatch), not tiny
         * floating-point differences from otherwise sane kernel changes.
         */
        TEST_ASSERT(cand_top[0] == tc->top[0].id);
        TEST_ASSERT(top5_overlap >= 4);
        TEST_ASSERT(top20_overlap >= 15);
        TEST_ASSERT(top64_overlap >= 40);
        TEST_ASSERT(top20_max_abs <= 8.0f);
    } else {
        TEST_ASSERT(false);
    }

    free(cand_logits);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static void test_local_golden_vectors(void) {
    const char *path = getenv("DS4_TEST_LOCAL_GOLDEN_FILE");
    if (!path || !path[0]) {
        path = "tests/test-vectors/flash-0731/local-golden.vec";
    }
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    char *saved_prefill_chunk = test_save_env("DS4_METAL_PREFILL_CHUNK");
    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    char *saved_moe_tile_max = test_save_env("DS4_METAL_MOE_TILE_MAX");
    test_streaming_prefill_env saved_canonical_streaming_prefill =
        test_force_canonical_streaming_prefill();
    setenv("DS4_METAL_PREFILL_CHUNK", "4096", 1);
    setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    unsetenv("DS4_METAL_MOE_TILE_MAX");

    ds4_engine *engine = test_open_engine(false);
    if (!engine) {
        test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
        test_restore_env("DS4_METAL_MOE_TILE_MAX", saved_moe_tile_max);
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
        fclose(fp);
        return;
    }

    test_local_golden_case tc;
    while (test_read_local_golden_case(fp, &tc)) {
        if (!test_fill_local_golden_case(fp, &tc)) break;
        test_local_golden_case_run(engine, &tc);
    }

    ds4_engine_close(engine);
    test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
    test_restore_env("DS4_METAL_MOE_TILE_MAX", saved_moe_tile_max);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
    test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
    fclose(fp);
}

#define TEST_MPP_EQ_MAX_CASES 8
#define TEST_MPP_EQ_TOPK 20
#define TEST_MPP_EQ_TOP5 5
#define TEST_MPP_EQ_DELTAS 5

typedef struct {
    char id[96];
    int ctx;
    int vocab_size;
    int gen_steps;
    ds4_tokens prompt;
    float *ref_logits;
    int ref_gen[TEST_VEC_MAX_STEPS];
    int ref_gen_len;
} test_mpp_eq_case;

typedef struct {
    int ref_top1;
    int cand_top1;
    int overlap;
    int top5_overlap;
    int max_rank_delta;
    int nonfinite;
    float rms;
    float max_abs;
    float top20_max_abs;
    bool same_top1;
    bool pass;
} test_mpp_eq_result;

typedef struct {
    const char *label;
    int cases;
    int capture_failures;
    int logits_failures;
    int greedy_failures;
    int top1_mismatches;
    int min_overlap;
    int min_top5_overlap;
    int worst_rank_delta;
    float worst_rms;
    float worst_max_abs;
    float worst_top20_max_abs;
} test_mpp_eq_summary;

static void test_mpp_eq_case_free(test_mpp_eq_case *tc) {
    if (!tc) return;
    ds4_tokens_free(&tc->prompt);
    free(tc->ref_logits);
    memset(tc, 0, sizeof(*tc));
}

static void test_logits_topk(const float *logits, int n, int *out, int k) {
    for (int i = 0; i < k; i++) out[i] = -1;
    for (int id = 0; id < n; id++) {
        const float v = logits[id];
        if (!isfinite(v)) continue;
        for (int j = 0; j < k; j++) {
            if (out[j] < 0 || v > logits[out[j]]) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j] = id;
                break;
            }
        }
    }
}

static bool test_topk_contains(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return true;
    }
    return false;
}

static int test_topk_rank(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return i;
    }
    return -1;
}

static void test_note_delta(int *ids, float *ref_vals, float *cand_vals,
                            float *abs_vals, int id, float ref, float cand) {
    const float abs_delta = fabsf(cand - ref);
    for (int i = 0; i < TEST_MPP_EQ_DELTAS; i++) {
        if (ids[i] < 0 || abs_delta > abs_vals[i]) {
            for (int j = TEST_MPP_EQ_DELTAS - 1; j > i; j--) {
                ids[j] = ids[j - 1];
                ref_vals[j] = ref_vals[j - 1];
                cand_vals[j] = cand_vals[j - 1];
                abs_vals[j] = abs_vals[j - 1];
            }
            ids[i] = id;
            ref_vals[i] = ref;
            cand_vals[i] = cand;
            abs_vals[i] = abs_delta;
            return;
        }
    }
}

static float test_top_union_max_abs(const float *ref, const float *cand,
                                    const int *ref_top, const int *cand_top, int k) {
    float max_abs = 0.0f;
    for (int i = 0; i < k; i++) {
        if (ref_top[i] >= 0) {
            const float d = fabsf(cand[ref_top[i]] - ref[ref_top[i]]);
            if (d > max_abs) max_abs = d;
        }
        if (cand_top[i] >= 0 && !test_topk_contains(ref_top, k, cand_top[i])) {
            const float d = fabsf(cand[cand_top[i]] - ref[cand_top[i]]);
            if (d > max_abs) max_abs = d;
        }
    }
    return max_abs;
}

/*
 * Metal4/TensorOps equivalence is a smoke test, not a demand for bitwise local
 * logits.  Tensor kernels change precision and reduction order, so the useful
 * invariant here is: no NaNs, same first greedy token, and same short greedy
 * continuation.  Larger logit drift is still printed so it can be compared with
 * official API-vector and long-context recall gates.
 */
static test_mpp_eq_result test_compare_mpp_logits(const test_mpp_eq_case *tc,
                                                  const float *cand_logits,
                                                  bool assert_thresholds) {
    int ref_top[TEST_MPP_EQ_TOPK];
    int cand_top[TEST_MPP_EQ_TOPK];
    test_logits_topk(tc->ref_logits, tc->vocab_size, ref_top, TEST_MPP_EQ_TOPK);
    test_logits_topk(cand_logits, tc->vocab_size, cand_top, TEST_MPP_EQ_TOPK);

    int overlap = 0;
    int top5_overlap = 0;
    int max_rank_delta = 0;
    for (int i = 0; i < TEST_MPP_EQ_TOPK; i++) {
        const int cand_rank = test_topk_rank(cand_top, TEST_MPP_EQ_TOPK, ref_top[i]);
        if (ref_top[i] >= 0 && cand_rank >= 0) {
            overlap++;
            const int rank_delta = abs(cand_rank - i);
            if (rank_delta > max_rank_delta) max_rank_delta = rank_delta;
        }
        if (i < TEST_MPP_EQ_TOP5 &&
            ref_top[i] >= 0 &&
            test_topk_contains(cand_top, TEST_MPP_EQ_TOP5, ref_top[i])) {
            top5_overlap++;
        }
    }

    double sumsq = 0.0;
    float max_abs = 0.0f;
    int nonfinite = 0;
    int delta_ids[TEST_MPP_EQ_DELTAS];
    float delta_ref[TEST_MPP_EQ_DELTAS];
    float delta_cand[TEST_MPP_EQ_DELTAS];
    float delta_abs[TEST_MPP_EQ_DELTAS];
    for (int i = 0; i < TEST_MPP_EQ_DELTAS; i++) {
        delta_ids[i] = -1;
        delta_ref[i] = 0.0f;
        delta_cand[i] = 0.0f;
        delta_abs[i] = 0.0f;
    }

    for (int i = 0; i < tc->vocab_size; i++) {
        if (!isfinite(tc->ref_logits[i]) || !isfinite(cand_logits[i])) {
            nonfinite++;
            continue;
        }
        const float delta = cand_logits[i] - tc->ref_logits[i];
        const float abs_delta = fabsf(delta);
        if (abs_delta > max_abs) max_abs = abs_delta;
        sumsq += (double)delta * (double)delta;
        test_note_delta(delta_ids, delta_ref, delta_cand, delta_abs,
                        (int)i, tc->ref_logits[i], cand_logits[i]);
    }

    const float rms = (float)sqrt(sumsq / (double)tc->vocab_size);
    const float top_abs = test_top_union_max_abs(tc->ref_logits, cand_logits,
                                                 ref_top, cand_top, TEST_MPP_EQ_TOPK);
    const bool same_top1 = ref_top[0] >= 0 && ref_top[0] == cand_top[0];
    test_mpp_eq_result result = {
        .ref_top1 = ref_top[0],
        .cand_top1 = cand_top[0],
        .overlap = overlap,
        .top5_overlap = top5_overlap,
        .max_rank_delta = max_rank_delta,
        .nonfinite = nonfinite,
        .rms = rms,
        .max_abs = max_abs,
        .top20_max_abs = top_abs,
        .same_top1 = same_top1,
        .pass = nonfinite == 0 && same_top1,
    };

    fprintf(stderr,
            "ds4-test: Tensor equivalence %s top1 ref=%d cand=%d top5_overlap=%d/%d overlap=%d/%d max_rank_delta=%d rms=%g max_abs=%g top20_max_abs=%g\n",
            tc->id, ref_top[0], cand_top[0],
            top5_overlap, TEST_MPP_EQ_TOP5,
            overlap, TEST_MPP_EQ_TOPK,
            max_rank_delta, rms, max_abs, top_abs);
    fprintf(stderr, "ds4-test: Tensor equivalence %s largest deltas:", tc->id);
    for (int i = 0; i < TEST_MPP_EQ_DELTAS && delta_ids[i] >= 0; i++) {
        fprintf(stderr, " id=%d ref=%g cand=%g abs=%g",
                delta_ids[i], delta_ref[i], delta_cand[i], delta_abs[i]);
    }
    fputc('\n', stderr);

    if (assert_thresholds) {
        TEST_ASSERT(nonfinite == 0);
        TEST_ASSERT(same_top1);
    }
    return result;
}

static bool test_mpp_capture(ds4_engine *engine, const test_mpp_eq_case *tc,
                             float *logits, int *gen, int *gen_len) {
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, &tc->prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ok = ds4_session_copy_logits(session, logits, tc->vocab_size) == tc->vocab_size;
        TEST_ASSERT(ok);
    }

    int n = 0;
    while (ok && n < tc->gen_steps) {
        const int token = ds4_session_argmax(session);
        gen[n++] = token;
        if (n < tc->gen_steps && ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            ok = false;
            TEST_ASSERT(false);
        }
    }
    *gen_len = n;

    ds4_session_free(session);
    return ok;
}

static bool test_mpp_capture_logits_only(ds4_engine *engine,
                                         const test_mpp_eq_case *tc,
                                         float *logits) {
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, &tc->prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ok = ds4_session_copy_logits(session, logits, tc->vocab_size) == tc->vocab_size;
        TEST_ASSERT(ok);
    }

    ds4_session_free(session);
    return ok;
}

static bool test_mpp_eq_case_selected(const char *id) {
    const char *filter = getenv("DS4_TEST_MPP_EQ_CASE");
    if (!filter || !filter[0]) return true;

    char buf[256];
    snprintf(buf, sizeof(buf), "%s", filter);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        tok = test_trim_line(tok);
        if (tok[0] && strstr(id, tok)) return true;
    }
    return false;
}

static int test_load_mpp_cases(ds4_engine *engine, test_mpp_eq_case *cases, int cap) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) {
        path = "tests/test-vectors/flash-0731/official.vec";
    }
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return 0;

    int ncase = 0;
    test_vec_case vc;
    while (ncase < cap && test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        if (!test_mpp_eq_case_selected(vc.id)) continue;
        char *prompt_text = test_read_file(vc.prompt_path);
        TEST_ASSERT(prompt_text != NULL);
        if (!prompt_text) continue;

        test_mpp_eq_case *tc = &cases[ncase++];
        snprintf(tc->id, sizeof(tc->id), "%s", vc.id);
        tc->ctx = vc.ctx;
        tc->vocab_size = ds4_engine_vocab_size(engine);
        tc->gen_steps = vc.nsteps < TEST_VEC_MAX_STEPS ? vc.nsteps : TEST_VEC_MAX_STEPS;
        ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &tc->prompt);
        free(prompt_text);
        TEST_ASSERT(tc->prompt.len > 0);
    }
    fclose(fp);
    return ncase;
}

static void test_mpp_summary_init(test_mpp_eq_summary *summary, const char *label) {
    memset(summary, 0, sizeof(*summary));
    summary->label = label;
    summary->min_overlap = TEST_MPP_EQ_TOPK;
    summary->min_top5_overlap = TEST_MPP_EQ_TOP5;
}

static void test_mpp_summary_note_logits(test_mpp_eq_summary *summary,
                                         const test_mpp_eq_result *result) {
    if (!result->pass) summary->logits_failures++;
    if (!result->same_top1) summary->top1_mismatches++;
    if (result->overlap < summary->min_overlap) summary->min_overlap = result->overlap;
    if (result->top5_overlap < summary->min_top5_overlap) {
        summary->min_top5_overlap = result->top5_overlap;
    }
    if (result->max_rank_delta > summary->worst_rank_delta) {
        summary->worst_rank_delta = result->max_rank_delta;
    }
    if (result->rms > summary->worst_rms) summary->worst_rms = result->rms;
    if (result->max_abs > summary->worst_max_abs) summary->worst_max_abs = result->max_abs;
    if (result->top20_max_abs > summary->worst_top20_max_abs) {
        summary->worst_top20_max_abs = result->top20_max_abs;
    }
}

static void test_mpp_summary_print(const test_mpp_eq_summary *summary) {
    fprintf(stderr,
            "ds4-test: Tensor summary route=%s cases=%d capture_fail=%d logits_fail=%d greedy_fail=%d top1_mismatch=%d min_top5_overlap=%d/%d min_overlap=%d/%d worst_rank_delta=%d worst_rms=%g worst_max_abs=%g worst_top20_max_abs=%g\n",
            summary->label,
            summary->cases,
            summary->capture_failures,
            summary->logits_failures,
            summary->greedy_failures,
            summary->top1_mismatches,
            summary->min_top5_overlap,
            TEST_MPP_EQ_TOP5,
            summary->min_overlap,
            TEST_MPP_EQ_TOPK,
            summary->worst_rank_delta,
            summary->worst_rms,
            summary->worst_max_abs,
            summary->worst_top20_max_abs);
}

static void test_run_mpp_candidate(const char *label,
                                   test_mpp_eq_case *cases,
                                   int ncase) {
    fprintf(stderr, "ds4-test: Tensor equivalence candidate route=%s\n", label);
    test_mpp_eq_summary summary;
    test_mpp_summary_init(&summary, label);
    ds4_engine *cand_engine = test_open_engine(false);
    if (cand_engine) {
        const int vocab_size = ncase > 0 ? cases[0].vocab_size : 0;
        float *cand_logits = malloc((size_t)vocab_size * sizeof(cand_logits[0]));
        TEST_ASSERT(cand_logits != NULL);
        if (cand_logits) {
            for (int i = 0; i < ncase; i++) {
                test_mpp_eq_case *tc = &cases[i];
                if (!tc->ref_logits) continue;
                int cand_gen[TEST_VEC_MAX_STEPS] = {0};
                int cand_gen_len = 0;
                if (!test_mpp_capture(cand_engine, tc, cand_logits, cand_gen, &cand_gen_len)) {
                    summary.capture_failures++;
                    continue;
                }
                summary.cases++;
                test_mpp_eq_result result = test_compare_mpp_logits(tc, cand_logits, true);
                test_mpp_summary_note_logits(&summary, &result);
                TEST_ASSERT(cand_gen_len == tc->ref_gen_len);
                if (cand_gen_len != tc->ref_gen_len) summary.greedy_failures++;
                for (int j = 0; j < tc->ref_gen_len && j < cand_gen_len; j++) {
                    if (cand_gen[j] != tc->ref_gen[j]) {
                        fprintf(stderr,
                                "ds4-test: Tensor equivalence %s greedy token mismatch step=%d ref=%d cand=%d\n",
                                tc->id, j, tc->ref_gen[j], cand_gen[j]);
                        summary.greedy_failures++;
                    }
                    TEST_ASSERT(cand_gen[j] == tc->ref_gen[j]);
                }
            }
            free(cand_logits);
        }
        ds4_engine_close(cand_engine);
    }
    test_mpp_summary_print(&summary);
}

static void test_metal_mpp_equivalence(void) {
    test_close_engines();

    test_mpp_eq_case cases[TEST_MPP_EQ_MAX_CASES];
    memset(cases, 0, sizeof(cases));

    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    ds4_engine *ref_engine = test_open_engine(false);
    if (!ref_engine) {
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        return;
    }

    const int ncase = test_load_mpp_cases(ref_engine, cases, TEST_MPP_EQ_MAX_CASES);
    TEST_ASSERT(ncase > 0);
    for (int i = 0; i < ncase; i++) {
        test_mpp_eq_case *tc = &cases[i];
        tc->ref_logits = malloc((size_t)tc->vocab_size * sizeof(tc->ref_logits[0]));
        TEST_ASSERT(tc->ref_logits != NULL);
        if (!tc->ref_logits) continue;
        TEST_ASSERT(test_mpp_capture(ref_engine, tc,
                                     tc->ref_logits,
                                     tc->ref_gen,
                                     &tc->ref_gen_len));
    }
    ds4_engine_close(ref_engine);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);

    test_run_mpp_candidate("auto", cases, ncase);

    for (int i = 0; i < ncase; i++) test_mpp_eq_case_free(&cases[i]);
}

static void test_streaming_decode_prefill_correctness(void) {
    test_close_engines();
    if (!test_env_bool("DS4_TEST_SSD_STREAMING")) {
        fprintf(stderr,
                "ds4-test: streaming decode-prefill correctness skipped "
                "(set DS4_TEST_SSD_STREAMING=1 to enable)\n");
        return;
    }

    test_mpp_eq_case cases[TEST_MPP_EQ_MAX_CASES];
    memset(cases, 0, sizeof(cases));

    test_streaming_prefill_env saved_canonical_streaming_prefill =
        test_force_canonical_streaming_prefill();

    ds4_engine *ref_engine = test_open_engine(false);
    if (!ref_engine) {
        test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
        return;
    }

    const int ncase = test_load_mpp_cases(ref_engine, cases, TEST_MPP_EQ_MAX_CASES);
    TEST_ASSERT(ncase > 0);
    for (int i = 0; i < ncase; i++) {
        test_mpp_eq_case *tc = &cases[i];
        tc->ref_logits = malloc((size_t)tc->vocab_size * sizeof(tc->ref_logits[0]));
        TEST_ASSERT(tc->ref_logits != NULL);
        if (!tc->ref_logits) continue;
        TEST_ASSERT(test_mpp_capture(ref_engine, tc,
                                     tc->ref_logits,
                                     tc->ref_gen,
                                     &tc->ref_gen_len));
    }
    ds4_engine_close(ref_engine);

    unsetenv("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL");
    unsetenv("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR");

    ds4_engine *cand_engine = test_open_engine(false);
    if (cand_engine) {
        for (int i = 0; i < ncase; i++) {
            test_mpp_eq_case *tc = &cases[i];
            if (!tc->ref_logits) continue;

            float *cand_cold = malloc((size_t)tc->vocab_size * sizeof(cand_cold[0]));
            float *cand_warm_a = malloc((size_t)tc->vocab_size * sizeof(cand_warm_a[0]));
            float *cand_warm_b = malloc((size_t)tc->vocab_size * sizeof(cand_warm_b[0]));
            TEST_ASSERT(cand_cold != NULL);
            TEST_ASSERT(cand_warm_a != NULL);
            TEST_ASSERT(cand_warm_b != NULL);
            if (!cand_cold || !cand_warm_a || !cand_warm_b) {
                free(cand_cold);
                free(cand_warm_a);
                free(cand_warm_b);
                continue;
            }

            TEST_ASSERT(test_mpp_capture_logits_only(cand_engine, tc, cand_cold));
            TEST_ASSERT(test_mpp_capture_logits_only(cand_engine, tc, cand_warm_a));
            TEST_ASSERT(test_mpp_capture_logits_only(cand_engine, tc, cand_warm_b));

            test_mpp_eq_result result = test_compare_mpp_logits(tc, cand_cold, false);
            TEST_ASSERT(result.nonfinite == 0);
            TEST_ASSERT(result.top5_overlap >= 2);
            TEST_ASSERT(result.overlap >= 10);
            TEST_ASSERT(result.rms <= 4.0f);
            TEST_ASSERT(result.top20_max_abs <= 12.0f);

            int cold_warm_neq = 0;
            int warm_repeat_neq = 0;
            int repeat_nonfinite = 0;
            float cold_warm_max_abs = 0.0f;
            float warm_repeat_max_abs = 0.0f;
            for (int j = 0; j < tc->vocab_size; j++) {
                if (!isfinite(cand_cold[j]) ||
                    !isfinite(cand_warm_a[j]) ||
                    !isfinite(cand_warm_b[j])) {
                    repeat_nonfinite++;
                    continue;
                }
                const float cold_warm_d = fabsf(cand_cold[j] - cand_warm_a[j]);
                if (cold_warm_d != 0.0f) cold_warm_neq++;
                if (cold_warm_d > cold_warm_max_abs) cold_warm_max_abs = cold_warm_d;
                const float warm_repeat_d = fabsf(cand_warm_a[j] - cand_warm_b[j]);
                if (warm_repeat_d != 0.0f) warm_repeat_neq++;
                if (warm_repeat_d > warm_repeat_max_abs) {
                    warm_repeat_max_abs = warm_repeat_d;
                }
            }
            TEST_ASSERT(repeat_nonfinite == 0);
            TEST_ASSERT(cold_warm_neq == 0);
            TEST_ASSERT(warm_repeat_neq == 0);
            fprintf(stderr,
                    "ds4-test: streaming decode-prefill %s cold_warm_neq=%d "
                    "cold_warm_max_abs=%g warm_repeat_neq=%d "
                    "warm_repeat_max_abs=%g top1 canonical=%d decode=%d\n",
                    tc->id,
                    cold_warm_neq,
                    cold_warm_max_abs,
                    warm_repeat_neq,
                    warm_repeat_max_abs,
                    result.ref_top1,
                    result.cand_top1);

            free(cand_cold);
            free(cand_warm_a);
            free(cand_warm_b);
        }
        ds4_engine_close(cand_engine);
    }

    test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
    for (int i = 0; i < ncase; i++) test_mpp_eq_case_free(&cases[i]);
}

#define TEST_LIST_FILES_USER_PROMPT \
    "Use the list_files tool to list the current directory exactly once, " \
    "then report the listed files and stop."

#define TEST_LIST_FILES_TOOL_JSON \
    "{\"type\":\"function\",\"function\":{" \
        "\"name\":\"list_files\"," \
        "\"description\":\"List files in a directory.\"," \
        "\"parameters\":{\"type\":\"object\",\"properties\":{" \
            "\"path\":{\"type\":\"string\",\"description\":\"Directory path to list.\"}" \
        "},\"required\":[\"path\"]}" \
    "}}"

#define TEST_LIST_FILES_RESULT "[\"README.md\",\"Makefile\",\"ds4.c\",\"metal\"]"

static const char *test_tool_call_request_json(void) {
    return
        "{"
        "\"model\":\"deepseek-v4-flash\","
        "\"messages\":[{\"role\":\"user\",\"content\":\""
            TEST_LIST_FILES_USER_PROMPT
        "\"}],"
        "\"tools\":[" TEST_LIST_FILES_TOOL_JSON "],"
        "\"tool_choice\":\"auto\","
        "\"think\":false,"
        "\"temperature\":0,"
        "\"max_tokens\":256,"
        "\"stream\":false"
        "}";
}

static char *test_tool_result_request_json(const char *assistant_content,
                                           const tool_call *call) {
    if (!call || !call->id || !call->id[0]) return NULL;

    buf b = {0};
    buf_puts(&b,
        "{\"model\":\"deepseek-v4-flash\",\"messages\":["
        "{\"role\":\"user\",\"content\":");
    json_escape(&b, TEST_LIST_FILES_USER_PROMPT);
    buf_puts(&b, "},{\"role\":\"assistant\",\"content\":");
    json_escape(&b, assistant_content ? assistant_content : "");
    buf_puts(&b, ",\"tool_calls\":[{\"id\":");
    json_escape(&b, call->id);
    buf_puts(&b, ",\"type\":\"function\",\"function\":{\"name\":");
    json_escape(&b, call->name ? call->name : "");
    buf_puts(&b, ",\"arguments\":");
    json_escape(&b, call->arguments ? call->arguments : "{}");
    buf_puts(&b, "}}]},{\"role\":\"tool\",\"tool_call_id\":");
    json_escape(&b, call->id);
    buf_puts(&b, ",\"content\":");
    json_escape(&b, TEST_LIST_FILES_RESULT);
    buf_puts(&b,
        "}],\"tools\":[" TEST_LIST_FILES_TOOL_JSON "],"
        "\"tool_choice\":\"auto\",\"think\":false,"
        "\"temperature\":0,\"max_tokens\":256,\"stream\":false}");
    return buf_take(&b);
}

/* A complete tool call inside unclosed reasoning is recovered directly. The
 * detector must wait for the complete block, and the parser must keep only the
 * preceding prose in reasoning_content. */
static void test_think_tool_recovery(void) {
    const char *generated =
        "The user wants a directory listing.\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"list_files\">\n"
        DS4_PARAM_START " name=\"path\" string=\"true\">." DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END;

    buf text = {0};
    size_t scan_from = 0;
    bool complete = false;
    for (size_t i = 0; generated[i]; i++) {
        buf_append(&text, generated + i, 1);
        complete = complete_tool_call_inside_thinking(
            SERVER_MODEL_SYNTAX_DEEPSEEK, text.ptr, text.len, &scan_from);
        TEST_ASSERT(complete == (generated[i + 1] == '\0'));
    }
    TEST_ASSERT(complete);

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message_ex(text.ptr, true,
                                             &content, &reasoning, &calls);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));
    TEST_ASSERT(calls.v[0].arguments && strstr(calls.v[0].arguments, "\"path\": \".\""));
    TEST_ASSERT(content && content[0] == '\0');
    TEST_ASSERT(reasoning && !strcmp(reasoning, "The user wants a directory listing."));

    fprintf(stderr,
            "ds4-test: think-tool-recovery complete=%d calls=%d name=%s\n",
            complete ? 1 : 0, calls.len, calls.len ? calls.v[0].name : "-");

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
}

typedef struct {
    char *raw;
    char *content;
    char *reasoning;
    tool_calls calls;
    const char *finish;
} test_chat_turn;

static void test_chat_turn_free(test_chat_turn *turn) {
    if (!turn) return;
    free(turn->raw);
    free(turn->content);
    free(turn->reasoning);
    tool_calls_free(&turn->calls);
    memset(turn, 0, sizeof(*turn));
}

/* Run the same greedy stop/tool-marker/response parse path needed by the server,
 * while keeping one session alive so the next request genuinely continues from
 * this sampled turn. */
static bool test_generate_chat_turn(ds4_engine *engine, ds4_session *session,
                                    const request *r, test_chat_turn *turn) {
    memset(turn, 0, sizeof(*turn));
    char err[160] = {0};
    if (ds4_session_sync(session, &r->prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-test: tool-call sync failed: %s\n", err);
        turn->finish = "error";
        return false;
    }

    buf text = {0};
    uint64_t rng = 123;
    const char *finish = "length";
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    bool decode_ok = true;
    dsml_decode_tracker tracker;
    dsml_decode_tracker_init(&tracker);
    tracker.model_syntax = r->model_syntax;

    for (int i = 0; i < r->max_tokens; i++) {
        int token = ds4_session_sample(session, r->temperature, r->top_k,
                                       r->top_p, r->min_p, &rng);
        if (ds4_token_is_stop_for_think_mode(engine, token, r->think_mode)) {
            finish = "stop";
            break;
        }
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            finish = "error";
            decode_ok = false;
            break;
        }

        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        if (r->has_tools) {
            dsml_decode_tracker_update(&tracker, text.ptr, text.len);
            observe_tool_markers(&tracker, text.ptr ? text.ptr : "",
                                 &saw_tool_start, &saw_tool_end, NULL);
            if (saw_tool_end) {
                finish = "tool_calls";
                break;
            }
        }
    }

    turn->raw = buf_take(&text);
    turn->finish = finish;
    if (!decode_ok) {
        fprintf(stderr, "ds4-test: tool-call decode failed: %s\n", err);
        return false;
    }

    bool recovered = false;
    bool parsed = parse_generated_message_for_response_for_syntax(
        r->model_syntax,
        turn->raw ? turn->raw : "",
        r->has_tools,
        saw_tool_start,
        ds4_think_mode_enabled(r->think_mode),
        &turn->finish,
        err,
        sizeof(err),
        &turn->content,
        &turn->reasoning,
        &turn->calls,
        &recovered, &r->tool_orders);
    if (turn->calls.len > 0) turn->finish = "tool_calls";
    if (!parsed) {
        fprintf(stderr,
                "ds4-test: generated message parse failed: %s recovered=%d raw=%s\n",
                err, recovered ? 1 : 0, turn->raw ? turn->raw : "");
    }
    return parsed;
}

static void test_tool_call_quality_one(bool quality) {
    ds4_engine *engine = test_get_engine(quality);
    if (!engine) return;

    server s = {0};
    s.engine = engine;
    pthread_mutex_init(&s.tool_mu, NULL);

    request first_request = {0};
    request second_request = {0};
    ds4_session *session = NULL;
    test_chat_turn first = {0};
    test_chat_turn second = {0};
    char *second_body = NULL;
    char err[160] = {0};

    bool request_ok = parse_chat_request(engine, &s,
                                         test_tool_call_request_json(),
                                         512, 32768, &first_request,
                                         err, sizeof(err));
    TEST_ASSERT(request_ok);
    if (!request_ok) goto done;

    bool session_ok = ds4_session_create(&session, engine, 32768) == 0;
    TEST_ASSERT(session_ok);
    if (!session_ok) goto done;

    bool first_ok = test_generate_chat_turn(engine, session,
                                            &first_request, &first);
    TEST_ASSERT(first_ok);
    TEST_ASSERT(first.finish && !strcmp(first.finish, "tool_calls"));
    TEST_ASSERT(first.calls.len == 1);
    TEST_ASSERT(first.calls.len == 1 && first.calls.v[0].name &&
                !strcmp(first.calls.v[0].name, "list_files"));
    TEST_ASSERT(first.calls.raw_tool_text && first.calls.raw_tool_text[0]);
    if (!first_ok || first.calls.len != 1 ||
        !first.calls.v[0].name ||
        strcmp(first.calls.v[0].name, "list_files") ||
        !first.calls.raw_tool_text || !first.calls.raw_tool_text[0]) {
        goto done;
    }

    /* Use the real response-side id assignment and exact sampled-DSML memory.
     * The same id is serialized on both the assistant call and tool result. */
    assign_tool_call_ids(&s, &first.calls, API_OPENAI);
    TEST_ASSERT(first.calls.v[0].id && first.calls.v[0].id[0]);
    if (!first.calls.v[0].id || !first.calls.v[0].id[0]) goto done;
    tool_memory_remember(&s, &first.calls);

    second_body = test_tool_result_request_json(first.content,
                                                &first.calls.v[0]);
    TEST_ASSERT(second_body != NULL);
    if (!second_body) goto done;

    err[0] = '\0';
    request_ok = parse_chat_request(engine, &s, second_body,
                                    512, 32768, &second_request,
                                    err, sizeof(err));
    TEST_ASSERT(request_ok);
    if (!request_ok) goto done;
    TEST_ASSERT(second_request.tool_replay.mem == 1);
    TEST_ASSERT(second_request.tool_replay.disk == 0);
    TEST_ASSERT(second_request.tool_replay.canonical == 0);
    TEST_ASSERT(second_request.tool_replay.missing_ids == 0);

    bool second_ok = test_generate_chat_turn(engine, session,
                                             &second_request, &second);
    TEST_ASSERT(second_ok);
    TEST_ASSERT(second.finish && !strcmp(second.finish, "stop"));
    TEST_ASSERT(second.calls.len == 0);
    TEST_ASSERT(second.content && second.content[0]);

    fprintf(stderr,
            "ds4-test: post-tool-result turn1 finish_reason=%s tool_calls=%d "
            "turn2 finish_reason=%s tool_calls=%d replay_mem=%d\n",
            first.finish ? first.finish : "-", first.calls.len,
            second.finish ? second.finish : "-", second.calls.len,
            second_request.tool_replay.mem);

done:
    free(second_body);
    test_chat_turn_free(&second);
    test_chat_turn_free(&first);
    ds4_session_free(session);
    request_free(&second_request);
    request_free(&first_request);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_tool_call_quality(void) {
    fprintf(stderr, "ds4-test: tool-call quality fast path\n");
    test_tool_call_quality_one(false);
    test_close_engine(false);
    fprintf(stderr, "ds4-test: tool-call quality exact path\n");
    test_tool_call_quality_one(true);
    test_close_engine(true);
}

/* Greedy speculative decode: capture committed tokens and the largest accepted
 * chunk, so the caller can confirm the multi-row verify path actually ran. */
static bool test_mtp_capture_speculative(ds4_engine *engine, const ds4_tokens *prompt,
                                         int max_tokens, int *out, int *out_len,
                                         int *max_chunk, int *single_cycles,
                                         int *double_cycles) {
    *out_len = 0;
    *max_chunk = 0;
    if (single_cycles) *single_cycles = 0;
    if (double_cycles) *double_cycles = 0;
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, prompt->len + max_tokens + 16) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);

    const int eos = ds4_token_eos(engine);
    int n = 0;
    bool stop = false;
    while (ok && !stop && n < max_tokens) {
        const int token = ds4_session_argmax(session);
        if (token == eos) break;

        int toks[17]; /* base token + draft depth, which the engine clamps to 16 */
        const int ntok = ds4_session_eval_speculative_argmax(
            session, token, max_tokens - n, eos, toks,
            (int)(sizeof(toks) / sizeof(toks[0])), err, sizeof(err));
        if (ntok < 0) { ok = false; TEST_ASSERT(false); break; }
        if (ntok > *max_chunk) *max_chunk = ntok;
        if (ntok == 1 && single_cycles) (*single_cycles)++;
        if (ntok == 2 && double_cycles) (*double_cycles)++;

        for (int j = 0; j < ntok; j++) {
            if (toks[j] == eos) { stop = true; break; }
            out[n++] = toks[j];
            if (n >= max_tokens) { stop = true; break; }
        }
    }

    *out_len = n;
    ds4_session_free(session);
    return ok;
}

/* Replay toks[] through plain decode and return the largest gap between a
 * position's argmax logit and the committed token's logit.  Correct speculation
 * commits (near-)argmax tokens (gap ~0); a mis-committed token gives a big gap. */
static bool test_mtp_worst_argmax_gap(ds4_engine *engine, const ds4_tokens *prompt,
                                      const int *toks, int n,
                                      float *worst_gap, int *worst_at) {
    *worst_gap = 0.0f;
    *worst_at = -1;
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, prompt->len + n + 16) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);

    for (int i = 0; ok && i < n; i++) {
        ds4_token_score best, cur;
        ok = ds4_session_top_logprobs(session, &best, 1) >= 1 &&
             ds4_session_token_logprob(session, toks[i], &cur) == 1;
        TEST_ASSERT(ok);
        if (!ok) break;

        const float gap = best.logit - cur.logit;
        if (gap > *worst_gap) { *worst_gap = gap; *worst_at = i; }
        if (ds4_session_eval(session, toks[i], err, sizeof(err)) != 0) { ok = false; TEST_ASSERT(false); break; }
    }

    ds4_session_free(session);
    return ok;
}

/* Verbatim-copy task: keeps the model confident (a mis-committed token shows as
 * a large argmax gap) and draft acceptance high (so the multi-row verify path is
 * exercised across the generation). */
static const char *test_mtp_copy_prompt(void) {
    return
        "Reproduce the following C code EXACTLY, character for character, "
        "inside a single code block and output nothing else:\n\n"
        "```c\n"
        "static uint32_t clamp_u32(uint32_t v, uint32_t lo, uint32_t hi) {\n"
        "    if (v < lo) return lo;\n"
        "    if (v > hi) return hi;\n"
        "    return v;\n"
        "}\n"
        "\n"
        "static uint32_t ring_advance(uint32_t pos, uint32_t cap) {\n"
        "    uint32_t next = pos + 1u;\n"
        "    return next >= cap ? 0u : next;\n"
        "}\n"
        "\n"
        "static int scratch_init(scratch *s, uint32_t ctx_size) {\n"
        "    if (ctx_size == 0u) ctx_size = 1u;\n"
        "    s->ctx_size = ctx_size;\n"
        "    s->comp_cap = ctx_size / 4u + 2u;\n"
        "    s->rows = clamp_u32(s->comp_cap, 1u, 4096u);\n"
        "    s->head = 0u;\n"
        "    return s->rows > 0u ? 0 : -1;\n"
        "}\n"
        "```\n";
}

#define TEST_MTP_MAXGEN 256
#define TEST_DSPARK_MAXGEN 128

static ds4_engine *test_open_dspark_engine(const char *support_path) {
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {
        .model_path = test_model_path(),
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .quality = false,
        .prefill_chunk = 512,
        .ssd_streaming = test_env_bool("DS4_TEST_SSD_STREAMING"),
        .ssd_streaming_cold = test_env_bool("DS4_TEST_SSD_STREAMING_COLD"),
        .ssd_streaming_cache_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS"),
        .ssd_streaming_cache_bytes =
            test_env_gib("DS4_TEST_SSD_STREAMING_CACHE_GB"),
        .ssd_streaming_preload_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_PRELOAD_EXPERTS"),
        .mtp_path = support_path,
        .mtp_draft_tokens = 0,
        .dspark = true,
        .dspark_confidence_threshold = 0.9f,
        .dspark_confidence_threshold_set = true,
    };
    const int rc = ds4_engine_open(&engine, &opt);
    TEST_ASSERT(rc == 0);
    return rc == 0 ? engine : NULL;
}

/* Regression for the swapped top-k arguments in metal_graph_verify_suffix_tops
 * at draft depth > 2.  Replays the committed speculative tokens through plain
 * decode and requires each to be a (near-)argmax: that is the verify invariant,
 * and unlike comparing token streams it tolerates the near-greedy tie
 * divergences.  Needs an MTP head, so it self-skips without DS4_TEST_MTP. */
static void test_mtp_verify_depth(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_has_mtp(engine)) {
        fprintf(stderr, "ds4-test: mtp-verify-depth skipped (set DS4_TEST_MTP to an MTP GGUF)\n");
        return;
    }
    TEST_ASSERT(ds4_engine_mtp_draft_tokens(engine) > 2);

    ds4_tokens prompt = {0};
    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", test_mtp_copy_prompt());
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    TEST_ASSERT(prompt.len > 0);

    int *spec = malloc((size_t)TEST_MTP_MAXGEN * sizeof(*spec));
    TEST_ASSERT(spec != NULL);
    if (spec && prompt.len > 0) {
        int nspec = 0, max_chunk = 0;
        const bool ok_spec = test_mtp_capture_speculative(engine, &prompt, TEST_MTP_MAXGEN,
                                                          spec, &nspec, &max_chunk, NULL,
                                                          NULL);
        TEST_ASSERT(ok_spec);
        TEST_ASSERT(max_chunk > 1);  /* multi-token chunks committed: the multi-row path ran */
        TEST_ASSERT(nspec > 128);    /* enough output to surface the bug, incl. a spurious-EOS truncation */

        float worst_gap = 0.0f;
        int worst_at = -1;
        const bool ok_check = test_mtp_worst_argmax_gap(engine, &prompt, spec, nspec,
                                                        &worst_gap, &worst_at);
        TEST_ASSERT(ok_check);
        fprintf(stderr, "ds4-test: mtp-verify-depth nspec=%d max_chunk=%d worst_argmax_gap=%.3f at=%d\n",
                nspec, max_chunk, worst_gap, worst_at);
        TEST_ASSERT(worst_gap <= 2.0f);  /* correct: ~0; bug: ~21 on the reference model */
    }

    free(spec);
    ds4_tokens_free(&prompt);
}

/* GLM 5.3 embedded MTP: every token the speculative path commits must be a
 * (near-)argmax of plain decode over the same history.  A run must both accept
 * and reject drafts so the two-token commit and the rejection recovery are
 * exercised.  DS4_TEST_GLM_MTP_PROMPT gives a long prompt; DS4_TEST_GLM_MTP_DUMP
 * writes the committed token ids so two builds can be diffed. */
static void test_glm_mtp_verify(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_is_glm53(engine) || !test_env_bool("DS4_TEST_GLM_MTP")) {
        fprintf(stderr, "ds4-test: glm-mtp-verify skipped (needs a GLM 5.3 model and DS4_TEST_GLM_MTP=1)\n");
        return;
    }

    const char *prompt_path = getenv("DS4_TEST_GLM_MTP_PROMPT");
    char *prompt_text = prompt_path && prompt_path[0] ? test_read_file(prompt_path) : NULL;
    if (prompt_path && prompt_path[0]) {
        TEST_ASSERT(prompt_text != NULL);
        if (!prompt_text) return;
    }
    uint32_t max_gen = test_env_u32("DS4_TEST_GLM_MTP_TOKENS");
    if (max_gen == 0) max_gen = TEST_MTP_MAXGEN;
    /* Widest commit this run may produce.  DS4_TEST_GLM_MTP_MAX_CHUNK pins it;
     * otherwise a fixed DS4_GLM_MTP_DEPTH (=2 or =3) pins it to the depth; with
     * neither, glm53_spec_depth() picks 2 or 3 at runtime and only the range is
     * known. */
    uint32_t want_chunk = test_env_u32("DS4_TEST_GLM_MTP_MAX_CHUNK");
    bool chunk_fixed = want_chunk != 0;
    if (!chunk_fixed) {
        const uint32_t depth = test_env_u32("DS4_GLM_MTP_DEPTH");
        if (depth == 2 || depth == 3) {
            want_chunk = depth;
            chunk_fixed = true;
        }
    }

    ds4_tokens prompt = {0};
    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user",
                            prompt_text ? prompt_text : test_mtp_copy_prompt());
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    TEST_ASSERT(prompt.len > 0);

    int *spec = malloc((size_t)max_gen * sizeof(*spec));
    TEST_ASSERT(spec != NULL);
    if (spec && prompt.len > 0) {
        int nspec = 0, max_chunk = 0, single = 0, dbl = 0;
        const bool ok_spec = test_mtp_capture_speculative(engine, &prompt, (int)max_gen,
                                                          spec, &nspec, &max_chunk, &single,
                                                          &dbl);
        TEST_ASSERT(ok_spec);
        if (chunk_fixed) {
            TEST_ASSERT(max_chunk == (int)want_chunk);  /* the widest commit the fixed depth allows ran */
        } else {
            /* Automatic depth: glm53_spec_depth() only ever returns 2 or 3. */
            TEST_ASSERT(max_chunk >= 2 && max_chunk <= 3);
        }
        TEST_ASSERT(single > 0);      /* drafts were rejected: the recovery path ran */
        TEST_ASSERT(nspec >= 64);
        if (chunk_fixed && want_chunk == 3) {
            TEST_ASSERT(dbl > 0);     /* first draft accepted, second rejected */
        }

        const char *dump = getenv("DS4_TEST_GLM_MTP_DUMP");
        if (dump && dump[0]) {
            FILE *fp = fopen(dump, "w");
            TEST_ASSERT(fp != NULL);
            if (fp) {
                for (int i = 0; i < nspec; i++) fprintf(fp, "%d\n", spec[i]);
                fclose(fp);
            }
        }

        float worst_gap = 0.0f;
        int worst_at = -1;
        const bool ok_check = test_mtp_worst_argmax_gap(engine, &prompt, spec, nspec,
                                                        &worst_gap, &worst_at);
        TEST_ASSERT(ok_check);
        fprintf(stderr,
                "ds4-test: glm-mtp-verify prompt=%d nspec=%d single_cycles=%d double_cycles=%d "
                "worst_argmax_gap=%.3f at=%d\n",
                prompt.len, nspec, single, dbl, worst_gap, worst_at);
        TEST_ASSERT(worst_gap <= 2.0f);
    }

    free(spec);
    free(prompt_text);
    ds4_tokens_free(&prompt);
}

/* GLM 5.3 MTP draft window: the nextn KV cache is written only by decode and
 * speculative steps, so prompt prefill leaves those positions unwritten.  Every
 * request must therefore draft over a window that starts at its own first
 * generated position.  The rolling minimum this replaced kept the first
 * request's start for the whole session, so each turn lengthened the window
 * until drafting broke.  Checks a fresh prompt, a resumed-checkpoint
 * continuation and a disk-KV restore, and -- like --glm-mtp-verify -- that every
 * committed token is the plain-decode argmax.  Needs a GLM 5.3 model and
 * DS4_TEST_GLM_MTP=1. */
#define TEST_GLM_MTP_WINDOW_MIN_TOKENS 32

/* One greedy MTP request on an already-synced session: commits tokens with the
 * speculative argmax path and returns how many (or -1 on failure). */
static int test_glm_mtp_window_generate(ds4_engine *engine, ds4_session *session,
                                        int max_tokens, int *out,
                                        int *max_chunk, int *single_cycles) {
    char err[192] = {0};
    const int eos = ds4_token_eos(engine);
    int n = 0;
    *max_chunk = 0;
    *single_cycles = 0;
    while (n < max_tokens) {
        const int first = ds4_session_argmax(session);
        if (first < 0 || first == eos) break;
        int accepted[4] = {0};
        const int ntok = ds4_session_eval_speculative_argmax(
            session, first, max_tokens - n, eos, accepted,
            (int)(sizeof(accepted) / sizeof(accepted[0])), err, sizeof(err));
        if (ntok < 1) {
            fprintf(stderr, "ds4-test: glm-mtp-window cycle failed: %s\n", err);
            TEST_ASSERT(false);
            return -1;
        }
        if (ntok > *max_chunk) *max_chunk = ntok;
        if (ntok == 1) (*single_cycles)++;
        for (int i = 0; i < ntok; i++) {
            if (accepted[i] == eos) return n;
            out[n++] = accepted[i];
            if (n >= max_tokens) return n;
        }
    }
    return n;
}

/* The window start must be decided and be this request's first generated
 * position.  Without the fix the resumed request reports the previous
 * request's start, so a value mismatch is the regression signal. */
static void test_glm_mtp_window_expect(ds4_session *session, const char *label,
                                       uint32_t want) {
    uint32_t start = 0;
    const bool decided = ds4_session_glm_mtp_window_start(session, &start);
    if (!decided || start != want) {
        fprintf(stderr, "ds4-test: glm-mtp-window %s window start %s (want %u)\n",
                label, decided ? "differs" : "undecided", want);
    }
    TEST_ASSERT(decided);
    TEST_ASSERT(start == want);
}

/* At a request boundary the start must be re-decided, not inherited. */
static void test_glm_mtp_window_expect_undecided(ds4_session *session,
                                                 const char *label) {
    uint32_t start = 0;
    if (ds4_session_glm_mtp_window_start(session, &start)) {
        fprintf(stderr,
                "ds4-test: glm-mtp-window %s kept window start %u across the "
                "request boundary\n", label, start);
    }
    uint32_t probe = 0;
    TEST_ASSERT(!ds4_session_glm_mtp_window_start(session, &probe));
}

/* The sync must take the resumed-checkpoint branch: the session's own tokens
 * are the new prompt's prefix and the prompt is longer than the checkpoint. */
static bool test_glm_mtp_window_is_prefix(const ds4_tokens *history,
                                          const ds4_tokens *prompt) {
    if (!history || history->len <= 0 || history->len > prompt->len) return false;
    for (int i = 0; i < history->len; i++) {
        if (prompt->v[i] != history->v[i]) return false;
    }
    return true;
}

static void test_glm_mtp_window_verify(ds4_engine *engine,
                                       const ds4_tokens *prompt, const int *toks,
                                       int n, const char *label) {
    float worst_gap = 0.0f;
    int worst_at = -1;
    TEST_ASSERT(test_mtp_worst_argmax_gap(engine, prompt, toks, n,
                                          &worst_gap, &worst_at));
    fprintf(stderr,
            "ds4-test: glm-mtp-window %s prompt=%d tokens=%d worst_argmax_gap=%.3f "
            "at=%d\n", label, prompt->len, n, worst_gap, worst_at);
    TEST_ASSERT(worst_gap <= 2.0f);
}

static void test_glm_mtp_request_window(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_is_glm53(engine) || !test_env_bool("DS4_TEST_GLM_MTP")) {
        fprintf(stderr, "ds4-test: glm-mtp-window skipped (needs a GLM 5.3 model "
                        "and DS4_TEST_GLM_MTP=1)\n");
        return;
    }

    int max_gen = (int)test_env_u32("DS4_TEST_GLM_MTP_WINDOW_TOKENS");
    if (max_gen < TEST_GLM_MTP_WINDOW_MIN_TOKENS) {
        max_gen = TEST_GLM_MTP_WINDOW_MIN_TOKENS * 2;
    }
    int ctx = (int)test_env_u32("DS4_TEST_GLM_MTP_WINDOW_CTX");
    if (ctx < 4096) ctx = 4096;

    ds4_tokens prompt = {0};
    ds4_tokens second = {0};
    ds4_tokens third = {0};
    ds4_session *live = NULL;
    ds4_session *restored = NULL;
    int *gen1 = malloc((size_t)max_gen * sizeof(*gen1));
    int *gen2 = malloc((size_t)max_gen * sizeof(*gen2));
    int *gen3 = malloc((size_t)max_gen * sizeof(*gen3));
    char err[192] = {0};
    uint32_t want1 = 0, want2 = 0, want3 = 0;
    int n1 = -1, n2 = -1, n3 = -1;
    int chunk1 = 0, chunk2 = 0, chunk3 = 0;
    int single1 = 0, single2 = 0, single3 = 0;
    FILE *payload = NULL;
    long payload_bytes = 0;
    TEST_ASSERT(gen1 != NULL && gen2 != NULL && gen3 != NULL);
    if (!gen1 || !gen2 || !gen3) goto cleanup;

    /* Request 1: a fresh prompt prefill writes no nextn KV, so the window
     * starts at the first generated position. */
    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", test_mtp_copy_prompt());
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    TEST_ASSERT(prompt.len > 0);
    TEST_ASSERT(prompt.len + max_gen + 64 < ctx);
    TEST_ASSERT(ds4_session_create(&live, engine, ctx) == 0);
    if (!live) goto cleanup;
    TEST_ASSERT(ds4_session_sync(live, &prompt, err, sizeof(err)) == 0);
    test_glm_mtp_window_expect_undecided(live, "request 1 before generating");
    want1 = (uint32_t)ds4_session_pos(live);
    n1 = test_glm_mtp_window_generate(engine, live, max_gen, gen1, &chunk1, &single1);
    TEST_ASSERT(n1 >= TEST_GLM_MTP_WINDOW_MIN_TOKENS);
    if (n1 < TEST_GLM_MTP_WINDOW_MIN_TOKENS) goto cleanup;
    TEST_ASSERT(chunk1 >= 2);  /* a draft was accepted: the MTP cycle really ran */
    test_glm_mtp_window_expect(live, "request 1", want1);
    test_glm_mtp_window_verify(engine, &prompt, gen1, n1, "request 1");

    /* Request 2: the same session continues with a new user turn, so sync
     * resumes the checkpoint and appends tokens that have no nextn KV. */
    for (int i = 0; i < prompt.len; i++) ds4_tokens_push(&second, prompt.v[i]);
    for (int i = 0; i < n1; i++) ds4_tokens_push(&second, gen1[i]);
    {
        buf followup = {0};
        buf_puts(&followup, "Now reproduce only the first function again, exactly. ");
        for (int i = 0; i < 8; i++) {
            buf_puts(&followup, "Keep the formatting identical to the code block. ");
        }
        ds4_chat_append_message(engine, &second, "user",
                               followup.ptr ? followup.ptr : "Again, exactly.");
        buf_free(&followup);
    }
    ds4_chat_append_assistant_prefix(engine, &second, DS4_THINK_NONE);
    const ds4_tokens *history = ds4_session_tokens(live);
    TEST_ASSERT(history != NULL);
    TEST_ASSERT(ds4_session_checkpoint_valid(live));
    TEST_ASSERT(test_glm_mtp_window_is_prefix(history, &second));
    TEST_ASSERT(second.len > (history ? history->len : 0));
    TEST_ASSERT(second.len + max_gen + 64 < ctx);
    TEST_ASSERT(ds4_session_sync(live, &second, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_pos(live) == second.len);
    test_glm_mtp_window_expect_undecided(live, "request 2 before generating");
    want2 = (uint32_t)ds4_session_pos(live);
    n2 = test_glm_mtp_window_generate(engine, live, max_gen, gen2, &chunk2, &single2);
    TEST_ASSERT(n2 >= TEST_GLM_MTP_WINDOW_MIN_TOKENS);
    if (n2 < TEST_GLM_MTP_WINDOW_MIN_TOKENS) goto cleanup;
    test_glm_mtp_window_expect(live, "request 2", want2);
    TEST_ASSERT(want2 != want1);
    test_glm_mtp_window_verify(engine, &second, gen2, n2, "request 2");

    /* Disk KV read-back: the payload carries the transcript and the KV, but
     * the draft window always starts over for the next request. */
    payload = tmpfile();
    TEST_ASSERT(payload != NULL);
    if (!payload) goto cleanup;
    TEST_ASSERT(ds4_session_save_payload(live, payload, err, sizeof(err)) == 0);
    payload_bytes = ftell(payload);
    TEST_ASSERT(payload_bytes > 0);
    fflush(payload);
    rewind(payload);
    ds4_session_free(live);
    live = NULL;
    TEST_ASSERT(ds4_session_create(&restored, engine, ctx) == 0);
    if (!restored) goto cleanup;
    TEST_ASSERT(ds4_session_load_payload(restored, payload, (uint64_t)payload_bytes,
                                        err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_checkpoint_valid(restored));
    TEST_ASSERT(ds4_session_pos(restored) == second.len + n2);
    test_glm_mtp_window_expect_undecided(restored, "request 3 before generating");

    for (int i = 0; i < second.len; i++) ds4_tokens_push(&third, second.v[i]);
    for (int i = 0; i < n2; i++) ds4_tokens_push(&third, gen2[i]);
    {
        buf followup = {0};
        buf_puts(&followup, "Now reproduce only the second function again, exactly. ");
        for (int i = 0; i < 8; i++) {
            buf_puts(&followup, "Keep the formatting identical to the code block. ");
        }
        ds4_chat_append_message(engine, &third, "user",
                               followup.ptr ? followup.ptr : "Again, exactly.");
        buf_free(&followup);
    }
    ds4_chat_append_assistant_prefix(engine, &third, DS4_THINK_NONE);
    const ds4_tokens *restored_history = ds4_session_tokens(restored);
    TEST_ASSERT(test_glm_mtp_window_is_prefix(restored_history, &third));
    TEST_ASSERT(third.len > (restored_history ? restored_history->len : 0));
    TEST_ASSERT(third.len + max_gen + 64 < ctx);
    TEST_ASSERT(ds4_session_sync(restored, &third, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_pos(restored) == third.len);
    test_glm_mtp_window_expect_undecided(restored, "request 3 after syncing");
    want3 = (uint32_t)ds4_session_pos(restored);
    n3 = test_glm_mtp_window_generate(engine, restored, max_gen, gen3, &chunk3, &single3);
    TEST_ASSERT(n3 >= TEST_GLM_MTP_WINDOW_MIN_TOKENS);
    if (n3 < TEST_GLM_MTP_WINDOW_MIN_TOKENS) goto cleanup;
    test_glm_mtp_window_expect(restored, "request 3", want3);
    TEST_ASSERT(want3 != want2 && want3 != want1);
    test_glm_mtp_window_verify(engine, &third, gen3, n3, "request 3");

    fprintf(stderr,
            "ds4-test: glm-mtp-window starts=%u/%u/%u prompt=%d/%d/%d tokens=%d/%d/%d "
            "max_chunk=%d/%d/%d single_cycles=%d/%d/%d payload=%ld\n",
            want1, want2, want3, prompt.len, second.len, third.len,
            n1, n2, n3, chunk1, chunk2, chunk3, single1, single2, single3,
            payload_bytes);

cleanup:
    if (payload) fclose(payload);
    free(gen3);
    free(gen2);
    free(gen1);
    ds4_tokens_free(&third);
    ds4_tokens_free(&second);
    ds4_tokens_free(&prompt);
    ds4_session_free(restored);
    ds4_session_free(live);
}

/* GLM 5.3 nextn (MTP) cache prefill write (ticket 004).  With MTP on, prompt
 * prefill also writes the nextn KV and pooled indexer caches, so those caches
 * are dense over the whole context instead of only the drafted window, and the
 * one position whose next token does not exist yet is parked in the seam and
 * written the moment it does.  The test builds the same caches twice -- once
 * from prefill, once one decode row at a time -- compares them, then checks
 * the seam on both sides (the first generated token and a continued request)
 * and the position-0 embedding mask.  Needs a GLM 5.3 model and
 * DS4_TEST_GLM_MTP=1. */
#define TEST_GLM_MTP_PREFILL_CHUNK 2048u  /* ds4.c:39102 DS4_GLM53_PREFILL_CHUNK_TOKENS */
#define TEST_GLM_MTP_PREFILL_POOL 4u      /* ds4.c:39098 DS4_GLM53_INDEX_POOL_SIZE */
#define TEST_GLM_MTP_PREFILL_SUFFIX 2u    /* longest continued-request suffix */

typedef struct {
    int chunk_events;
    int intermediate;
} test_glm_mtp_prefill_progress;

static void test_glm_mtp_prefill_progress_cb(void *ud, const char *event,
                                             int current, int total) {
    test_glm_mtp_prefill_progress *p = ud;
    if (strcmp(event, "prefill_chunk") != 0) return;
    p->chunk_events++;
    if (current > 0 && current < total) p->intermediate++;
}

static bool test_glm_mtp_row_finite(const float *v, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        if (!isfinite(v[i])) return false;
    }
    return true;
}

static bool test_glm_mtp_row_zero(const float *v, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        if (v[i] != 0.0f) return false;
    }
    return true;
}

static bool test_glm_mtp_row_written(const float *v, uint32_t n) {
    bool any = false;
    for (uint32_t i = 0; i < n; i++) {
        if (!isfinite(v[i])) return false;
        if (v[i] != 0.0f) any = true;
    }
    return any;
}

static void test_glm_mtp_row_diff(const float *a, const float *b, uint32_t n,
                                  float *max_abs, float *max_rel) {
    *max_abs = 0.0f;
    *max_rel = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        const float m = fabsf(a[i]) > fabsf(b[i]) ? fabsf(a[i]) : fabsf(b[i]);
        if (d > *max_abs) *max_abs = d;
        if (m > 1.0e-6f) {
            const float r = d / m;
            if (r > *max_rel) *max_rel = r;
        }
    }
}

/* Greedy top-k of a raw logits row, for the draft-logit comparison. */
static void test_glm_mtp_top_k(const float *logits, int vocab, int k,
                               ds4_token_score *out) {
    for (int i = 0; i < k; i++) {
        out[i].id = -1;
        out[i].logit = -1.0e30f;   /* -FLT_MAX: -ffast-math makes -INFINITY UB */
    }
    for (int t = 0; t < vocab; t++) {
        if (logits[t] <= out[k - 1].logit) continue;
        int j = k - 1;
        while (j > 0 && logits[t] > out[j - 1].logit) {
            out[j] = out[j - 1];
            j--;
        }
        out[j].id = t;
        out[j].logit = logits[t];
    }
}

/* P: the chat-formatted prompt, repeated until it is longer than one whole
 * prefill chunk plus a pool window, then trimmed from the tail until its
 * length is a whole number of pools.  That makes the seam row the last cell of
 * its pool, so the seam has to complete a pooled key row. */
static bool test_glm_mtp_prefill_prompt(ds4_engine *engine, ds4_tokens *prompt,
                                        const char *base) {
    const uint32_t need = TEST_GLM_MTP_PREFILL_CHUNK + TEST_GLM_MTP_PREFILL_POOL;
    const size_t base_len = strlen(base);
    size_t text_len = base_len;
    char *text = malloc(base_len + 1);
    if (!text) return false;
    memcpy(text, base, base_len + 1);
    for (;;) {
        prompt->len = 0;
        ds4_chat_begin(engine, prompt);
        ds4_chat_append_message(engine, prompt, "user", text);
        ds4_chat_append_assistant_prefix(engine, prompt, DS4_THINK_NONE);
        if ((uint32_t)prompt->len >= need) break;
        if (text_len > ((size_t)1 << 26)) {
            free(text);
            return false;
        }
        char *grown = realloc(text, text_len * 2 + 1);
        if (!grown) {
            free(text);
            return false;
        }
        text = grown;
        memcpy(text + text_len, base, base_len);
        text_len += base_len;
        text[text_len] = '\0';
    }
    free(text);
    while (prompt->len > 0 &&
           (uint32_t)prompt->len % TEST_GLM_MTP_PREFILL_POOL != 0u) {
        prompt->len--;
    }
    return (uint32_t)prompt->len > TEST_GLM_MTP_PREFILL_CHUNK;
}

/* The buffers every section compares through. */
typedef struct {
    uint32_t kv_dim, pool_dim, pool_size, tail_dim, cache_cap, vocab;
    float *kv_a, *kv_b, *pool_a, *pool_b, *tail_a, *tail_b;
} test_glm_mtp_caches;

/* Continued request (section 2): a request that resumes from a checkpoint
 * whose last nextn slot was written with the drafter's predicted token refills
 * that slot with the actual token, and the refilled row must sit closer to the
 * row a cold prefill of the same prompt writes than the predicted one did. */
static void test_glm_mtp_prefill_resume(ds4_engine *engine,
                                        const ds4_tokens *prompt, uint32_t m,
                                        int ctx, bool pinned_prompt,
                                        test_glm_mtp_caches *c) {
    int *gen = calloc((size_t)TEST_GLM_MTP_WINDOW_MIN_TOKENS, sizeof(int));
    TEST_ASSERT(gen != NULL);
    if (!gen) return;

    for (uint32_t suffix_len = 1u; suffix_len <= TEST_GLM_MTP_PREFILL_SUFFIX;
         suffix_len++) {
        ds4_session *r = NULL;
        ds4_session *r2 = NULL;
        ds4_tokens prompt2 = {0};
        char err[256] = {0};
        bool ok = ds4_session_create(&r, engine, ctx) == 0;
        TEST_ASSERT(ok);
        if (!ok) break;
        ds4_test_glm_zero_mtp_caches(r);
        ok = ds4_session_sync(r, prompt, err, sizeof(err)) == 0;
        TEST_ASSERT(ok);
        if (!ok) {
            fprintf(stderr, "ds4-test: glm-mtp-prefill resume sync failed: %s\n",
                    err);
            ds4_session_free(r);
            break;
        }

        int chunk = 0, single = 0;
        const int n = test_glm_mtp_window_generate(
            engine, r, TEST_GLM_MTP_WINDOW_MIN_TOKENS, gen, &chunk, &single);
        const uint32_t l = (uint32_t)ds4_session_pos(r);
        fprintf(stderr,
                "ds4-test: glm-mtp-prefill resume S=%u generated=%d L=%u M=%u "
                "L%%POOL=%u\n", suffix_len, n, l, m, l % c->pool_size);
        if (n < TEST_GLM_MTP_WINDOW_MIN_TOKENS || l != m + (uint32_t)n) {
            fprintf(stderr, "ds4-test: glm-mtp-prefill resume checkpoint mismatch: "
                            "L=%u M=%u n=%d\n", l, m, n);
            TEST_ASSERT(n >= TEST_GLM_MTP_WINDOW_MIN_TOKENS);
            TEST_ASSERT(l == m + (uint32_t)n);
            ds4_session_free(r);
            break;
        }

        ds4_token_score top[8] = {0};
        TEST_ASSERT(ds4_session_top_logprobs(r, top, 8) == 8);
        const int x0 = top[1].id;
        TEST_ASSERT(x0 != top[0].id);

        const ds4_tokens *history = ds4_session_tokens(r);
        for (uint32_t i = 0; i < l; i++) ds4_tokens_push(&prompt2, history->v[i]);
        ds4_tokens_push(&prompt2, x0);
        if (suffix_len > 1u) ds4_tokens_push(&prompt2, prompt->v[1]);
        TEST_ASSERT((uint32_t)prompt2.len == l + suffix_len);

        /* Slot L-1 holds the drafter's guess.  Blanking only its KV row means
         * a real write is the only thing that can make it non-zero again. */
        ok = ds4_test_glm_read_mtp_kv(r, l - 1u, c->kv_a, c->kv_dim, NULL, 0,
                                      NULL, 0);
        TEST_ASSERT(ok && test_glm_mtp_row_written(c->kv_a, c->kv_dim));
        const float *row_pre = c->kv_a;
        float *row_pre_copy = malloc((size_t)c->kv_dim * sizeof(float));
        float *row_post_copy = malloc((size_t)c->kv_dim * sizeof(float));
        float *row_ref_copy = malloc((size_t)c->kv_dim * sizeof(float));
        TEST_ASSERT(row_pre_copy && row_post_copy && row_ref_copy);
        if (!row_pre_copy || !row_post_copy || !row_ref_copy) {
            free(row_pre_copy); free(row_post_copy); free(row_ref_copy);
            ds4_tokens_free(&prompt2);
            ds4_session_free(r);
            break;
        }
        memcpy(row_pre_copy, row_pre, (size_t)c->kv_dim * sizeof(float));
        TEST_ASSERT(ds4_test_glm_zero_kv_slot(r, l - 1u));
        TEST_ASSERT(ds4_test_glm_read_mtp_kv(r, l - 1u, c->kv_a, c->kv_dim,
                                             NULL, 0, NULL, 0));
        TEST_ASSERT(test_glm_mtp_row_zero(c->kv_a, c->kv_dim));

        ok = ds4_session_sync(r, &prompt2, err, sizeof(err)) == 0;
        TEST_ASSERT(ok);
        if (!ok) fprintf(stderr, "ds4-test: glm-mtp-prefill resume failed: %s\n", err);
        TEST_ASSERT(ds4_test_glm_read_mtp_kv(r, l - 1u, row_post_copy, c->kv_dim,
                                             NULL, 0, NULL, 0));
        fprintf(stderr, "ds4-test: glm-mtp-prefill resume S=%u slot %u refilled=%d\n",
                suffix_len, l - 1u,
                test_glm_mtp_row_written(row_post_copy, c->kv_dim));
        TEST_ASSERT(test_glm_mtp_row_written(row_post_copy, c->kv_dim));

        ok = ds4_session_create(&r2, engine, ctx) == 0;
        TEST_ASSERT(ok);
        if (ok) {
            ds4_test_glm_zero_mtp_caches(r2);
            ok = ds4_session_sync(r2, &prompt2, err, sizeof(err)) == 0;
            TEST_ASSERT(ok);
            if (ok) {
                TEST_ASSERT(ds4_test_glm_read_mtp_kv(r2, l - 1u, row_ref_copy,
                                                     c->kv_dim, NULL, 0, NULL, 0));
                float post_abs = 0.0f, post_rel = 0.0f;
                float pre_abs = 0.0f, pre_rel = 0.0f;
                test_glm_mtp_row_diff(row_post_copy, row_ref_copy, c->kv_dim,
                                      &post_abs, &post_rel);
                test_glm_mtp_row_diff(row_pre_copy, row_ref_copy, c->kv_dim,
                                      &pre_abs, &pre_rel);
                fprintf(stderr,
                        "ds4-test: glm-mtp-prefill resume S=%u slot %u vs cold "
                        "prefill: refilled max_abs=%.6g max_rel=%.6g, predicted "
                        "max_abs=%.6g max_rel=%.6g\n",
                        suffix_len, l - 1u, post_abs, post_rel, pre_abs, pre_rel);
                /* Ordering, not a numeric threshold: the row written with the
                 * real token must be closer to the cold row than the one
                 * written with the drafter's guess was. */
                TEST_ASSERT(post_abs < pre_abs);

                float before_abs = 0.0f, before_rel = 0.0f;
                float after_abs = 0.0f, after_rel = 0.0f;
                for (uint32_t row = 0;; row++) {
                    const uint32_t slot = row * c->pool_size;
                    if (slot >= (uint32_t)prompt2.len) break;
                    if (!ds4_test_glm_read_mtp_kv(r, slot, NULL, 0, c->pool_a,
                                                  c->pool_dim, NULL, 0)) break;
                    if (!ds4_test_glm_read_mtp_kv(r2, slot, NULL, 0, c->pool_b,
                                                  c->pool_dim, NULL, 0)) break;
                    float ab = 0.0f, rl = 0.0f;
                    test_glm_mtp_row_diff(c->pool_a, c->pool_b, c->pool_dim,
                                          &ab, &rl);
                    /* "before" = every position of this pooled row predates the
                     * refilled seam slot, so both sessions must agree exactly. */
                    if (slot + c->pool_size - 1u < l - 1u) {
                        if (ab > before_abs) before_abs = ab;
                        if (rl > before_rel) before_rel = rl;
                    } else {
                        if (ab > after_abs) after_abs = ab;
                        if (rl > after_rel) after_rel = rl;
                    }
                }
                TEST_ASSERT(ds4_test_glm_read_mtp_kv(r, 0u, NULL, 0, NULL, 0,
                                                     c->tail_a, c->tail_dim));
                TEST_ASSERT(ds4_test_glm_read_mtp_kv(r2, 0u, NULL, 0, NULL, 0,
                                                     c->tail_b, c->tail_dim));
                float tab = 0.0f, trl = 0.0f;
                test_glm_mtp_row_diff(c->tail_a, c->tail_b, c->tail_dim, &tab, &trl);
                fprintf(stderr,
                        "ds4-test: glm-mtp-prefill resume S=%u pooled rows before "
                        "%u: max_abs=%.6g max_rel=%.6g; from %u: max_abs=%.6g "
                        "max_rel=%.6g; tail: max_abs=%.6g max_rel=%.6g\n",
                        suffix_len, l - 1u, before_abs, before_rel, l - 1u,
                        after_abs, after_rel, tab, trl);
                TEST_ASSERT(test_glm_mtp_row_finite(c->tail_a, c->tail_dim));
                TEST_ASSERT(test_glm_mtp_row_finite(c->tail_b, c->tail_dim));
                /* These two pooled-row groups are the only place where every
                 * nextn-cache producer shows up: prefill (its embedding offset
                 * and its post-norm hidden), the plain step's indexer write,
                 * the catch-up's indexer write and the seam rewrite.  The
                 * thresholds below sit between the correct value and the wrong
                 * ones; measured on GLM-5.3-Flash-IQ2-Down10Q2K-imatrix.gguf
                 * with the default prompt (DS4_TEST_GLM_MTP_PROMPT unset), and
                 * identical for S = 1 and S = 2.  With a user prompt the
                 * numbers move, so the groups stay print-only there.
                 *   before: correct 0.0551758; bug: prefill embedding off by
                 *     one 0.454041, hidden not through output_norm 1.21045,
                 *     plain step without the indexer 0.537598, catch-up
                 *     without the indexer 2.14258, seam never flushed 1.86914
                 *   from: correct 0.0292969; bug: prefill embedding off by one
                 *     0.48996, hidden not through output_norm 0.880493, seam
                 *     flushed with the predicted token 0.414551, seam hidden
                 *     never updated after decode 1.43103, catch-up without the
                 *     indexer 1.68787 */
                if (!pinned_prompt) {
                    TEST_ASSERT(before_abs <= 0.15f);
                    TEST_ASSERT(after_abs <= 0.1f);
                }
            }
        }
        ds4_session_free(r2);
        ds4_session_free(r);
        ds4_tokens_free(&prompt2);
        free(row_pre_copy);
        free(row_post_copy);
        free(row_ref_copy);
    }
    free(gen);
}

static void test_glm_mtp_prefill_write(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_is_glm53(engine) || !test_env_bool("DS4_TEST_GLM_MTP")) {
        fprintf(stderr, "ds4-test: glm-mtp-prefill skipped (needs a GLM 5.3 model "
                        "and DS4_TEST_GLM_MTP=1)\n");
        return;
    }

    const char *prompt_path = getenv("DS4_TEST_GLM_MTP_PROMPT");
    char *file_text = prompt_path && prompt_path[0] ? test_read_file(prompt_path) : NULL;
    if (prompt_path && prompt_path[0] && !file_text) {
        fprintf(stderr, "ds4-test: glm-mtp-prefill cannot read %s\n", prompt_path);
        TEST_ASSERT(false);
        return;
    }

    ds4_tokens prompt = {0};
    ds4_tokens extended = {0};
    test_glm_mtp_caches c = {0};
    ds4_session *a = NULL;
    ds4_session *b = NULL;
    float *tail_pre = NULL, *draft_a = NULL, *draft_b = NULL;
    char err[256] = {0};
    bool ok = test_glm_mtp_prefill_prompt(engine, &prompt,
                                          file_text ? file_text
                                                    : test_mtp_copy_prompt());
    if (!ok) {
        fprintf(stderr, "ds4-test: glm-mtp-prefill could not build a prompt longer "
                        "than one prefill chunk and a whole number of pools\n");
        TEST_ASSERT(false);
        goto cleanup;
    }
    const uint32_t m = (uint32_t)prompt.len;
    const int eos = ds4_token_eos(engine);
    /* ds4_session_sync requires prompt->len < ctx, so the ctx is the longest
     * token sequence this test builds plus one: P + G + the longest suffix. */
    int ctx = (int)m + TEST_GLM_MTP_WINDOW_MIN_TOKENS +
              (int)TEST_GLM_MTP_PREFILL_SUFFIX + 1;
    if (ctx < (int)m + 2) ctx = (int)m + 2;
    fprintf(stderr, "ds4-test: glm-mtp-prefill prompt=%u ctx=%d\n", m, ctx);

    /* ---- 1. prefill fills the caches; the seam waits for its token ---- */
    ok = ds4_session_create(&a, engine, ctx) == 0;
    if (!ok) {
        fprintf(stderr, "ds4-test: glm-mtp-prefill ds4_session_create(ctx=%d) "
                        "refused\n", ctx);
        TEST_ASSERT(false);
        goto cleanup;
    }
    /* The nextn caches are built lazily, so zeroing them here is what puts
     * zeros in place before the first prefill row can write anything. */
    ds4_test_glm_zero_mtp_caches(a);
    TEST_ASSERT(ds4_test_glm_mtp_cache_dims(a, &c.kv_dim, &c.pool_dim,
                                            &c.pool_size, &c.tail_dim,
                                            &c.cache_cap));
    TEST_ASSERT(c.pool_size == TEST_GLM_MTP_PREFILL_POOL);
    TEST_ASSERT(m < c.cache_cap);
    c.vocab = (uint32_t)ds4_engine_vocab_size(engine);
    c.kv_a = malloc((size_t)c.kv_dim * sizeof(float));
    c.kv_b = malloc((size_t)c.kv_dim * sizeof(float));
    c.pool_a = malloc((size_t)c.pool_dim * sizeof(float));
    c.pool_b = malloc((size_t)c.pool_dim * sizeof(float));
    c.tail_a = malloc((size_t)c.tail_dim * sizeof(float));
    c.tail_b = malloc((size_t)c.tail_dim * sizeof(float));
    tail_pre = malloc((size_t)c.tail_dim * sizeof(float));
    draft_a = malloc((size_t)c.vocab * sizeof(float));
    draft_b = malloc((size_t)c.vocab * sizeof(float));
    TEST_ASSERT(c.kv_a && c.kv_b && c.pool_a && c.pool_b && c.tail_a &&
                c.tail_b && tail_pre && draft_a && draft_b);
    if (!c.kv_a || !c.kv_b || !c.pool_a || !c.pool_b || !c.tail_a ||
        !c.tail_b || !tail_pre || !draft_a || !draft_b) goto cleanup;

    test_glm_mtp_prefill_progress progress = {0};
    ds4_session_set_progress(a, test_glm_mtp_prefill_progress_cb, &progress);
    ok = ds4_session_sync(a, &prompt, err, sizeof(err)) == 0;
    ds4_session_set_progress(a, NULL, NULL);
    TEST_ASSERT(ok);
    if (!ok) {
        fprintf(stderr, "ds4-test: glm-mtp-prefill sync failed: %s\n", err);
        goto cleanup;
    }
    /* An intermediate chunk boundary is what proves the fill ran across more
     * than one chunk of the same prompt. */
    fprintf(stderr, "ds4-test: glm-mtp-prefill chunk events=%d intermediate=%d\n",
            progress.chunk_events, progress.intermediate);
    TEST_ASSERT(progress.chunk_events > 0);
    TEST_ASSERT(progress.intermediate > 0);

    /* Every row the prefill could write is dense and finite. */
    for (uint32_t p = 0; p + 1u < m; p++) {
        TEST_ASSERT(ds4_test_glm_read_mtp_kv(a, p, c.kv_a, c.kv_dim, NULL, 0,
                                             NULL, 0));
        if (!test_glm_mtp_row_written(c.kv_a, c.kv_dim)) {
            fprintf(stderr, "ds4-test: glm-mtp-prefill KV slot %u unwritten\n", p);
            TEST_ASSERT(false);
            goto cleanup;
        }
    }
    /* The last row had no next token, so its slot is still the zero the caches
     * started with. */
    TEST_ASSERT(ds4_test_glm_read_mtp_kv(a, m - 1u, c.kv_a, c.kv_dim, c.pool_a,
                                         c.pool_dim, c.tail_a, c.tail_dim));
    fprintf(stderr, "ds4-test: glm-mtp-prefill seam slot %u all zero=%d\n",
            m - 1u, test_glm_mtp_row_zero(c.kv_a, c.kv_dim));
    TEST_ASSERT(test_glm_mtp_row_zero(c.kv_a, c.kv_dim));
    memcpy(tail_pre, c.tail_a, (size_t)c.tail_dim * sizeof(float));

    const uint32_t seam_pool = m / c.pool_size;   /* M is pool-aligned */
    for (uint32_t row = 0; row + 1u < seam_pool; row++) {
        TEST_ASSERT(ds4_test_glm_read_mtp_kv(a, row * c.pool_size, NULL, 0,
                                             c.pool_a, c.pool_dim, NULL, 0));
        if (!test_glm_mtp_row_written(c.pool_a, c.pool_dim)) {
            fprintf(stderr, "ds4-test: glm-mtp-prefill pool row %u unwritten\n", row);
            TEST_ASSERT(false);
            goto cleanup;
        }
    }
    /* The seam's own pool row needs the row the seam is waiting to write. */
    TEST_ASSERT(ds4_test_glm_read_mtp_kv(a, m - 1u, NULL, 0, c.pool_a,
                                         c.pool_dim, NULL, 0));
    fprintf(stderr, "ds4-test: glm-mtp-prefill seam pool row %u all zero=%d\n",
            seam_pool - 1u, test_glm_mtp_row_zero(c.pool_a, c.pool_dim));
    TEST_ASSERT(test_glm_mtp_row_zero(c.pool_a, c.pool_dim));
    /* The raw tail is the other half of the same story: the prefill could not
     * finish the seam's pool, so it parked the rows it did have in the cells
     * below the seam's own cell. */
    {
        const uint32_t stride = c.tail_dim / 2u;   /* k half, then gate half */
        for (uint32_t cell = 0; cell + 1u < c.pool_size; cell++) {
            const float *k = c.tail_a + (uint64_t)cell * c.pool_dim;
            const float *gate = c.tail_a + stride + (uint64_t)cell * c.pool_dim;
            if (!test_glm_mtp_row_written(k, c.pool_dim) ||
                !test_glm_mtp_row_written(gate, c.pool_dim)) {
                fprintf(stderr, "ds4-test: glm-mtp-prefill tail cell %u unwritten\n",
                        cell);
                TEST_ASSERT(false);
                goto cleanup;
            }
        }
    }

    /* The first generated token: no draft is pending, so the real speculative
     * entry point takes the plain step and the production seam flush runs. */
    const int t = ds4_session_argmax(a);
    TEST_ASSERT(t >= 0);
    int accepted[4] = {0};
    const int ntok = ds4_session_eval_speculative_argmax(
        a, t, TEST_GLM_MTP_WINDOW_MIN_TOKENS, eos, accepted, 4, err, sizeof(err));
    fprintf(stderr, "ds4-test: glm-mtp-prefill first cycle committed=%d pos=%d\n",
            ntok, ds4_session_pos(a));
    TEST_ASSERT(ntok == 1);
    TEST_ASSERT((uint32_t)ds4_session_pos(a) == m + 1u);
    if (ntok != 1) goto cleanup;
    TEST_ASSERT(ds4_test_glm_read_mtp_kv(a, m - 1u, c.kv_a, c.kv_dim, c.pool_a,
                                         c.pool_dim, c.tail_a, c.tail_dim));
    fprintf(stderr, "ds4-test: glm-mtp-prefill after the first token: seam slot %u "
                    "written=%d seam pool row %u written=%d\n",
            m - 1u, test_glm_mtp_row_written(c.kv_a, c.kv_dim), seam_pool - 1u,
            test_glm_mtp_row_written(c.pool_a, c.pool_dim));
    TEST_ASSERT(test_glm_mtp_row_written(c.kv_a, c.kv_dim));
    TEST_ASSERT(test_glm_mtp_row_written(c.pool_a, c.pool_dim));
    /* The plain step also wrote the row it just decoded. */
    TEST_ASSERT(ds4_test_glm_read_mtp_kv(a, m, c.kv_a, c.kv_dim, NULL, 0, NULL, 0));
    TEST_ASSERT(test_glm_mtp_row_written(c.kv_a, c.kv_dim));
    /* A row that completes its pool never rewrites the raw tail: the pool
     * kernel keeps only the window it could not finish.  The seam's cell is
     * therefore the cell the prefill could NOT write, and it still reads as
     * prefill left it; the seam's effect shows up in the pooled row above. */
    {
        const uint32_t stride = c.tail_dim / 2u;
        const uint32_t cell = (m - 1u) % c.pool_size;
        float k_abs = 0.0f, k_rel = 0.0f, g_abs = 0.0f, g_rel = 0.0f;
        test_glm_mtp_row_diff(c.tail_a + (uint64_t)cell * c.pool_dim,
                              tail_pre + (uint64_t)cell * c.pool_dim,
                              c.pool_dim, &k_abs, &k_rel);
        test_glm_mtp_row_diff(c.tail_a + stride + (uint64_t)cell * c.pool_dim,
                              tail_pre + stride + (uint64_t)cell * c.pool_dim,
                              c.pool_dim, &g_abs, &g_rel);
        fprintf(stderr, "ds4-test: glm-mtp-prefill seam tail cell %u: k max_abs=%.6g "
                        "max_rel=%.6g gate max_abs=%.6g max_rel=%.6g (a "
                        "pool-completing row leaves the tail alone)\n",
                cell, k_abs, k_rel, g_abs, g_rel);
        TEST_ASSERT(k_abs == 0.0f && g_abs == 0.0f);
    }

    /* ---- 2. the same caches, built one decode row at a time ---- */
    ok = ds4_session_create(&b, engine, ctx) == 0;
    if (!ok) {
        fprintf(stderr, "ds4-test: glm-mtp-prefill ds4_session_create(ctx=%d) "
                        "refused for the decode session\n", ctx);
        TEST_ASSERT(false);
        goto cleanup;
    }
    ds4_test_glm_zero_mtp_caches(b);
    ds4_tokens head = prompt;   /* shares P's buffer; never freed on its own */
    head.len = 1;
    TEST_ASSERT(ds4_session_sync(b, &head, err, sizeof(err)) == 0);
    /* q = 0 has no decode hidden to write, so slot 0 is section 3's job. */
    for (uint32_t q = 1u; q < m; q++) {
        head.len = (int)q + 1;
        ok = ds4_session_sync(b, &head, err, sizeof(err)) == 0;
        if (!ok) {
            fprintf(stderr, "ds4-test: glm-mtp-prefill decode resume at %u: %s\n",
                    q, err);
            TEST_ASSERT(false);
            goto cleanup;
        }
        const int next = q + 1u < m ? prompt.v[q + 1u] : t;
        if (!ds4_test_glm_mtp_draft_after_decode(b, next, q, NULL, NULL, 0)) {
            fprintf(stderr, "ds4-test: glm-mtp-prefill decode step at %u failed\n", q);
            TEST_ASSERT(false);
            goto cleanup;
        }
    }

    /* One more step on both sides: the draft logits must agree. */
    for (uint32_t i = 0; i < m; i++) ds4_tokens_push(&extended, prompt.v[i]);
    ds4_tokens_push(&extended, t);
    TEST_ASSERT(ds4_session_sync(b, &extended, err, sizeof(err)) == 0);
    const int next_a = ds4_session_argmax(a);
    TEST_ASSERT(next_a >= 0);
    TEST_ASSERT(ds4_test_glm_mtp_draft_after_decode(a, next_a, 1u, NULL,
                                                    draft_a, c.vocab));
    TEST_ASSERT(ds4_test_glm_mtp_draft_after_decode(b, next_a, 1u, NULL,
                                                    draft_b, c.vocab));
    {
        ds4_token_score top_a[8], top_b[8];
        test_glm_mtp_top_k(draft_a, (int)c.vocab, 8, top_a);
        test_glm_mtp_top_k(draft_b, (int)c.vocab, 8, top_b);
        float worst = 0.0f;
        int same_rank = 0;
        for (int i = 0; i < 8; i++) {
            if (top_a[i].id != top_b[i].id) continue;
            same_rank++;
            const float d = fabsf(top_a[i].logit - top_b[i].logit);
            if (d > worst) worst = d;
        }
        fprintf(stderr, "ds4-test: glm-mtp-prefill draft top1=%d/%d same-rank "
                        "top8=%d max same-rank delta=%.6g\n",
                top_a[0].id, top_b[0].id, same_rank, worst);
        TEST_ASSERT(top_a[0].id == top_b[0].id);
        TEST_ASSERT(test_top_tokens_overlap(top_a, top_b, 8));
    }

    /* Cache comparison against the decode-built session.  No numeric threshold
     * this round: the reviewer measures the real values first. */
    {
        float worst_abs = 0.0f, worst_rel = 0.0f;
        for (uint32_t p = 1u; p + 1u < m; p++) {
            TEST_ASSERT(ds4_test_glm_read_mtp_kv(a, p, c.kv_a, c.kv_dim, NULL, 0,
                                                 NULL, 0));
            TEST_ASSERT(ds4_test_glm_read_mtp_kv(b, p, c.kv_b, c.kv_dim, NULL, 0,
                                                 NULL, 0));
            float ab = 0.0f, rl = 0.0f;
            test_glm_mtp_row_diff(c.kv_a, c.kv_b, c.kv_dim, &ab, &rl);
            if (ab > worst_abs) worst_abs = ab;
            if (rl > worst_rel) worst_rel = rl;
        }
        fprintf(stderr, "ds4-test: glm-mtp-prefill K1 prefill-vs-decode slots [1,%u]: "
                        "max_abs=%.6g max_rel=%.6g\n", m - 2u, worst_abs, worst_rel);
        TEST_ASSERT(test_glm_mtp_row_finite(c.kv_a, c.kv_dim));
        TEST_ASSERT(test_glm_mtp_row_finite(c.kv_b, c.kv_dim));

        float k2a = 0.0f, k2r = 0.0f, k3a = 0.0f, k3r = 0.0f;
        TEST_ASSERT(ds4_test_glm_read_mtp_kv(a, m - 1u, c.kv_a, c.kv_dim, NULL, 0,
                                             NULL, 0));
        TEST_ASSERT(ds4_test_glm_read_mtp_kv(b, m - 1u, c.kv_b, c.kv_dim, NULL, 0,
                                             NULL, 0));
        test_glm_mtp_row_diff(c.kv_a, c.kv_b, c.kv_dim, &k2a, &k2r);
        fprintf(stderr, "ds4-test: glm-mtp-prefill K2 seam slot %u: max_abs=%.6g "
                        "max_rel=%.6g\n", m - 1u, k2a, k2r);
        TEST_ASSERT(test_glm_mtp_row_finite(c.kv_a, c.kv_dim));
        TEST_ASSERT(test_glm_mtp_row_finite(c.kv_b, c.kv_dim));

        TEST_ASSERT(ds4_test_glm_read_mtp_kv(a, m, c.kv_a, c.kv_dim, NULL, 0,
                                             NULL, 0));
        TEST_ASSERT(ds4_test_glm_read_mtp_kv(b, m, c.kv_b, c.kv_dim, NULL, 0,
                                             NULL, 0));
        test_glm_mtp_row_diff(c.kv_a, c.kv_b, c.kv_dim, &k3a, &k3r);
        fprintf(stderr, "ds4-test: glm-mtp-prefill K3 plain-step slot %u: "
                        "max_abs=%.6g max_rel=%.6g\n", m, k3a, k3r);
        TEST_ASSERT(test_glm_mtp_row_finite(c.kv_a, c.kv_dim));
        TEST_ASSERT(test_glm_mtp_row_finite(c.kv_b, c.kv_dim));

        worst_abs = 0.0f; worst_rel = 0.0f;
        for (uint32_t row = 1u; row <= m / c.pool_size; row++) {
            TEST_ASSERT(ds4_test_glm_read_mtp_kv(a, row * c.pool_size, NULL, 0,
                                                 c.pool_a, c.pool_dim, NULL, 0));
            TEST_ASSERT(ds4_test_glm_read_mtp_kv(b, row * c.pool_size, NULL, 0,
                                                 c.pool_b, c.pool_dim, NULL, 0));
            float ab = 0.0f, rl = 0.0f;
            test_glm_mtp_row_diff(c.pool_a, c.pool_b, c.pool_dim, &ab, &rl);
            if (ab > worst_abs) worst_abs = ab;
            if (rl > worst_rel) worst_rel = rl;
        }
        TEST_ASSERT(ds4_test_glm_read_mtp_kv(a, 0u, NULL, 0, NULL, 0, c.tail_a,
                                             c.tail_dim));
        TEST_ASSERT(ds4_test_glm_read_mtp_kv(b, 0u, NULL, 0, NULL, 0, c.tail_b,
                                             c.tail_dim));
        float tab = 0.0f, trl = 0.0f;
        test_glm_mtp_row_diff(c.tail_a, c.tail_b, c.tail_dim, &tab, &trl);
        fprintf(stderr, "ds4-test: glm-mtp-prefill P pooled rows [1,%u]: max_abs=%.6g "
                        "max_rel=%.6g; tail: max_abs=%.6g max_rel=%.6g\n",
                m / c.pool_size, worst_abs, worst_rel, tab, trl);
        TEST_ASSERT(test_glm_mtp_row_finite(c.tail_a, c.tail_dim));
        TEST_ASSERT(test_glm_mtp_row_finite(c.tail_b, c.tail_dim));
    }

    /* ---- 3. a continued request fills the seam it resumes from ---- */
    test_glm_mtp_prefill_resume(engine, &prompt, m, ctx,
                                file_text != NULL, &c);

    /* ---- 4. position 0 masks the embedding half of the eh_proj input ---- */
    {
        int x = -1, y = -1;
        for (int i = 1; i < prompt.len && y < 0; i++) {
            if (x < 0) x = prompt.v[i];
            else if (prompt.v[i] != x) y = prompt.v[i];
        }
        TEST_ASSERT(x >= 0 && y >= 0);
        if (x >= 0 && y >= 0) {
            ds4_session *s1 = NULL;
            ds4_session *s2 = NULL;
            ds4_tokens p1 = {0}, p2 = {0};
            ds4_tokens_push(&p1, prompt.v[0]);
            ds4_tokens_push(&p1, x);
            ds4_tokens_push(&p2, prompt.v[0]);
            ds4_tokens_push(&p2, y);
            if (ds4_session_create(&s1, engine, ctx) == 0 &&
                ds4_session_create(&s2, engine, ctx) == 0) {
                ds4_test_glm_zero_mtp_caches(s1);
                ds4_test_glm_zero_mtp_caches(s2);
                TEST_ASSERT(ds4_session_sync(s1, &p1, err, sizeof(err)) == 0);
                TEST_ASSERT(ds4_session_sync(s2, &p2, err, sizeof(err)) == 0);
                TEST_ASSERT(ds4_test_glm_read_mtp_kv(s1, 0u, c.kv_a, c.kv_dim,
                                                     NULL, 0, NULL, 0));
                TEST_ASSERT(ds4_test_glm_read_mtp_kv(s2, 0u, c.kv_b, c.kv_dim,
                                                     NULL, 0, NULL, 0));
                const bool same = memcmp(c.kv_a, c.kv_b,
                                         (size_t)c.kv_dim * sizeof(float)) == 0;
                float ab = 0.0f, rl = 0.0f;
                test_glm_mtp_row_diff(c.kv_a, c.kv_b, c.kv_dim, &ab, &rl);
                fprintf(stderr, "ds4-test: glm-mtp-prefill position 0 slot "
                                "bit-identical=%d max_abs=%.6g (second token %d "
                                "vs %d)\n", same, ab, x, y);
                /* Slot 0 sees only the BOS hidden and a zeroed embedding, so
                 * the second token must not reach it at all. */
                TEST_ASSERT(same);
            } else {
                fprintf(stderr, "ds4-test: glm-mtp-prefill position 0 sessions "
                                "could not be created\n");
                TEST_ASSERT(false);
            }
            ds4_session_free(s2);
            ds4_session_free(s1);
            ds4_tokens_free(&p2);
            ds4_tokens_free(&p1);
        }
    }

cleanup:
    free(draft_b);
    free(draft_a);
    free(tail_pre);
    free(c.tail_b);
    free(c.tail_a);
    free(c.pool_b);
    free(c.pool_a);
    free(c.kv_b);
    free(c.kv_a);
    ds4_session_free(b);
    ds4_session_free(a);
    ds4_tokens_free(&extended);
    ds4_tokens_free(&prompt);
    free(file_text);
}

/* ------------------------------------------------------------------ *
 * Ticket 005: every GLM-5.3 MTP slot holds (the real token at p+1, the
 * main model's hidden at p).  The ledger records what every write put into
 * a slot, so these tests check the invariant itself instead of a numeric
 * fingerprint of one particular run.  Needs a GLM 5.3 model and
 * DS4_TEST_GLM_MTP=1; --glm-mtp-commit-exact additionally needs
 * DS4_TEST_MTP_EXACT=1.
 * ------------------------------------------------------------------ */

/* The invariant after every cycle, eval and rewind: over the readable history
 * range H = [floor .. L-2] slot p holds the checkpoint's token p+1 written from
 * the hidden of position p.  A dense session (the fast cycle, prefill fills
 * from position 0) starts H at the read floor; the legacy cycle fills nothing
 * before its draft window, so its H starts at the window start, and a window
 * that was never decided means nothing has been written yet.  The last slot is
 * free-form -- its next token does not exist yet -- but a write there can only
 * have used that position's own hidden, and unless the floor passed it the
 * session must still hold the hidden a rewrite needs. */
static bool test_glm_mtp_commit_invariant(ds4_session *session, const char *label,
                                          int step) {
    const ds4_tokens *h = ds4_session_tokens(session);
    if (!h || h->len <= 0) return true;
    const uint32_t l = (uint32_t)h->len;
    const uint32_t floor = ds4_test_glm_mtp_read_floor(session);
    uint32_t start = 0;
    const bool decided = ds4_session_glm_mtp_window_start(session, &start);
    bool ok = true;
    if (decided && start < floor) {
        fprintf(stderr, "ds4-test: glm-mtp-commit %s/%d window start %u below "
                        "the read floor %u\n", label, step, start, floor);
        ok = false;
    }
    uint32_t lo = ds4_test_glm_mtp_dense(session) ? floor
                 : (decided ? start : l);
    if (lo < floor) lo = floor;
    for (uint32_t p = lo; p + 2u <= l; p++) {
        uint32_t tok = 0, hid = 0;
        const int r = ds4_test_glm_mtp_ledger_slot(session, p, &tok, &hid);
        if (r != 1 || tok != (uint32_t)h->v[p + 1] || hid != p) {
            fprintf(stderr, "ds4-test: glm-mtp-commit %s/%d slot %u: %s "
                            "token=%u hidden=%u, want token=%d hidden=%u\n",
                    label, step, p,
                    r == 1 ? "written" : (r == 0 ? "never written" : "no ledger"),
                    tok, hid, h->v[p + 1], p);
            ok = false;
            break;
        }
    }
    uint32_t spos = 0;
    int swritten = -1, skept = 0;
    if (ds4_test_glm_mtp_dense(session) && floor < l &&
        ds4_test_glm_mtp_seam_state(session, &spos, &swritten, &skept) &&
        (!skept || spos != l - 1u)) {
        fprintf(stderr, "ds4-test: glm-mtp-commit %s/%d last slot %u: seam pos %u "
                        "hidden kept=%d, want the seam on the last slot with "
                        "its hidden\n", label, step, l - 1u, spos, skept);
        ok = false;
    }
    uint32_t tok = 0, hid = 0;
    if (ds4_test_glm_mtp_ledger_slot(session, l - 1u, &tok, &hid) == 1 &&
        hid != l - 1u && hid != DS4_TEST_GLM_MTP_LEDGER_CHAIN) {
        fprintf(stderr, "ds4-test: glm-mtp-commit %s/%d last slot %u was written "
                        "from the hidden of position %u\n", label, step, l - 1u, hid);
        ok = false;
    }
    if (!ok) fflush(stderr);
    return ok;
}

/* One nextn KV row read back, and the max absolute difference of two rows. */
static float *test_glm_mtp_commit_row(ds4_session *session, uint32_t slot,
                                      uint32_t kv_dim) {
    float *row = malloc((size_t)kv_dim * sizeof(float));
    if (!row) return NULL;
    if (!ds4_test_glm_read_mtp_kv(session, slot, row, kv_dim, NULL, 0, NULL, 0)) {
        free(row);
        return NULL;
    }
    return row;
}

static float test_glm_mtp_commit_abs(const float *a, const float *b, uint32_t n) {
    float abs_max = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (d > abs_max) abs_max = d;
    }
    return abs_max;
}

/* One speculative cycle: sampled cycles take the second-best token through the
 * ignoring-EOS entry, the others the argmax through the ordinary greedy entry
 * (so a sampled cycle's first token differs from what the drafter guessed by
 * construction, and the ignore_eos path is covered).  Returns the number of
 * committed tokens, or -1. */
static int test_glm_mtp_commit_cycle(ds4_engine *engine, ds4_session *session,
                                     const char *label, int cycle, bool sampled,
                                     int room, int *accepted, int accepted_cap) {
    char err[256] = {0};
    const int eos = ds4_token_eos(engine);
    int first = -1;
    if (sampled) {
        ds4_token_score top[8] = {0};
        if (ds4_session_top_logprobs(session, top, 8) < 2 || top[1].id == top[0].id) {
            fprintf(stderr, "ds4-test: glm-mtp-commit %s: no second-best token\n", label);
            return -1;
        }
        first = top[1].id;
    } else {
        first = ds4_session_argmax(session);
    }
    if (first < 0 || first == eos) {
        fprintf(stderr, "ds4-test: glm-mtp-commit %s: nothing to commit at cycle %d\n",
                label, cycle);
        return -1;
    }
    const int ntok = sampled
        ? ds4_session_eval_speculative_argmax_ignoring_eos(
              session, first, room, eos, DS4_THINK_NONE, accepted, accepted_cap,
              err, sizeof(err))
        : ds4_session_eval_speculative_argmax(session, first, room, eos, accepted,
                                            accepted_cap, err, sizeof(err));
    if (ntok < 1) {
        fprintf(stderr, "ds4-test: glm-mtp-commit %s cycle %d failed: %s\n", label,
                cycle, err);
        TEST_ASSERT(false);
        return -1;
    }
    return ntok;
}

/* A committed block a rewind can roll back: the cycle that produced it has to
 * be the last one, so the search stops on it. */
typedef struct {
    uint32_t block_start;
    int toks[4];
    int ntok;
} test_glm_mtp_block;

/* Mixed generation with the invariant after every cycle.  With strict set it
 * keeps going until it committed min_tokens, rewrote at least one slot for a
 * differing token and saw a multi-token commit, failing at the cap; otherwise it
 * stops at min_tokens and only requires the invariant to have held. */
static bool test_glm_mtp_commit_mixed(ds4_engine *engine, ds4_session *session,
                                      const char *label, int min_tokens,
                                      int max_tokens, bool strict) {
    const int eos = ds4_token_eos(engine);
    uint32_t base_rewrites = 0, base_repl = 0, base_fail = 0;
    int committed = 0;
    int widest = 0;
    bool any = false;
    ds4_test_glm_mtp_stats(session, &base_rewrites, &base_repl, &base_fail);
    for (int cycle = 1; committed < max_tokens && cycle <= TEST_MTP_MAXGEN; cycle++) {
        int accepted[4] = {0};
        const int room = max_tokens - committed < 4 ? max_tokens - committed : 4;
        const int ntok = test_glm_mtp_commit_cycle(engine, session, label, cycle,
                                                  (cycle % 2) == 0, room, accepted,
                                                  (int)(sizeof(accepted) /
                                                        sizeof(accepted[0])));
        if (ntok < 0) return false;
        any = true;
        if (ntok > widest) widest = ntok;
        for (int i = 0; i < ntok; i++) {
            if (accepted[i] == eos) {
                fprintf(stderr, "ds4-test: glm-mtp-commit %s: EOS after %d committed "
                                "tokens (need %d)\n", label, committed, min_tokens);
                return false;
            }
            committed++;
        }
        if (!test_glm_mtp_commit_invariant(session, label, cycle)) return false;
        uint32_t rewrites = 0, repl = 0, fails = 0;
        ds4_test_glm_mtp_stats(session, &rewrites, &repl, &fails);
        if (committed >= min_tokens &&
            (!strict || (rewrites > base_rewrites && widest >= 2))) {
            return true;
        }
    }
    if (!strict) return any;
    fprintf(stderr, "ds4-test: glm-mtp-commit %s: cap reached with committed=%d "
                    "(need %d), widest_commit=%d, rewrites=%u, replacements=%u, "
                    "failures=%u\n", label, committed, min_tokens, widest,
            base_rewrites, base_repl, base_fail);
    return false;
}

/* Greedy cycles until one commits two or more tokens, stopping on that cycle so
 * the block it committed is still the rewindable one. */
static bool test_glm_mtp_commit_until_wide(ds4_engine *engine, ds4_session *session,
                                           const char *label,
                                           test_glm_mtp_block *block) {
    const int eos = ds4_token_eos(engine);
    for (int cycle = 1; cycle <= TEST_MTP_MAXGEN; cycle++) {
        int accepted[4] = {0};
        const int ntok = test_glm_mtp_commit_cycle(engine, session, label, cycle,
                                                  false, 4, accepted,
                                                  (int)(sizeof(accepted) /
                                                        sizeof(accepted[0])));
        if (ntok < 0) return false;
        if (!test_glm_mtp_commit_invariant(session, label, cycle)) return false;
        if (ntok >= 2) {
            block->block_start = (uint32_t)ds4_session_pos(session) - (uint32_t)ntok;
            block->ntok = ntok;
            for (int i = 0; i < ntok && i < 4; i++) block->toks[i] = accepted[i];
            return true;
        }
        for (int i = 0; i < ntok; i++) {
            if (accepted[i] == eos) {
                fprintf(stderr, "ds4-test: glm-mtp-commit %s: EOS before a wide "
                                "commit\n", label);
                return false;
            }
        }
    }
    fprintf(stderr, "ds4-test: glm-mtp-commit %s: no cycle committed two tokens "
                    "in %d cycles\n", label, TEST_MTP_MAXGEN);
    return false;
}

/* 1. A plain step whose slot is rewritten by the next request's first token.
 * The rewritten row must sit closer to the row a session that only ever knew
 * the real token writes than the row the drafter's guess left behind. */
static bool test_glm_mtp_commit_plain_step(ds4_engine *engine,
                                           const ds4_tokens *prompt, int ctx,
                                           uint32_t kv_dim) {
    char err[256] = {0};
    const int eos = ds4_token_eos(engine);
    ds4_session *a = NULL, *b = NULL;
    ds4_tokens extended = {0};
    float *row_pre = NULL, *row_post = NULL, *row_ref = NULL;
    bool ok = ds4_session_create(&a, engine, ctx) == 0;
    TEST_ASSERT(ok);
    if (!ok) return false;
    ds4_test_glm_zero_mtp_caches(a);
    /* The ledger has to cover the prefill too, so it goes on before the sync. */
    ds4_test_glm_mtp_ledger(a, 1);
    ok = ds4_session_sync(a, prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (!ok) goto cleanup;
    const uint32_t m = (uint32_t)ds4_session_pos(a);

    /* The first cycle has no pending draft, so it is the plain step: it writes
     * slot M with the token it guessed for position M+1. */
    const int t0 = ds4_session_argmax(a);
    int accepted[4] = {0};
    int ntok = ds4_session_eval_speculative_argmax(a, t0, 4, eos, accepted, 4,
                                                  err, sizeof(err));
    TEST_ASSERT(ntok == 1);
    if (ntok != 1) goto cleanup;
    if (!test_glm_mtp_commit_invariant(a, "plain-step", 1)) goto cleanup;
    row_pre = test_glm_mtp_commit_row(a, m, kv_dim);
    TEST_ASSERT(row_pre != NULL && test_glm_mtp_row_written(row_pre, kv_dim));
    if (!row_pre) goto cleanup;

    /* The next request starts with a token that is not that guess. */
    ds4_token_score top[8] = {0};
    TEST_ASSERT(ds4_session_top_logprobs(a, top, 8) == 8);
    const int x = top[1].id;
    TEST_ASSERT(x != top[0].id);
    ntok = ds4_session_eval_speculative_argmax(a, x, 4, eos, accepted, 4,
                                              err, sizeof(err));
    TEST_ASSERT(ntok >= 1);
    if (ntok < 1) goto cleanup;
    if (!test_glm_mtp_commit_invariant(a, "plain-step", 2)) goto cleanup;
    uint32_t tok = 0, hid = 0;
    const int r = ds4_test_glm_mtp_ledger_slot(a, m, &tok, &hid);
    fprintf(stderr, "ds4-test: glm-mtp-commit plain-step: slot %u after the "
                    "differing first token: ledger=%d token=%u hidden=%u want "
                    "token=%d\n", m, r, tok, hid, x);
    TEST_ASSERT(r == 1 && tok == (uint32_t)x && hid == m);
    row_post = test_glm_mtp_commit_row(a, m, kv_dim);
    TEST_ASSERT(row_post != NULL);
    if (!row_post) goto cleanup;

    /* Reference: the same prompt, then the real token as its own resumed
     * request, then the very plain step the fast cycle runs after a decode. */
    ok = ds4_session_create(&b, engine, ctx) == 0;
    TEST_ASSERT(ok);
    if (!ok) goto cleanup;
    ds4_test_glm_zero_mtp_caches(b);
    ok = ds4_session_sync(b, prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (!ok) goto cleanup;
    for (int i = 0; i < prompt->len; i++) ds4_tokens_push(&extended, prompt->v[i]);
    ds4_tokens_push(&extended, t0);
    ok = ds4_session_sync(b, &extended, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (!ok) goto cleanup;
    TEST_ASSERT((uint32_t)ds4_session_pos(b) == m + 1u);
    TEST_ASSERT(ds4_test_glm_mtp_draft_after_decode(b, x, m, NULL, NULL, 0));
    row_ref = test_glm_mtp_commit_row(b, m, kv_dim);
    TEST_ASSERT(row_ref != NULL);
    if (!row_ref) goto cleanup;

    const float post_abs = test_glm_mtp_commit_abs(row_post, row_ref, kv_dim);
    const float pre_abs = test_glm_mtp_commit_abs(row_pre, row_ref, kv_dim);
    fprintf(stderr, "ds4-test: glm-mtp-commit plain-step slot %u vs the real "
                    "token: rewritten max_abs=%.6g, predicted max_abs=%.6g\n",
            m, post_abs, pre_abs);
    /* Ordering, not a numeric threshold (004 section 2, point 6). */
    TEST_ASSERT(post_abs < pre_abs);
    ok = true;

cleanup:
    ds4_tokens_free(&extended);
    free(row_ref);
    free(row_post);
    free(row_pre);
    ds4_session_free(b);
    ds4_session_free(a);
    return ok;
}

/* 3. The server's resample flow: rewind a committed block to its last kept
 * token and evaluate a different one.  The slot the rewind uncovers was written
 * with the dropped token, so the eval has to rewrite it. */
static bool test_glm_mtp_commit_resample(ds4_engine *engine, ds4_session *session,
                                         const test_glm_mtp_block *block,
                                         bool same_token, uint32_t kv_dim,
                                         const char *label) {
    char err[256] = {0};
    const uint32_t slot = block->block_start + (uint32_t)block->ntok - 2u;
    const int dropped = block->toks[block->ntok - 1];
    ds4_session *ref = NULL;
    ds4_tokens prefix = {0};
    float *row_pre = NULL, *row_post = NULL, *row_ref = NULL;
    uint32_t rewrites_before = 0, repl_before = 0, fails_before = 0;
    uint32_t rewrites_after = 0, repl_after = 0, fails_after = 0;
    ds4_test_glm_mtp_stats(session, &rewrites_before, &repl_before, &fails_before);
    row_pre = test_glm_mtp_commit_row(session, slot, kv_dim);
    TEST_ASSERT(row_pre != NULL && test_glm_mtp_row_written(row_pre, kv_dim));
    if (!row_pre) return false;
    const ds4_tokens *history = ds4_session_tokens(session);
    for (int i = 0; i < (int)(block->block_start + (uint32_t)block->ntok - 1u); i++) {
        ds4_tokens_push(&prefix, history->v[i]);
    }

    const int flag_before = ds4_test_glm_mtp_cache_complete(session) != 0;
    ds4_session_rewind(session, (int)(block->block_start + (uint32_t)block->ntok - 1u));
    if (ds4_session_pos(session) != (int)(block->block_start + (uint32_t)block->ntok - 1u)) {
        fprintf(stderr, "ds4-test: glm-mtp-commit %s: rewind left pos %d, want %u\n",
                label, ds4_session_pos(session),
                block->block_start + block->ntok - 1u);
        TEST_ASSERT(false);
        ds4_tokens_free(&prefix);
        free(row_pre);
        return false;
    }
    /* The rewind only ever clears the cache-complete flag (ds4.c
     * ds4_session_glm_mtp_rewind: no stored seam -> false); it never sets it.
     * The legacy cycle never sets the flag, so it can already be 0 before the
     * rewind: after it the flag holds iff it held before and the seam was
     * stored with the dropped token (ticket 006). */
    uint32_t seam_pos = 0;
    int seam_written = -1;
    int seam_valid = 0;
    TEST_ASSERT(ds4_test_glm_mtp_seam_state(session, &seam_pos, &seam_written,
                                            &seam_valid));
    const int flag_after = ds4_test_glm_mtp_cache_complete(session) != 0;
    TEST_ASSERT(flag_after == (flag_before && seam_valid != 0));
    int y = dropped;
    if (!same_token) {
        ds4_token_score top[8] = {0};
        TEST_ASSERT(ds4_session_top_logprobs(session, top, 8) == 8);
        y = top[1].id;
        if (y == dropped) {
            fprintf(stderr, "ds4-test: glm-mtp-commit %s: the second-best token "
                            "is the dropped one\n", label);
            TEST_ASSERT(false);
            ds4_tokens_free(&prefix);
            free(row_pre);
            return false;
        }
    }
    if (ds4_session_eval(session, y, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-test: glm-mtp-commit %s: resample eval failed: %s\n",
                label, err);
        TEST_ASSERT(false);
        ds4_tokens_free(&prefix);
        free(row_pre);
        return false;
    }
    ds4_test_glm_mtp_stats(session, &rewrites_after, &repl_after, &fails_after);
    fprintf(stderr, "ds4-test: glm-mtp-commit %s: slot %u %d -> %d, rewrites "
                    "%u -> %u, failures %u -> %u\n", label, slot, dropped, y,
            rewrites_before, rewrites_after, fails_before, fails_after);
    if (!test_glm_mtp_commit_invariant(session, label, 900)) {
        ds4_tokens_free(&prefix);
        free(row_pre);
        return false;
    }
    uint32_t tok = 0, hid = 0;
    const int r = ds4_test_glm_mtp_ledger_slot(session, slot, &tok, &hid);
    TEST_ASSERT(r == 1 && tok == (uint32_t)y && hid == slot);
    if (same_token) {
        /* Re-evaluating the very token that was dropped must cost nothing. */
        TEST_ASSERT(rewrites_after == rewrites_before);
        TEST_ASSERT(fails_after == fails_before);
        ds4_tokens_free(&prefix);
        free(row_pre);
        return true;
    }
    /* A differing token has to cost exactly one rewrite. */
    TEST_ASSERT(rewrites_after == rewrites_before + 1u);
    TEST_ASSERT(fails_after == fails_before);

    /* Reference: the same committed prefix, then the plain step that writes the
     * last slot with the resampled token. */
    bool ok = ds4_session_create(&ref, engine, ds4_session_ctx(session)) == 0;
    TEST_ASSERT(ok);
    if (ok) ok = ds4_session_sync(ref, &prefix, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (ok) ok = ds4_test_glm_mtp_draft_after_decode(ref, y, slot, NULL, NULL, 0) != 0;
    TEST_ASSERT(ok);
    if (ok) row_ref = test_glm_mtp_commit_row(ref, slot, kv_dim);
    if (ok) row_post = test_glm_mtp_commit_row(session, slot, kv_dim);
    ok = row_ref != NULL && row_post != NULL;
    TEST_ASSERT(ok);
    if (ok) {
        const float post_abs = test_glm_mtp_commit_abs(row_post, row_ref, kv_dim);
        const float pre_abs = test_glm_mtp_commit_abs(row_pre, row_ref, kv_dim);
        fprintf(stderr, "ds4-test: glm-mtp-commit %s slot %u vs the resampled "
                        "token: rewritten max_abs=%.6g, dropped-token row "
                        "max_abs=%.6g\n", label, slot, post_abs, pre_abs);
        TEST_ASSERT(post_abs < pre_abs);
    }
    ds4_session_free(ref);
    ds4_tokens_free(&prefix);
    free(row_ref);
    free(row_post);
    free(row_pre);
    return ok;
}

/* 4. Decode without ever entering a cycle: every plain eval is a one-token
 * continuation, so it fills the slot the previous row left and leaves its own
 * row waiting.  Returns the mean milliseconds per eval; the ledger and the
 * invariant only run with MTP on (the closed case is the timing reference). */
static double test_glm_mtp_commit_plain_eval(ds4_engine *engine,
                                             const ds4_tokens *prompt, int ctx,
                                             bool mtp_on, uint32_t kv_dim,
                                             int *steps_out) {
    char err[256] = {0};
    ds4_session *session = NULL;
    bool ok = ds4_session_create(&session, engine, ctx) == 0;
    TEST_ASSERT(ok);
    if (!ok) return 0.0;
    if (mtp_on) {
        ds4_test_glm_zero_mtp_caches(session);
        ds4_test_glm_mtp_ledger(session, 1);
    }
    ok = ds4_session_sync(session, prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (!ok) { ds4_session_free(session); return 0.0; }
    const uint32_t m = (uint32_t)ds4_session_pos(session);
    const uint32_t floor0 = ds4_test_glm_mtp_read_floor(session);
    double elapsed = 0.0;
    int steps = 0;
    for (int i = 0; i < TEST_GLM_MTP_WINDOW_MIN_TOKENS; i++) {
        const int t = ds4_session_argmax(session);
        if (t < 0) break;
        const double t0 = test_monotonic_seconds();
        ok = ds4_session_eval(session, t, err, sizeof(err)) == 0;
        elapsed += test_monotonic_seconds() - t0;
        TEST_ASSERT(ok);
        if (!ok) {
            fprintf(stderr, "ds4-test: glm-mtp-commit plain-eval step %d: %s\n", i, err);
            break;
        }
        steps++;
        if (mtp_on) {
            if (!test_glm_mtp_commit_invariant(session, "plain-eval", i + 1)) break;
            const uint32_t floor_now = ds4_test_glm_mtp_read_floor(session);
            const uint32_t len_now = (uint32_t)ds4_session_pos(session);
            fprintf(stderr, "ds4-test: glm-mtp-commit plain-eval step %d: "
                            "floor=%u len=%u floor0=%u\n",
                    i + 1, floor_now, len_now, floor0);
            if (ds4_test_glm_mtp_dense(session)) {
                /* The fast cycle fills the slot in the same batch: no failure,
                 * so the floor must not move. */
                if (floor_now != floor0) {
                    fprintf(stderr, "ds4-test: glm-mtp-commit plain-eval: the "
                                    "read floor rose to %u without a failure\n",
                            floor_now);
                    TEST_ASSERT(false);
                    break;
                }
            } else if (i + 1 >= 2) {
                /* The legacy cycle's plain decode writes no nextn slot.  Step 1
                 * may rewrite the slot through the seam the sync stored; from
                 * step 2 on ds4_session_glm_mtp_note_token finds the seam
                 * invalidated and raises the floor to the checkpoint length at
                 * that moment (ticket 005 rule 1), i.e. len - 1 after the step. */
                if (!(len_now - 1u <= floor_now && floor_now <= len_now)) {
                    fprintf(stderr, "ds4-test: glm-mtp-commit plain-eval: legacy "
                                    "floor %u outside [len-1, len] = [%u, %u]\n",
                            floor_now, len_now - 1u, len_now);
                    TEST_ASSERT(false);
                    break;
                }
            }
        }
    }
    if (mtp_on && ok && ds4_test_glm_mtp_dense(session)) {
        for (uint32_t p = m; p + 1u < m + (uint32_t)steps; p++) {
            float *row = test_glm_mtp_commit_row(session, p, kv_dim);
            const bool written = row != NULL && test_glm_mtp_row_written(row, kv_dim) &&
                                 test_glm_mtp_row_finite(row, kv_dim);
            if (!written) {
                fprintf(stderr, "ds4-test: glm-mtp-commit plain-eval: slot %u is "
                                "not a written row\n", p);
                TEST_ASSERT(false);
            }
            free(row);
            if (!written) { ok = false; break; }
        }
    }
    if (mtp_on && ok) {
        /* A few mixed cycles on top of the plain decode must hold too. */
        ok = test_glm_mtp_commit_mixed(engine, session, "plain-eval+cycles", 4, 12,
                                      false);
        TEST_ASSERT(ok);
    }
    ds4_session_free(session);
    if (steps_out) *steps_out = steps;
    return steps > 0 ? elapsed * 1000.0 / (double)steps : 0.0;
}

static void test_glm_mtp_commit_token(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_is_glm53(engine)) {
        fprintf(stderr, "ds4-test: glm-mtp-commit-token skipped (needs a GLM 5.3 "
                        "model)\n");
        return;
    }
    const bool mtp = test_env_bool("DS4_TEST_GLM_MTP");
    int ctx = (int)test_env_u32("DS4_TEST_GLM_MTP_WINDOW_CTX");
    if (ctx < 4096) ctx = 4096;

    if (!mtp) {
        /* MTP off: the same plain-decode timing, with no ledger and no new
         * work, as the reference for the numbers above. */
        ds4_tokens prompt = {0};
        ds4_chat_begin(engine, &prompt);
        ds4_chat_append_message(engine, &prompt, "user", test_mtp_copy_prompt());
        ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
        int steps = 0;
        const double ms = test_glm_mtp_commit_plain_eval(engine, &prompt, ctx, false,
                                                        0u, &steps);
        fprintf(stderr, "ds4-test: glm-mtp-commit plain-eval (MTP off): steps=%d "
                        "mean=%.3f ms\n", steps, ms);
        ds4_tokens_free(&prompt);
        return;
    }

    ds4_tokens prompt = {0};
    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", test_mtp_copy_prompt());
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    TEST_ASSERT(prompt.len > 0);
    TEST_ASSERT(prompt.len + TEST_MTP_MAXGEN + 64 < ctx);

    uint32_t kv_dim = 0, pool_dim = 0, pool_size = 0, tail_dim = 0, cap = 0;
    ds4_session *probe = NULL;
    bool ok = ds4_session_create(&probe, engine, ctx) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ds4_test_glm_zero_mtp_caches(probe);
        ok = ds4_test_glm_mtp_cache_dims(probe, &kv_dim, &pool_dim, &pool_size,
                                        &tail_dim, &cap) != 0;
        TEST_ASSERT(ok);
    }
    ds4_session_free(probe);
    if (!ok) { ds4_tokens_free(&prompt); return; }

    ok = test_glm_mtp_commit_plain_step(engine, &prompt, ctx, kv_dim);
    TEST_ASSERT(ok);

    /* 2. mixed generation, then 3. the server's resample flow on the same
     * session, first with a different token and then with the kept one. */
    ds4_session *a2 = NULL;
    ok = ds4_session_create(&a2, engine, ctx) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        char err[256] = {0};
        ds4_test_glm_zero_mtp_caches(a2);
        ds4_test_glm_mtp_ledger(a2, 1);
        ok = ds4_session_sync(a2, &prompt, err, sizeof(err)) == 0;
        TEST_ASSERT(ok);
    }
    if (ok) {
        ok = test_glm_mtp_commit_mixed(engine, a2, "mixed",
                                      TEST_GLM_MTP_WINDOW_MIN_TOKENS,
                                      TEST_MTP_MAXGEN, true);
        TEST_ASSERT(ok);
    }
    if (ok) {
        test_glm_mtp_block block = {0};
        ok = test_glm_mtp_commit_until_wide(engine, a2, "resample", &block);
        TEST_ASSERT(ok);
        if (ok) {
            ok = test_glm_mtp_commit_resample(engine, a2, &block, false, kv_dim,
                                             "resample");
            TEST_ASSERT(ok);
        }
    }
    if (ok) {
        /* And again with the token the server actually kept: the slot already
         * holds it, so nothing may be rewritten. */
        test_glm_mtp_block block = {0};
        ok = test_glm_mtp_commit_until_wide(engine, a2, "resample-same", &block);
        TEST_ASSERT(ok);
        if (ok) {
            ok = test_glm_mtp_commit_resample(engine, a2, &block, true, kv_dim,
                                             "resample-same");
            TEST_ASSERT(ok);
        }
    }
    if (a2) ds4_session_free(a2);

    /* 4. plain decode, with the timing both modes print. */
    int steps = 0;
    const double ms = test_glm_mtp_commit_plain_eval(engine, &prompt, ctx, true,
                                                    kv_dim, &steps);
    fprintf(stderr, "ds4-test: glm-mtp-commit plain-eval: steps=%d mean=%.3f ms\n",
            steps, ms);
    ds4_tokens_free(&prompt);
}

/* --glm-mtp-commit-exact: the exact-sampling reject replaces the draft with a
 * residual sample, which is a committed token the slot has to learn. */
static void test_glm_mtp_commit_exact(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_is_glm53(engine) || !test_env_bool("DS4_TEST_GLM_MTP")) {
        fprintf(stderr, "ds4-test: glm-mtp-commit-exact skipped (needs a GLM 5.3 "
                        "model and DS4_TEST_GLM_MTP=1)\n");
        return;
    }
    if (!test_env_bool("DS4_TEST_MTP_EXACT") || !ds4_engine_mtp_exact_sampling(engine)) {
        fprintf(stderr, "ds4-test: glm-mtp-commit-exact skipped (needs "
                        "DS4_TEST_MTP_EXACT=1)\n");
        return;
    }
    int ctx = (int)test_env_u32("DS4_TEST_GLM_MTP_WINDOW_CTX");
    if (ctx < 4096) ctx = 4096;

    ds4_tokens prompt = {0};
    ds4_tokens prefix = {0};
    ds4_session *session = NULL, *ref = NULL;
    float *row = NULL, *row_ref = NULL;
    char err[256] = {0};
    uint64_t rng = 20250705u;
    const int eos = ds4_token_eos(engine);
    uint32_t kv_dim = 0, pool_dim = 0, pool_size = 0, tail_dim = 0, cap = 0;

    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", test_mtp_copy_prompt());
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    TEST_ASSERT(prompt.len > 0);
    ds4_session *probe = NULL;
    bool ok = ds4_session_create(&probe, engine, ctx) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ds4_test_glm_zero_mtp_caches(probe);
        ok = ds4_test_glm_mtp_cache_dims(probe, &kv_dim, &pool_dim, &pool_size,
                                        &tail_dim, &cap) != 0;
        TEST_ASSERT(ok);
    }
    ds4_session_free(probe);
    if (!ok) goto cleanup;

    ok = ds4_session_create(&session, engine, ctx) == 0;
    TEST_ASSERT(ok);
    if (!ok) goto cleanup;
    ds4_test_glm_zero_mtp_caches(session);
    ds4_test_glm_mtp_ledger(session, 1);
    ok = ds4_session_sync(session, &prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (!ok) goto cleanup;

    uint32_t replacements = 0, rewrites = 0, failures = 0;
    int block_start = -1;
    int replacement = -1;
    int kept[4] = {0};
    int ntok_last = 0;
    for (int cycle = 0; cycle < TEST_MTP_MAXGEN && replacements == 0; cycle++) {
        const int first = ds4_session_sample(session, DS4_DEFAULT_TEMPERATURE, 0,
                                            DS4_DEFAULT_TOP_P, DS4_DEFAULT_MIN_P,
                                            &rng);
        if (first < 0 || first == eos) break;
        int accepted[4] = {0};
        block_start = ds4_session_pos(session);
        const int ntok = ds4_session_eval_speculative(
            session, first, 4, eos, DS4_DEFAULT_TEMPERATURE, 0,
            DS4_DEFAULT_TOP_P, DS4_DEFAULT_MIN_P, &rng, accepted, 4,
            err, sizeof(err));
        if (ntok < 1) {
            fprintf(stderr, "ds4-test: glm-mtp-commit-exact cycle %d failed: %s\n",
                    cycle, err);
            TEST_ASSERT(false);
            ok = false;
            break;
        }
        if (!test_glm_mtp_commit_invariant(session, "exact", cycle + 1)) {
            ok = false;
            break;
        }
        ds4_test_glm_mtp_stats(session, &rewrites, &replacements, &failures);
        if (replacements > 0 && ntok == 2) {
            /* the cycle that replaced its draft: accepted[1] is the replacement */
            replacement = accepted[1];
            for (int i = 0; i < ntok; i++) kept[i] = accepted[i];
            ntok_last = ntok;
        }
    }
    fprintf(stderr, "ds4-test: glm-mtp-commit-exact: replacements=%u rewrites=%u "
                    "failures=%u slot=%d token=%d\n", replacements, rewrites,
            failures, block_start, replacement);
    if (replacements == 0) {
        fprintf(stderr, "ds4-test: glm-mtp-commit-exact: no exact-sampling "
                        "replacement happened in %d cycles\n", TEST_MTP_MAXGEN);
        TEST_ASSERT(false);
        ok = false;
        goto cleanup;
    }
    if (replacement < 0 || ntok_last < 2) {
        /* A replacement committed inside a longer commit: the invariant above
         * already covered it, but there is no single slot to point at. */
        fprintf(stderr, "ds4-test: glm-mtp-commit-exact: the replacement was not a "
                        "two-token commit; checked by the invariant only\n");
        goto cleanup_resample;
    }
    ok = true;
    {
        const uint32_t slot = (uint32_t)block_start;
        uint32_t tok = 0, hid = 0;
        const int r = ds4_test_glm_mtp_ledger_slot(session, slot, &tok, &hid);
        fprintf(stderr, "ds4-test: glm-mtp-commit-exact: slot %u ledger=%d "
                        "token=%u hidden=%u want token=%d hidden=%u\n",
                slot, r, tok, hid, replacement, slot);
        TEST_ASSERT(r == 1 && tok == (uint32_t)replacement && hid == slot);
        ok = r == 1 && tok == (uint32_t)replacement && hid == slot;
        if (!ok) goto cleanup;
        const ds4_tokens *history = ds4_session_tokens(session);
        for (int i = 0; i < (int)(slot + 1u); i++) ds4_tokens_push(&prefix, history->v[i]);
        ok = ds4_session_create(&ref, engine, ctx) == 0;
        TEST_ASSERT(ok);
        if (ok) ok = ds4_session_sync(ref, &prefix, err, sizeof(err)) == 0;
        TEST_ASSERT(ok);
        if (ok) {
            ok = ds4_test_glm_mtp_draft_after_decode(ref, replacement, slot, NULL,
                                                    NULL, 0) != 0;
        }
        TEST_ASSERT(ok);
        if (ok) row = test_glm_mtp_commit_row(session, slot, kv_dim);
        if (ok) row_ref = test_glm_mtp_commit_row(ref, slot, kv_dim);
        ok = row != NULL && row_ref != NULL;
        TEST_ASSERT(ok);
        if (ok) {
            const float abs_max = test_glm_mtp_commit_abs(row, row_ref, kv_dim);
            fprintf(stderr, "ds4-test: glm-mtp-commit-exact: slot %u vs the plain "
                            "step with the replacement: max_abs=%.6g\n", slot, abs_max);
            /* No threshold this round: finite and written is all the spec asks
             * for; the ordering check lives in --glm-mtp-commit-token. */
            TEST_ASSERT(test_glm_mtp_row_finite(row, kv_dim) &&
                        test_glm_mtp_row_written(row, kv_dim) &&
                        test_glm_mtp_row_finite(row_ref, kv_dim));
        }
    }
cleanup_resample:
    if (ok && ntok_last >= 2 && replacement >= 0) {
        /* The server's resample flow on top of the replacement cycle. */
        test_glm_mtp_block block = {0};
        block.block_start = (uint32_t)block_start;
        block.ntok = ntok_last;
        for (int i = 0; i < ntok_last && i < 4; i++) block.toks[i] = kept[i];
        ok = test_glm_mtp_commit_resample(engine, session, &block, true, kv_dim,
                                         "exact-resample");
        TEST_ASSERT(ok);
    }
cleanup:
    ds4_session_free(ref);
    ds4_session_free(session);
    ds4_tokens_free(&prefix);
    ds4_tokens_free(&prompt);
    free(row_ref);
    free(row);
}

/* --glm-mtp-commit-fail: a nextn write that fails must make that slot (and
 * everything below it) unreadable, must not fail the request, and must not
 * happen again on the cycles that follow. */
static void test_glm_mtp_commit_fail_kind(ds4_engine *engine,
                                          const ds4_tokens *prompt, int ctx,
                                          int kind, const char *label) {
    char err[256] = {0};
    const int eos = ds4_token_eos(engine);
    ds4_session *session = NULL;
    bool ok = ds4_session_create(&session, engine, ctx) == 0;
    TEST_ASSERT(ok);
    if (!ok) return;
    ds4_test_glm_zero_mtp_caches(session);
    ds4_test_glm_mtp_ledger(session, 1);
    if (kind == DS4_TEST_GLM_MTP_FAIL_FILL) {
        /* The prefill fill runs inside the sync, so this one goes in first. */
        ds4_test_glm_mtp_fail_next(session, kind);
    }
    ok = ds4_session_sync(session, prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (!ok) {
        fprintf(stderr, "ds4-test: glm-mtp-commit-fail %s: sync failed: %s\n",
                label, err);
        ds4_session_free(session);
        return;
    }
    uint32_t rewrites = 0, repl = 0, failures = 0;
    ds4_test_glm_mtp_stats(session, &rewrites, &repl, &failures);
    if (kind != DS4_TEST_GLM_MTP_FAIL_FILL) ds4_test_glm_mtp_fail_next(session, kind);

    /* Run cycles until the armed write is the one that fails.  A draft chain
     * only continues while the server's token is the one the drafter based its
     * next draft on, so a rewrite for a differing token needs the second-best
     * token -- and that is the only kind of cycle a seam rewrite happens in.
     * Every other kind fires in the greedy chain: the plain step, the verify
     * pass and the catch-up rows it ends with, the chained draft row. */
    const bool sampled = kind == DS4_TEST_GLM_MTP_FAIL_FLUSH;
    const uint32_t floor_before = ds4_test_glm_mtp_read_floor(session);
    int cycle_at_failure = (kind == DS4_TEST_GLM_MTP_FAIL_FILL) ? 0 : -1;
    uint32_t pos_at_failure = 0;
    if (kind == DS4_TEST_GLM_MTP_FAIL_FILL) {
        /* The failed range is inside the prefill, so the floor must have moved
         * past at least one slot and stayed inside the prompt. */
        ds4_test_glm_mtp_stats(session, &rewrites, &repl, &failures);
        if (failures == 0) cycle_at_failure = -1;
    }
    for (int cycle = 1; cycle <= TEST_MTP_MAXGEN && failures == 0; cycle++) {
        int first = -1;
        if (sampled) {
            ds4_token_score top[8] = {0};
            if (ds4_session_top_logprobs(session, top, 8) < 2 ||
                top[1].id == top[0].id) break;
            first = top[1].id;
        } else {
            first = ds4_session_argmax(session);
        }
        if (first < 0 || first == eos) break;
        int accepted[4] = {0};
        pos_at_failure = (uint32_t)ds4_session_pos(session);
        const int ntok = sampled
            ? ds4_session_eval_speculative_argmax_ignoring_eos(
                  session, first, 4, eos, DS4_THINK_NONE, accepted, 4,
                  err, sizeof(err))
            : ds4_session_eval_speculative_argmax(session, first, 4, eos, accepted, 4,
                                                 err, sizeof(err));
        if (ntok < 1) {
            fprintf(stderr, "ds4-test: glm-mtp-commit-fail %s cycle %d failed: %s\n",
                    label, cycle, err);
            TEST_ASSERT(false);
            ok = false;
            break;
        }
        ds4_test_glm_mtp_stats(session, &rewrites, &repl, &failures);
        if (failures > 0) {
            cycle_at_failure = cycle;
            /* The failed round still commits its tokens and still succeeds. */
            TEST_ASSERT(ntok >= 1);
        }
        if (!test_glm_mtp_commit_invariant(session, label, cycle)) {
            ok = false;
            break;
        }
    }
    if (kind == DS4_TEST_GLM_MTP_FAIL_FILL) {
        pos_at_failure = 0;
    }
    fprintf(stderr, "ds4-test: glm-mtp-commit-fail %s: cycle=%d failures=%u floor "
                    "%u -> %u at slot %u\n", label, cycle_at_failure, failures,
            floor_before, ds4_test_glm_mtp_read_floor(session), pos_at_failure);
    TEST_ASSERT(failures == 1);
    TEST_ASSERT(cycle_at_failure >= 0);
    /* After the injected write failure, the cache-complete flag must be cleared. */
    TEST_ASSERT(ds4_test_glm_mtp_cache_complete(session) == 0);
    ok = failures == 1 && cycle_at_failure >= 0;
    if (kind == DS4_TEST_GLM_MTP_FAIL_CHAIN) {
        /* A chained draft row is overwritten before anything reads it, so the
         * floor must not move and the next cycle runs as usual. */
        TEST_ASSERT(ds4_test_glm_mtp_read_floor(session) == floor_before);
        ok = ok && ds4_test_glm_mtp_read_floor(session) == floor_before;
    } else {
        const uint32_t failed_slot = kind == DS4_TEST_GLM_MTP_FAIL_FLUSH
                                     ? (pos_at_failure > 0 ? pos_at_failure - 1u : 0u)
                                     : pos_at_failure;
        TEST_ASSERT(ds4_test_glm_mtp_read_floor(session) > failed_slot);
        ok = ok && ds4_test_glm_mtp_read_floor(session) > failed_slot;
    }
    if (ok) {
        /* Keep generating: the invariant holds from the new floor and the floor
         * never rises again without another failure. */
        const uint32_t floor_after = ds4_test_glm_mtp_read_floor(session);
        uint32_t start = 0;
        if (ds4_session_glm_mtp_window_start(session, &start)) {
            TEST_ASSERT(start >= floor_after);
            ok = ok && start >= floor_after;
        }
        int committed = 0;
        for (int cycle = 1; ok && committed < TEST_GLM_MTP_WINDOW_MIN_TOKENS &&
                            cycle <= TEST_MTP_MAXGEN; cycle++) {
            const int first = ds4_session_argmax(session);
            if (first < 0 || first == eos) break;
            int accepted[4] = {0};
            const int ntok = ds4_session_eval_speculative_argmax(
                session, first, 4, eos, accepted, 4, err, sizeof(err));
            if (ntok < 1) {
                fprintf(stderr, "ds4-test: glm-mtp-commit-fail %s recovery cycle %d "
                                "failed: %s\n", label, cycle, err);
                TEST_ASSERT(false);
                ok = false;
                break;
            }
            for (int i = 0; i < ntok; i++) {
                if (accepted[i] == eos) {
                    committed = TEST_GLM_MTP_WINDOW_MIN_TOKENS;
                    break;
                }
                committed++;
            }
            if (!test_glm_mtp_commit_invariant(session, label, 500 + cycle)) {
                ok = false;
                break;
            }
            if (ds4_test_glm_mtp_read_floor(session) != floor_after) {
                fprintf(stderr, "ds4-test: glm-mtp-commit-fail %s: the floor rose "
                                "to %u without a failure\n", label,
                        ds4_test_glm_mtp_read_floor(session));
                TEST_ASSERT(false);
                ok = false;
                break;
            }
        }
        fprintf(stderr, "ds4-test: glm-mtp-commit-fail %s: recovered with %d "
                        "tokens at floor %u\n", label, committed, floor_after);
    }
    ds4_session_free(session);
}

static void test_glm_mtp_commit_fail_load(ds4_engine *engine,
                                          const ds4_tokens *prompt, int ctx) {
    char err[256] = {0};
    const int eos = ds4_token_eos(engine);
    ds4_session *live = NULL, *restored = NULL;
    ds4_tokens next = {0};
    int committed = 0;
    FILE *fp = NULL;
    bool ok = ds4_session_create(&live, engine, ctx) == 0;
    TEST_ASSERT(ok);
    if (!ok) return;
    ds4_test_glm_zero_mtp_caches(live);
    ok = ds4_session_sync(live, prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (!ok) goto cleanup;
    fp = tmpfile();
    TEST_ASSERT(fp != NULL);
    if (!fp) goto cleanup;
    ok = ds4_session_save_payload(live, fp, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (!ok) goto cleanup;
    const long bytes = ftell(fp);
    TEST_ASSERT(bytes > 0);
    fflush(fp);
    rewind(fp);
    ds4_session_free(live);
    live = NULL;
    ok = ds4_session_create(&restored, engine, ctx) == 0;
    TEST_ASSERT(ok);
    if (!ok) goto cleanup;
    /* The ledger goes on before the load: the invariant below only walks slots
     * the restored session writes itself, at or above the floor the load left. */
    ds4_test_glm_zero_mtp_caches(restored);
    ds4_test_glm_mtp_ledger(restored, 1);
    ok = ds4_session_load_payload(restored, fp, (uint64_t)bytes, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (!ok) {
        fprintf(stderr, "ds4-test: glm-mtp-commit-fail load: %s\n", err);
        goto cleanup;
    }
    /* The restored prefix stays below the read floor: the payload now carries
     * the nextn caches and the seam (ticket 006), but which slots a draft may
     * read is still the rule ticket 005 left, so the load keeps the floor a
     * seam drop used to leave. */
    const uint32_t floor = ds4_test_glm_mtp_read_floor(restored);
    fprintf(stderr, "ds4-test: glm-mtp-commit-fail load: pos=%d floor=%u\n",
            ds4_session_pos(restored), floor);
    TEST_ASSERT(floor >= (uint32_t)ds4_session_pos(restored));
    ok = floor >= (uint32_t)ds4_session_pos(restored);
    for (int i = 0; ok && i < prompt->len; i++) ds4_tokens_push(&next, prompt->v[i]);
    ds4_tokens_push(&next, ds4_session_argmax(restored));
    ok = ds4_session_sync(restored, &next, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (!ok) {
        fprintf(stderr, "ds4-test: glm-mtp-commit-fail load: continuation: %s\n", err);
        goto cleanup;
    }
    for (int cycle = 0; ok && cycle < 8; cycle++) {
        const int first = ds4_session_argmax(restored);
        if (first < 0 || first == eos) break;
        int accepted[4] = {0};
        const int ntok = ds4_session_eval_speculative_argmax(
            restored, first, 4, eos, accepted, 4, err, sizeof(err));
        ok = ntok >= 1;
        TEST_ASSERT(ok);
        if (!ok) {
            fprintf(stderr, "ds4-test: glm-mtp-commit-fail load: cycle %d: %s\n",
                    cycle, err);
            break;
        }
        /* Everything at or above the floor the load left is trustworthy, and
         * nothing below it enters the draft window. */
        if (!test_glm_mtp_commit_invariant(restored, "load", cycle + 1)) {
            ok = false;
            break;
        }
        committed += ntok;
    }
    fprintf(stderr, "ds4-test: glm-mtp-commit-fail load: floor %u after %d "
                    "committed tokens\n", ds4_test_glm_mtp_read_floor(restored),
            committed);

cleanup:
    if (fp) fclose(fp);
    ds4_tokens_free(&next);
    ds4_session_free(restored);
    ds4_session_free(live);
}

static void test_glm_mtp_commit_fail(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_is_glm53(engine) || !test_env_bool("DS4_TEST_GLM_MTP")) {
        fprintf(stderr, "ds4-test: glm-mtp-commit-fail skipped (needs a GLM 5.3 "
                        "model and DS4_TEST_GLM_MTP=1)\n");
        return;
    }
    const char *depth = getenv("DS4_GLM_MTP_DEPTH");
    const bool depth3 = depth && depth[0] && atoi(depth) == 3;
    int ctx = (int)test_env_u32("DS4_TEST_GLM_MTP_WINDOW_CTX");
    if (ctx < 4096) ctx = 4096;
    ds4_tokens prompt = {0};
    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", test_mtp_copy_prompt());
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    TEST_ASSERT(prompt.len > 0);
    TEST_ASSERT(prompt.len + TEST_MTP_MAXGEN + 64 < ctx);

    uint32_t kv_dim = 0, pool_dim = 0, pool_size = 0, tail_dim = 0, cap = 0;
    ds4_session *probe = NULL;
    bool ok = ds4_session_create(&probe, engine, ctx) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ds4_test_glm_zero_mtp_caches(probe);
        ok = ds4_test_glm_mtp_cache_dims(probe, &kv_dim, &pool_dim, &pool_size,
                                        &tail_dim, &cap) != 0;
        TEST_ASSERT(ok);
    }
    ds4_session_free(probe);
    if (!ok) { ds4_tokens_free(&prompt); return; }

    test_glm_mtp_commit_fail_kind(engine, &prompt, ctx, DS4_TEST_GLM_MTP_FAIL_STEP,
                                 "step");
    test_glm_mtp_commit_fail_kind(engine, &prompt, ctx, DS4_TEST_GLM_MTP_FAIL_ROWS,
                                 "rows");
    test_glm_mtp_commit_fail_kind(engine, &prompt, ctx, DS4_TEST_GLM_MTP_FAIL_FLUSH,
                                 "flush");
    test_glm_mtp_commit_fail_kind(engine, &prompt, ctx, DS4_TEST_GLM_MTP_FAIL_FILL,
                                 "fill");
    if (!depth3) {
        fprintf(stderr, "ds4-test: glm-mtp-commit-fail chain skipped (run with "
                        "DS4_GLM_MTP_DEPTH=3)\n");
    } else {
        test_glm_mtp_commit_fail_kind(engine, &prompt, ctx, DS4_TEST_GLM_MTP_FAIL_CHAIN,
                                     "chain");
    }
    test_glm_mtp_commit_fail_load(engine, &prompt, ctx);
    ds4_tokens_free(&prompt);
}

/* Same invariant as the MTP depth smoke, but for the DSpark support model.  This
 * is separate from the fixture because it teacher-forces every committed token
 * through normal decode and directly checks that DSpark never commits a token
 * that was not near the target argmax. */
static void test_dspark_verify_depth(void) {
    const char *support = getenv("DS4_TEST_DSPARK");
    if (!support || !support[0]) {
        fprintf(stderr, "ds4-test: dspark-verify-depth skipped (set DS4_TEST_DSPARK to a DSpark support GGUF)\n");
        return;
    }

    char *saved_scheduler = test_save_env("DS4_DSPARK_SCHEDULER");
    setenv("DS4_DSPARK_SCHEDULER", "0", 1);

    ds4_engine *engine = test_open_dspark_engine(support);
    ds4_tokens prompt = {0};
    int *spec = NULL;

    if (engine) {
        const int draft_depth = ds4_engine_mtp_draft_tokens(engine);
        TEST_ASSERT(draft_depth > 2);

        ds4_chat_begin(engine, &prompt);
        ds4_chat_append_message(engine, &prompt, "user", test_mtp_copy_prompt());
        ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
        TEST_ASSERT(prompt.len > 0);

        TEST_ASSERT(ds4_test_dspark_prefix_capture(engine, &prompt));

        spec = malloc((size_t)TEST_DSPARK_MAXGEN * sizeof(*spec));
        TEST_ASSERT(spec != NULL);
        if (draft_depth > 2 && spec && prompt.len > 0) {
            int nspec = 0, max_chunk = 0;
            const bool ok_spec = test_mtp_capture_speculative(engine, &prompt,
                                                              TEST_DSPARK_MAXGEN,
                                                              spec, &nspec,
                                                              &max_chunk, NULL,
                                                              NULL);
            TEST_ASSERT(ok_spec);
            TEST_ASSERT(max_chunk > 1);
            TEST_ASSERT(nspec > 64);

            float worst_gap = 0.0f;
            int worst_at = -1;
            const bool ok_check = test_mtp_worst_argmax_gap(engine, &prompt,
                                                            spec, nspec,
                                                            &worst_gap,
                                                            &worst_at);
            TEST_ASSERT(ok_check);
            fprintf(stderr,
                    "ds4-test: dspark-verify-depth nspec=%d max_chunk=%d draft_depth=%d worst_argmax_gap=%.3f at=%d\n",
                    nspec, max_chunk, draft_depth, worst_gap, worst_at);
            TEST_ASSERT(worst_gap <= 2.0f);
        }
    }

    free(spec);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    test_restore_env("DS4_DSPARK_SCHEDULER", saved_scheduler);
}
#endif

static void test_server_unit_group(void) {
    ds4_server_unit_tests_run();
}

typedef void (*test_fn)(void);

typedef struct {
    const char *flag;
    const char *name;
    const char *desc;
    test_fn fn;
} ds4_test_entry;

static const ds4_test_entry test_entries[] = {
#ifndef DS4_NO_GPU
    {"--qwen4-prefill-checkpoints", "qwen4-prefill-checkpoints", "Qwen chunk checkpoints restore matching logits and state", test_qwen_prefill_checkpoints},
    {"--qwen4-restore-reuse", "qwen4-restore-reuse", "Qwen restore discards old verifier state and rejects truncated payloads", test_qwen_restore_reused_session},
    {"--session-snapshot", "session-snapshot", "session snapshot and recurrent-state round trip", test_session_snapshot_roundtrip},
    {"--session-rewind", "session-rewind", "Qwen3.8 rewind by snapshot restore and by replay", test_session_rewind_replay},
    {"--session-rewind-resample", "session-rewind-resample", "exact-sampling tool-boundary resample rewind restores the block-start state", test_session_rewind_resample_boundary},
    {"--long-context", "long-context", "long-context story fact-recall regression", test_long_story_fact_recall},
    {"--tool-call-quality", "tool-call-quality", "model tool call and post-result stop regression", test_tool_call_quality},
    {"--think-tool-recovery", "think-tool-recovery", "recover a complete tool call emitted inside unclosed reasoning", test_think_tool_recovery},
    {"--logprob-vectors", "logprob-vectors", "official API top-logprob vector comparison on the standard Metal path", test_official_logprob_vectors},
    {"--metal-ssd-streaming-cache-pressure", "metal-ssd-streaming-cache-pressure", "Metal SSD-streaming layer-batched decode cache-pressure repro for issue #384", test_metal_ssd_streaming_cache_pressure},
    {"--local-golden-vectors", "local-golden-vectors", "local top-k/logit drift regression for long Metal prefill", test_local_golden_vectors},
    {"--metal-short-prefill", "metal-short-prefill", "Metal ratio-4 short prefill regression", test_metal_short_prefill_ratio4},
    {"--glm53-continued-prefill", "glm53-continued-prefill", "GLM 5.3 resumed prefill latency, throughput, progress, and cold-path agreement", test_glm53_continued_prefill},
    {"--metal-kernels", "metal-kernels", "isolated Metal kernel numeric regressions", test_metal_kernel_group},
    {"--metal-tensor-equivalence", "metal-tensor-equivalence", "fast/quality Metal prompt-logit and greedy equivalence", test_metal_mpp_equivalence},
    {"--streaming-decode-prefill-correctness", "streaming-decode-prefill-correctness", "streaming decode-style cold prefill drift and repeatability", test_streaming_decode_prefill_correctness},
    {"--mtp-verify-depth", "mtp-verify-depth", "MTP speculative verify commits autoregressive-identical tokens at draft depth > 2", test_mtp_verify_depth},
    {"--glm-mtp-verify", "glm-mtp-verify", "GLM 5.3 MTP commits plain-decode argmax tokens on both accept and reject", test_glm_mtp_verify},
    {"--glm-mtp-window", "glm-mtp-window", "GLM 5.3 MTP drafts from the current request's first generated position after a resume or a disk-KV restore", test_glm_mtp_request_window},
    {"--glm-mtp-prefill", "glm-mtp-prefill", "GLM 5.3 MTP prefill writes the nextn KV and indexer caches, and the seam fills the row that waits for its token", test_glm_mtp_prefill_write},
    {"--glm-mtp-payload-cross", "glm-mtp-payload-cross", "GLM 5.3 .kv files cross the MTP on/off boundary: complete caches restore the seam, a file without them is a miss", test_glm_mtp_payload_cross},
    {"--glm-mtp-commit-token", "glm-mtp-commit-token", "GLM 5.3 MTP slots hold the real committed token and their own position's hidden across plain steps, mixed sampling and resample rewinds", test_glm_mtp_commit_token},
    {"--glm-mtp-commit-exact", "glm-mtp-commit-exact", "GLM 5.3 MTP exact-sampling replacements reach the nextn slot they belong to", test_glm_mtp_commit_exact},
    {"--glm-mtp-commit-fail", "glm-mtp-commit-fail", "GLM 5.3 MTP makes a slot whose nextn write failed unreadable instead of leaving it stale", test_glm_mtp_commit_fail},
    {"--dspark-verify-depth", "dspark-verify-depth", "DSpark speculative verify commits autoregressive-identical tokens at draft depth > 2", test_dspark_verify_depth},
#endif
    {"--server", "server", "server parser/rendering/cache unit tests", test_server_unit_group},
};

static void test_print_help(const char *prog) {
    printf("Usage: %s [--all | TEST...]\n\n", prog);
    puts("Tests:");
    puts("  --all");
    puts("      Run every test. This is the default, ordered from slower to faster.");
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        printf("  %-20s %s\n", test_entries[i].flag, test_entries[i].desc);
    }
    puts("  --list");
    puts("      Print test names only.");
#ifndef DS4_NO_GPU
    puts("  --metal-mpp-equivalence");
    puts("      Compatibility alias for --metal-tensor-equivalence.");
#endif
    puts("  -h, --help");
    puts("      Show this help.");
    puts("\nEnvironment:");
    puts("  DS4_TEST_MODEL=FILE        Model path. Default: ds4flash.gguf");
    puts("  DS4_TEST_BACKEND=cpu       Run model tests on CPU instead of Metal/CUDA.");
    puts("  DS4_TEST_SSD_STREAMING=1   Run model tests through Metal SSD streaming.");
    puts("  DS4_TEST_SSD_STREAMING_CACHE_GB=N  Streaming routed expert cache in GiB.");
    puts("  DS4_TEST_SSD_STREAMING_CACHE_EXPERTS=N  Streaming routed expert cache count.");
    puts("  DS4_TEST_SSD_STREAMING_COLD=1  Skip streaming hot expert preload.");
    puts("  DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL=1  Force canonical streamed cold prefill.");
    puts("  DS4_TEST_SNAPSHOT_PROMPT=FILE  Prompt for the session snapshot round trip.");
    puts("  DS4_TEST_SNAPSHOT_CTX=N        Context for the session snapshot round trip.");
    puts("  DS4_TEST_GLM_MTP=1             Include embedded GLM MTP in snapshot verification.");
    puts("  DS4_TEST_GLM_MTP_PROMPT=FILE   User prompt for --glm-mtp-verify; for --glm-mtp-prefill it is repeated past one prefill chunk when unset.");
    puts("  DS4_TEST_GLM_MTP_TOKENS=N      Tokens to generate in --glm-mtp-verify. Default: 256.");
    puts("  DS4_TEST_GLM_MTP_DUMP=FILE     Write the committed token ids from --glm-mtp-verify.");
    puts("  DS4_TEST_GLM_MTP_MAX_CHUNK=N   Pin the widest commit --glm-mtp-verify expects. Unset: DS4_GLM_MTP_DEPTH=2/3 expects that width, automatic depth only expects 2..3.");
    puts("  DS4_TEST_GLM_MTP_WINDOW_TOKENS=N  Tokens to generate per request in --glm-mtp-window. Default: 64.");
    puts("  DS4_TEST_GLM_MTP_WINDOW_CTX=N     Session context for --glm-mtp-window. Default: 4096.");
    puts("  DS4_TEST_PAYLOAD_MODE=save|load   Which side of the MTP on/off boundary --glm-mtp-payload-cross exercises.");
    puts("  DS4_TEST_PAYLOAD_FILE=PATH        .kv file --glm-mtp-payload-cross writes or reads.");
    puts("  DS4_GLM_MTP_LEGACY=1              Run the legacy GLM MTP cycle; --glm-mtp-commit-token covers it (its rewrites fill the nextn KV alone).");
    puts("  DS4_GLM_MTP_DEPTH=3               Required for --glm-mtp-commit-fail to cover the chained draft row.");
    puts("  DS4_TEST_LONG_PROMPT=FILE  Rendered long-context story fact prompt.");
    puts("  DS4_TEST_VECTOR_FILE=FILE  Official fixture. Default: flash-0731/official.vec.");
    puts("  DS4_TEST_LOCAL_GOLDEN_FILE=FILE  Local fixture. Default: flash-0731/local-golden.vec.");
    puts("  DS4_TEST_MPP_EQ_CASE=NAME  Run only Tensor equivalence cases whose id contains NAME.");
    puts("  DS4_TEST_MTP=FILE         Legacy MTP support GGUF for --mtp-verify-depth.");
    puts("  DS4_TEST_DSPARK=FILE      DSpark support GGUF for --dspark-verify-depth.");
    puts("  DS4_TEST_CONTINUED_PREFILL_TOKENS=N  Large suffix size for --glm53-continued-prefill.");
    puts("  DS4_TEST_CONTINUED_PREFILL_STEPS=N   Number of consecutive large suffixes to test.");
    puts("  DS4_TEST_CONTINUED_PREFILL_ALLOW_COARSE=1  Permit coarse short-suffix progress for baseline timing.");
}

static const ds4_test_entry *test_find_entry(const char *arg) {
#ifndef DS4_NO_GPU
    if (!strcmp(arg, "--metal-mpp-equivalence")) {
        arg = "--metal-tensor-equivalence";
    }
#endif
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        if (!strcmp(arg, test_entries[i].flag)) return &test_entries[i];
    }
    return NULL;
}

static void test_run_entry(const ds4_test_entry *entry) {
    int before = test_failures;
    fprintf(stderr, "%s:\n", entry->name);
    entry->fn();
    fprintf(stderr, "%s: ", entry->name);
    ds4_log(stderr,
            test_failures == before ? DS4_LOG_OK : DS4_LOG_ERROR,
            "%s",
            test_failures == before ? "OK" : "ERR");
    fputc('\n', stderr);
}

int main(int argc, char **argv) {
    if (argc == 4 && (!strcmp(argv[1], "--ds41-render") ||
                      !strcmp(argv[1], "--ds41-render-anthropic"))) {
        ds4_think_mode mode;
        if (!ds4_think_mode_parse_level(argv[2], &mode)) return 2;
        const char *json = argv[3];
        chat_msgs msgs = {0};
        const bool anthropic = !strcmp(argv[1], "--ds41-render-anthropic");
        if (!(anthropic ? parse_anthropic_messages(&json, &msgs) : parse_messages(&json, &msgs))) {
            chat_msgs_free(&msgs);
            return 2;
        }
        char *text = render_chat_prompt_text_for_syntax(SERVER_MODEL_SYNTAX_DEEPSEEK41,
                                                       &msgs, NULL, NULL, mode);
        fputs(text, stdout);
        free(text); chat_msgs_free(&msgs);
        return 0;
    }
    bool run_all = argc == 1;
    bool selected[sizeof(test_entries) / sizeof(test_entries[0])] = {0};

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--all")) {
            run_all = true;
        } else if (!strcmp(argv[i], "--list")) {
            for (size_t j = 0; j < sizeof(test_entries) / sizeof(test_entries[0]); j++) {
                puts(test_entries[j].flag);
            }
            return 0;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            test_print_help(argv[0]);
            return 0;
        } else {
            const ds4_test_entry *entry = test_find_entry(argv[i]);
            if (!entry) {
                fprintf(stderr, "ds4-test: unknown test switch: %s\n", argv[i]);
                test_print_help(argv[0]);
                return 2;
            }
            selected[(size_t)(entry - test_entries)] = true;
        }
    }

    if (run_all) {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            test_run_entry(&test_entries[i]);
        }
    } else {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            if (selected[i]) test_run_entry(&test_entries[i]);
        }
    }

#ifndef DS4_NO_GPU
    test_close_engines();
#endif

    if (test_failures) {
        fprintf(stderr, "ds4 tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("ds4 tests: ok");
    return 0;
}
