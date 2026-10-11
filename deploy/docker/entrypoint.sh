#!/bin/sh
# ProximaDB container entrypoint.
#
# Before launching the server, optionally refreshes the tier
# configuration from an operator-supplied URL. The fetched JSON is
# atomic-replaced into /config/tier-config.json on success; failures
# fall back to whatever was baked into the image at build time. So:
#
#   - Air-gapped deployments work as before (use the baked file).
#   - Online deployments get fresh tier config on every restart.
#   - Network failures during boot don't crash the pod.
#
# To disable the remote fetch entirely (use baked file only), set
# PROXIMADB_TIER_CONFIG_URL=''. To use a custom URL (your own R2 / S3 /
# CDN / config server), set it to that URL. The schema the URL must
# serve is documented in config/TIER_CONFIG.md.
#
# Backward compatibility: the legacy ANVAIOPS_PRICING_URL +
# /config/pricing.json variables are still honored if the new
# PROXIMADB_TIER_CONFIG_* variables are unset. The engine itself also
# falls back to reading /config/pricing.json at startup if
# /config/tier-config.json is absent (see src/catalog/tenant_tier.rs
# resolve_tier_config_source). This keeps existing AnvaiOps deployments
# working during the migration window. The legacy code path is
# scheduled for removal in the next major version.

set -eu

# ─── New canonical variables (operator-neutral) ─────────────────────────────
TIER_CONFIG_URL="${PROXIMADB_TIER_CONFIG_URL-}"
TIER_CONFIG_PATH="${PROXIMADB_TIER_CONFIG_PATH:-/config/tier-config.json}"
FETCH_TIMEOUT_SECS="${PROXIMADB_TIER_CONFIG_FETCH_TIMEOUT:-10}"
FETCH_RETRIES="${PROXIMADB_TIER_CONFIG_FETCH_RETRIES:-2}"

# ─── Legacy variables (deprecated; honored for backward compatibility) ──────
LEGACY_URL="${ANVAIOPS_PRICING_URL-}"
LEGACY_PATH="${ANVAIOPS_PRICING_PATH:-/config/pricing.json}"

# If the new variable is unset AND the legacy variable is set, use the
# legacy values + write a deprecation warning so operators see it in the
# logs and migrate at their own pace.
if [ -z "${TIER_CONFIG_URL+x}" ] && [ -n "${LEGACY_URL}" ]; then
    echo "[entrypoint] WARN: ANVAIOPS_PRICING_URL is deprecated; set PROXIMADB_TIER_CONFIG_URL instead." >&2
    echo "[entrypoint] WARN: see config/TIER_CONFIG.md for the new schema; legacy path will be removed in the next major release." >&2
    TIER_CONFIG_URL="${LEGACY_URL}"
    TIER_CONFIG_PATH="${LEGACY_PATH}"
fi

# ─── Embedding model fetch (optional; gated) ────────────────────────────────
# The engine loads the ONNX model + tokenizer from disk and FAILS LOUD if they
# are absent — it cannot download them itself (see crates/modalities/
# proximadb-embedding/src/models/bge.rs). So, exactly like the tier config
# above, we optionally refresh them here before exec. Unset URLs => no-op; the
# baked-in copies (present in the :*-full image) remain the offline fallback so
# a network blip never crashes the pod. The minimal/default image has no models
# and no onnx feature, so leaving these unset is the correct no-op there.
EMBED_MODEL_DIR="${PROXIMADB_EMBED_MODEL_DIR:-/var/lib/proximadb/models}"
MODEL_ONNX_URL="${PROXIMADB_MODEL_ONNX_URL-}"
MODEL_ONNX_FILE="${PROXIMADB_MODEL_ONNX_FILE:-bge-small-en-v1.5.onnx}"
MODEL_ONNX_SHA256="${PROXIMADB_MODEL_ONNX_SHA256-}"
MODEL_TOKENIZER_URL="${PROXIMADB_MODEL_TOKENIZER_URL-}"
MODEL_TOKENIZER_SHA256="${PROXIMADB_MODEL_TOKENIZER_SHA256-}"
MODEL_FETCH_TIMEOUT="${PROXIMADB_MODEL_FETCH_TIMEOUT:-120}"
MODEL_FETCH_RETRIES="${PROXIMADB_MODEL_FETCH_RETRIES:-2}"

refresh_tier_config() {
    if [ -z "${TIER_CONFIG_URL}" ]; then
        echo "[entrypoint] No tier config URL set; using baked-in ${TIER_CONFIG_PATH}"
        return 0
    fi

    echo "[entrypoint] fetching tier config from ${TIER_CONFIG_URL}..."

    tmpfile="$(mktemp)"

    if ! curl --fail --silent --show-error --location \
              --max-time "${FETCH_TIMEOUT_SECS}" \
              --retry "${FETCH_RETRIES}" --retry-delay 2 \
              -H "User-Agent: proximadb-entrypoint/1.0" \
              "${TIER_CONFIG_URL}" -o "${tmpfile}"; then
        echo "[entrypoint] WARN: tier config fetch from ${TIER_CONFIG_URL} failed; using baked-in ${TIER_CONFIG_PATH}" >&2
        rm -f "${tmpfile}"
        return 0
    fi

    # Sanity check: refuse to install a non-JSON / empty body. A bad
    # response is worse than a stale one — the baked-in file is at
    # least the version this image was built against.
    if [ ! -s "${tmpfile}" ]; then
        echo "[entrypoint] WARN: tier config fetch returned empty body; using baked-in ${TIER_CONFIG_PATH}" >&2
        rm -f "${tmpfile}"
        return 0
    fi
    if ! head -c 1 "${tmpfile}" | grep -q '{'; then
        echo "[entrypoint] WARN: tier config fetch returned non-JSON body; using baked-in ${TIER_CONFIG_PATH}" >&2
        rm -f "${tmpfile}"
        return 0
    fi

    # Atomic replace: write tmp first, then rename. Prevents the
    # server reading a half-written file if it tries to load tier
    # config mid-update on a hot reload.
    mv "${tmpfile}" "${TIER_CONFIG_PATH}"
    chmod 0644 "${TIER_CONFIG_PATH}"
    echo "[entrypoint] installed fresh tier config at ${TIER_CONFIG_PATH} ($(wc -c < "${TIER_CONFIG_PATH}") bytes)"
}

# Fetch one model file: $1=url $2=dest-path $3=expected-sha256 (may be empty).
# On ANY failure the baked-in file (if present) is left untouched and we return
# 0 — a model fetch must never crash the pod (mirrors the tier-config policy).
fetch_model_file() {
    _url="$1"; _dest="$2"; _sha="$3"
    [ -z "${_url}" ] && return 0
    echo "[entrypoint] fetching model ${_dest} from ${_url}..."
    _tmp="$(mktemp)"
    if ! curl --fail --silent --show-error --location \
              --max-time "${MODEL_FETCH_TIMEOUT}" \
              --retry "${MODEL_FETCH_RETRIES}" --retry-delay 3 \
              -H "User-Agent: proximadb-entrypoint/1.0" \
              "${_url}" -o "${_tmp}"; then
        echo "[entrypoint] WARN: model fetch from ${_url} failed; keeping baked-in ${_dest}" >&2
        rm -f "${_tmp}"; return 0
    fi
    if [ ! -s "${_tmp}" ]; then
        echo "[entrypoint] WARN: model fetch returned empty body; keeping baked-in ${_dest}" >&2
        rm -f "${_tmp}"; return 0
    fi
    # Optional integrity pin (ADR-0016): if a sha256 is supplied, a mismatch is
    # worse than a stale-but-known baked-in file, so reject the download.
    if [ -n "${_sha}" ]; then
        _got="$(sha256sum "${_tmp}" | cut -d' ' -f1)"
        if [ "${_got}" != "${_sha}" ]; then
            echo "[entrypoint] WARN: sha256 mismatch for ${_dest} (want ${_sha}, got ${_got}); keeping baked-in" >&2
            rm -f "${_tmp}"; return 0
        fi
    fi
    mkdir -p "$(dirname "${_dest}")"
    mv "${_tmp}" "${_dest}"
    chmod 0644 "${_dest}"
    echo "[entrypoint] installed ${_dest} ($(wc -c < "${_dest}") bytes)"
}

refresh_models() {
    if [ -z "${MODEL_ONNX_URL}" ] && [ -z "${MODEL_TOKENIZER_URL}" ]; then
        echo "[entrypoint] no model fetch URL set; using baked-in models in ${EMBED_MODEL_DIR} (if present)"
        return 0
    fi
    mkdir -p "${EMBED_MODEL_DIR}"
    fetch_model_file "${MODEL_ONNX_URL}" "${EMBED_MODEL_DIR}/${MODEL_ONNX_FILE}" "${MODEL_ONNX_SHA256}"
    fetch_model_file "${MODEL_TOKENIZER_URL}" "${EMBED_MODEL_DIR}/tokenizer.json" "${MODEL_TOKENIZER_SHA256}"
}

# ─── Bind address (container reachability) ──────────────────────────────────
# The shipped config/config.toml has `bind_address = "127.0.0.1"`, which is the
# right default for a developer running the binary on a laptop and the WRONG one
# for a container: a loopback listener inside a container namespace is
# unreachable through `docker run -p`, so the published port accepts nothing.
#
# This was invisible because the image's own HEALTHCHECK curls localhost:5678
# from INSIDE the container, where loopback works. The in-container healthcheck
# is structurally incapable of catching it; CI's host-side
# `docker run -p 5678:5678` + `curl localhost:5678/health` is what failed, after
# the server had already logged "started successfully on 127.0.0.1:5678".
#
# So normalise it here, where we know we are in a container. In-process defaults
# stay untouched (the binary on a laptop still binds loopback), and an operator
# can still pin it: PROXIMADB_BIND_ADDRESS=127.0.0.1 restores the old behaviour,
# and any other value is honoured verbatim.
BIND_ADDRESS="${PROXIMADB_BIND_ADDRESS:-0.0.0.0}"
# Set by normalize_bind_address when it cannot write the config in place; the
# exec below then points the server at this copy instead.
NORMALIZED_CONFIG=""

# The value lands in a sed replacement, where `&` and `\` are metacharacters and
# an embedded `"` or newline can append a whole config line. Verified: a crafted
# value inserted a `node_id` line into the config. An operator is not an attacker,
# but a config that silently gains a setting is a real bug -- so accept only what
# a bind address can be: IPv4/IPv6 literals and hostname characters.
case "${BIND_ADDRESS}" in
    *[!0-9a-zA-Z.:_-]*|"")
        echo "[entrypoint] WARN: PROXIMADB_BIND_ADDRESS='${BIND_ADDRESS}' is not a plausible bind address; ignoring it" >&2
        BIND_ADDRESS="0.0.0.0"
        ;;
esac

normalize_bind_address() {
    config_path=""
    want_config=0
    for arg in "$@"; do
        if [ "${want_config}" -eq 1 ]; then
            config_path="${arg}"
            want_config=0
            continue
        fi
        case "${arg}" in
            --config | -c) want_config=1 ;;
            --config=*) config_path="${arg#--config=}" ;;
            # clap also accepts `-c=path` and `-cpath` for a short option with a
            # value. Missing these meant "no --config file found" and a silent
            # loopback bind.
            -c=*) config_path="${arg#-c=}" ;;
            -c?*) config_path="${arg#-c}" ;;
        esac
    done

    if [ -z "${config_path}" ] || [ ! -f "${config_path}" ]; then
        echo "[entrypoint] no --config file found; leaving bind_address to the binary's default" >&2
        return 0
    fi
    # `--` so a config path beginning with `-` is a FILE, not an option. Without
    # it grep/sed parsed it as a flag, reported "declares no bind_address" for a
    # file that declares one, and left the loopback bind with a reassuring log.
    if ! grep -qE -- '^[[:space:]]*bind_address[[:space:]]*=' "${config_path}"; then
        echo "[entrypoint] ${config_path} declares no bind_address; leaving it alone" >&2
        return 0
    fi

    current="$(sed -n -e 's/^[[:space:]]*bind_address[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' -- "${config_path}" | head -1)"
    if [ "${current}" = "${BIND_ADDRESS}" ]; then
        echo "[entrypoint] bind_address already ${BIND_ADDRESS}" >&2
        return 0
    fi

    # Write through the existing inode, NOT via rename. `mv` was wrong twice:
    #
    #   * rename(2) over a bind-mounted FILE fails EBUSY, and an unguarded `mv`
    #     under `set -eu` then exits before exec. deploy/docker/docker-compose.yml
    #     mounts `../../config/config.toml:/config/config.toml:ro` with
    #     `restart: unless-stopped`, so that is a permanent crash loop -- strictly
    #     worse than the loopback bind this function exists to fix.
    #   * /config/config.toml is root-owned 0644 while the server runs as uid 999
    #     (Dockerfile COPYs it after chowning /config, then USER proximadb). With a
    #     TTY, GNU mv prompts "overriding mode 0644" and blocks on stdin forever.
    #     docs/04-operations/deployment.adoc documents `docker run -it`, and CI
    #     uses `-d`, so CI cannot catch that -- the same blindness the HEALTHCHECK
    #     comment warns about, one layer up.
    #
    # `cat > file` needs only write permission on the file's directory-resolved
    # inode, never a rename or a mode change: it works through a rw bind mount,
    # fails inside the `if` on a :ro mount (so `set -e` stays suspended and the
    # container still boots), and never prompts. The atomicity `mv` bought is not
    # needed here -- nothing has read the config yet, the server is exec'd after.
    # The scratch file goes in TMPDIR, NOT next to the config. Writing it as
    # "${config_path}.bind.$$" meant a read-only /config (a k8s ConfigMap mount,
    # or `--read-only` rootfs) failed at the FIRST write, so the not-writable
    # fallback below was never even attempted -- the one shape that most needs
    # it. mktemp also replaces a fixed, predictable path that was a
    # symlink-follow truncate (CWE-61) in a 1777 /tmp, and 0600 keeps a config
    # holding API keys and a JWT secret from being cloned world-readable.
    tmp="$(mktemp "${TMPDIR:-/tmp}/proximadb-config.XXXXXX" 2>/dev/null)" || tmp=""
    if [ -z "${tmp}" ]; then
        # TMPDIR is preferred because it survives a read-only /config, but it is
        # not always the writable one: `--read-only` rootfs with a WRITABLE
        # /config bind mount and no tmpfs is the mirror image, and the earlier
        # next-to-the-config placement handled exactly that. Try both.
        tmp="$(mktemp "$(dirname "${config_path}")/proximadb-config.XXXXXX" 2>/dev/null)" || tmp=""
    fi
    if [ -z "${tmp}" ]; then
        echo "[entrypoint] WARN: no writable scratch dir (tried TMPDIR=${TMPDIR:-/tmp} and $(dirname "${config_path}")); leaving bind_address ${current}. The container will NOT be reachable on its published ports." >&2
        return 0
    fi
    chmod 0600 "${tmp}" 2>/dev/null || true

    if ! sed "s|^\([[:space:]]*bind_address[[:space:]]*=[[:space:]]*\)\"[^\"]*\"|\1\"${BIND_ADDRESS}\"|" \
        -- "${config_path}" > "${tmp}" 2>/dev/null || [ ! -s "${tmp}" ]; then
        rm -f "${tmp}"
        echo "[entrypoint] WARN: could not normalize ${config_path}; leaving ${current}. The container will NOT be reachable on its published ports." >&2
        return 0
    fi

    # Braces, not just a command redirect: `cat x > y 2>/dev/null` lets the
    # SHELL's own "cannot create y: Permission denied" through, which lands in
    # the CI log head looking like a crash. Grouping redirects that too.
    if { cat "${tmp}" > "${config_path}"; } 2>/dev/null; then
        rm -f "${tmp}"
        echo "[entrypoint] bind_address ${current} -> ${BIND_ADDRESS} (container reachability; set PROXIMADB_BIND_ADDRESS to override)" >&2
    else
        # In-place failed. Two real causes, both observed: the file is root-owned
        # 0644 while we run as uid 999 (the Dockerfile COPY landed before the
        # chown -- fixed there, but a bind-mounted config can reproduce it), and
        # docker-compose.yml mounts the config `:ro`.
        #
        # Do NOT fall through to booting with the loopback bind. That is the
        # defect this function exists to fix and it fails as an *unreachable
        # container* -- `-p` cannot reach a loopback listener, the in-container
        # HEALTHCHECK is structurally blind to it, and the CI health check's
        # `docker logs | tail -100` scrolls this warning away. It cost one full
        # promotion round to find. Write the normalized copy somewhere writable
        # and point the server at that instead.
        # The normalized file already exists and is already correct -- serve it
        # directly rather than copying it again to a second path.
        NORMALIZED_CONFIG="${tmp}"
        echo "[entrypoint] ${config_path} is not writable; using a normalized copy at ${tmp} (bind_address ${current} -> ${BIND_ADDRESS})" >&2
    fi
}

refresh_tier_config
refresh_models
normalize_bind_address "$@"

# Hand off to the server. `exec` so the server becomes PID 1 (signal
# handling, OOM, healthchecks all work correctly).
if [ -n "${NORMALIZED_CONFIG}" ]; then
    # Rebuild the argument list with the --config value replaced, preserving
    # every other argument and its order. Done inline rather than in a function
    # because `set --` only rewrites the positional parameters of its own shell,
    # and echoing the list back would corrupt any argument containing a space.
    n=$#
    i=0
    want=0
    while [ "${i}" -lt "${n}" ]; do
        arg="$1"
        shift
        i=$((i + 1))
        if [ "${want}" -eq 1 ]; then
            want=0
            set -- "$@" "${NORMALIZED_CONFIG}"
            continue
        fi
        case "${arg}" in
            --config | -c)
                want=1
                set -- "$@" "${arg}"
                ;;
            --config=*)
                set -- "$@" "--config=${NORMALIZED_CONFIG}"
                ;;
            # Must mirror normalize_bind_address's arms exactly. They diverged
            # once: detection accepted these two, the rebuild did not, so the
            # server got the ORIGINAL path while the log below announced the
            # normalized one.
            -c=*)
                set -- "$@" "-c=${NORMALIZED_CONFIG}"
                ;;
            -c?*)
                set -- "$@" "-c${NORMALIZED_CONFIG}"
                ;;
            *)
                set -- "$@" "${arg}"
                ;;
        esac
    done
    # $* is fine for a LOG line (the exec'd list above is byte-exact); it only
    # renders ambiguously for an argument containing whitespace. The comment
    # above is about rebuilding the list, not about printing it.
    echo "[entrypoint] serving with the normalized config: $*" >&2
fi

exec /usr/local/bin/proximadb-server "$@"
