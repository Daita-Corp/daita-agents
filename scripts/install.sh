#!/usr/bin/env bash
# Daita managed installer. The release team replaces every UNRESOLVED literal
# with reviewed immutable artifact evidence before publishing these bytes.

set -euo pipefail
IFS=$'\n\t'
umask 077

readonly OWNER_MARKER="daita-managed-install-v1"
readonly LAUNCHER_MARKER="# DAITA-MANAGED-LAUNCHER:v1"
readonly PATH_START="# >>> Daita managed PATH >>>"
readonly PATH_END="# <<< Daita managed PATH <<<"

readonly INSTALLER_VERSION="UNRESOLVED_INSTALLER_VERSION"
readonly RELEASE_SEQUENCE="UNRESOLVED_RELEASE_SEQUENCE"
readonly DAITA_VERSION="1.0.0"
readonly WHEEL_FILENAME="daita_agents-1.0.0-py3-none-any.whl"
readonly WHEEL_URL="UNRESOLVED_WHEEL_URL"
readonly WHEEL_SHA256="UNRESOLVED_WHEEL_SHA256"
readonly WHEEL_REQUIRES_PYTHON="<3.13,>=3.11"
readonly UV_VERSION="UNRESOLVED_UV_VERSION"
readonly PYTHON_REQUEST="UNRESOLVED_PYTHON_REQUEST"

readonly UV_DARWIN_ARM64_ARCHIVE="UNRESOLVED_UV_DARWIN_ARM64_ARCHIVE"
readonly UV_DARWIN_ARM64_MEMBER="UNRESOLVED_UV_DARWIN_ARM64_MEMBER"
readonly UV_DARWIN_ARM64_URL="UNRESOLVED_UV_DARWIN_ARM64_URL"
readonly UV_DARWIN_ARM64_SHA256="UNRESOLVED_UV_DARWIN_ARM64_SHA256"
readonly PYTHON_DARWIN_ARM64_IDENTITY="UNRESOLVED_PYTHON_DARWIN_ARM64_IDENTITY"

readonly UV_DARWIN_X86_64_ARCHIVE="UNRESOLVED_UV_DARWIN_X86_64_ARCHIVE"
readonly UV_DARWIN_X86_64_MEMBER="UNRESOLVED_UV_DARWIN_X86_64_MEMBER"
readonly UV_DARWIN_X86_64_URL="UNRESOLVED_UV_DARWIN_X86_64_URL"
readonly UV_DARWIN_X86_64_SHA256="UNRESOLVED_UV_DARWIN_X86_64_SHA256"
readonly PYTHON_DARWIN_X86_64_IDENTITY="UNRESOLVED_PYTHON_DARWIN_X86_64_IDENTITY"

readonly UV_LINUX_ARM64_ARCHIVE="UNRESOLVED_UV_LINUX_ARM64_ARCHIVE"
readonly UV_LINUX_ARM64_MEMBER="UNRESOLVED_UV_LINUX_ARM64_MEMBER"
readonly UV_LINUX_ARM64_URL="UNRESOLVED_UV_LINUX_ARM64_URL"
readonly UV_LINUX_ARM64_SHA256="UNRESOLVED_UV_LINUX_ARM64_SHA256"
readonly PYTHON_LINUX_ARM64_IDENTITY="UNRESOLVED_PYTHON_LINUX_ARM64_IDENTITY"

readonly UV_LINUX_X86_64_ARCHIVE="UNRESOLVED_UV_LINUX_X86_64_ARCHIVE"
readonly UV_LINUX_X86_64_MEMBER="UNRESOLVED_UV_LINUX_X86_64_MEMBER"
readonly UV_LINUX_X86_64_URL="UNRESOLVED_UV_LINUX_X86_64_URL"
readonly UV_LINUX_X86_64_SHA256="UNRESOLVED_UV_LINUX_X86_64_SHA256"
readonly PYTHON_LINUX_X86_64_IDENTITY="UNRESOLVED_PYTHON_LINUX_X86_64_IDENTITY"

ACTION="install"
ACTION_COUNT=0
DRY_RUN=0
NO_ONBOARD=0
NO_MODIFY_PATH=0
LOCK_HELD=0
STAGE=""
NEW_GENERATION=""
ACTIVATION_STARTED=0
ACTIVATION_COMMITTED=0
OLD_CURRENT=""
OLD_PREVIOUS=""
NEW_LAUNCHER_PUBLISHED=0
OLD_CURRENT_VERIFIED=0
UV_BIN=""
PYTHON_BIN=""
GENERATION_STAGE=""
PYTHON_INSTALL_DIR=""
UV_CACHE_SCOPE=""
PENDING_UV_ROOT=""
PENDING_PYTHON_ROOT=""

HOME_REAL=""
LOCAL_ROOT=""
MANAGED_ROOT=""
PUBLIC_BIN_DIR=""
PUBLIC_LAUNCHER=""
INSTALL_STATE=""
GENERATIONS_DIR=""
STAGING_DIR=""
LOCK_DIR=""
UV_ROOT=""
PYTHON_ROOT=""
CACHE_ROOT=""

TARGET=""
UV_ARCHIVE=""
UV_ARCHIVE_MEMBER=""
UV_ARCHIVE_URL=""
UV_ARCHIVE_SHA256=""
PYTHON_IDENTITY=""

usage() {
    cat <<'EOF'
Usage: install.sh [ACTION] [OPTIONS]

Install or forward-upgrade Daita using the exact artifacts pinned in this
reviewed installer. Installation decisions are never read from stdin.

Actions (choose at most one):
  --verify          Verify the managed installation without changing it
  --repair          Build and activate a fresh generation of the pinned release
  --rollback        Reactivate the recorded previous binary generation
  --uninstall       Remove only validated installer-owned application files

Options:
  --dry-run         Report resolved paths, artifacts, and mutations; write nothing
  --no-onboard      Do not launch Daita after a successful install
  --no-modify-path  Do not edit a supported shell startup file
  --version         Print pinned installer, Daita, and uv versions; write nothing
  -h, --help        Show this help; write nothing

Close every running Daita process before install, upgrade, repair, rollback,
or uninstall. --verify is read-only and may run while Daita is active.
EOF
}

version() {
    printf 'Daita installer %s (release sequence %s); Daita %s; uv %s\n' \
        "$INSTALLER_VERSION" "$RELEASE_SEQUENCE" "$DAITA_VERSION" "$UV_VERSION"
}

say() {
    printf '%s\n' "$*"
}

warn() {
    printf 'warning: %s\n' "$*" >&2
}

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

usage_error() {
    printf 'error: %s\n' "$*" >&2
    printf "Try 'install.sh --help' for more information.\n" >&2
    exit 2
}

test_failpoint() {
    case "$INSTALLER_VERSION" in
        *-fixture)
            if [[ "${DAITA_INSTALLER_TEST_FAILPOINT:-}" == "$1" ]]; then
                fail "deterministic fixture failure at $1"
            fi
            ;;
    esac
}

set_action() {
    ACTION="$1"
    ACTION_COUNT=$((ACTION_COUNT + 1))
}

parse_arguments() {
    while (($#)); do
        case "$1" in
            --verify) set_action "verify" ;;
            --repair) set_action "repair" ;;
            --rollback) set_action "rollback" ;;
            --uninstall) set_action "uninstall" ;;
            --dry-run) DRY_RUN=1 ;;
            --no-onboard) NO_ONBOARD=1 ;;
            --no-modify-path) NO_MODIFY_PATH=1 ;;
            --version)
                if (($# != 1 || ACTION_COUNT != 0 || DRY_RUN != 0 || NO_ONBOARD != 0 || NO_MODIFY_PATH != 0)); then
                    usage_error "--version cannot be combined with another argument"
                fi
                version
                exit 0
                ;;
            -h|--help)
                if (($# != 1 || ACTION_COUNT != 0 || DRY_RUN != 0 || NO_ONBOARD != 0 || NO_MODIFY_PATH != 0)); then
                    usage_error "--help cannot be combined with another argument"
                fi
                usage
                exit 0
                ;;
            --) usage_error "positional arguments are not supported" ;;
            -*) usage_error "unknown argument: $1" ;;
            *) usage_error "unexpected positional argument: $1" ;;
        esac
        shift
    done
    if ((ACTION_COUNT > 1)); then
        usage_error "lifecycle actions conflict; choose exactly one"
    fi
}

on_signal_int() {
    exit 130
}

on_signal_term() {
    exit 143
}

safe_remove_tree() {
    local target="$1"
    [[ -n "$MANAGED_ROOT" && -n "$target" ]] || fail "refusing an empty removal target"
    case "$target" in
        "$MANAGED_ROOT"/staging/*|"$MANAGED_ROOT"/generations/*|"$MANAGED_ROOT"/cache)
            ;;
        *) fail "refusing unsafe managed removal target: $target" ;;
    esac
    [[ "$target" != "/" && "$target" != "$HOME_REAL" && "$target" != "$MANAGED_ROOT" ]] || \
        fail "refusing broad managed removal target: $target"
    rm -rf -- "$target"
}

atomic_symlink() {
    local target="$1"
    local link="$2"
    local temporary="${link}.tmp.$$"
    rm -f -- "$temporary"
    ln -s -- "$target" "$temporary"
    if [[ "$(uname -s)" == "Darwin" ]]; then
        mv -fh -- "$temporary" "$link"
    else
        mv -fT -- "$temporary" "$link"
    fi
}

restore_link() {
    local link="$1"
    local target="$2"
    if [[ -n "$target" ]]; then
        atomic_symlink "$target" "$link"
    elif [[ -L "$link" ]]; then
        rm -f -- "$link"
    fi
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if ((ACTIVATION_STARTED == 1 && ACTIVATION_COMMITTED == 0)); then
        restore_link "$MANAGED_ROOT/current" "$OLD_CURRENT" || true
        restore_link "$MANAGED_ROOT/previous" "$OLD_PREVIOUS" || true
        if ((NEW_LAUNCHER_PUBLISHED == 1)); then
            rm -f -- "$PUBLIC_LAUNCHER" || true
        fi
    fi
    if [[ -n "$STAGE" && -e "$STAGE" ]]; then
        safe_remove_tree "$STAGE" || true
    fi
    if [[ -n "$NEW_GENERATION" && -e "$NEW_GENERATION" && "$ACTIVATION_COMMITTED" == 0 ]]; then
        safe_remove_tree "$NEW_GENERATION" || true
    fi
    if ((LOCK_HELD == 1)) && [[ -d "$LOCK_DIR" ]]; then
        rm -f -- "$LOCK_DIR/pid" || true
        rmdir -- "$LOCK_DIR" 2>/dev/null || true
    fi
    exit "$status"
}

trap cleanup EXIT
trap on_signal_int INT
trap on_signal_term TERM

validate_home() {
    [[ -n "${HOME:-}" ]] || fail "HOME is required"
    [[ "$HOME" = /* && "$HOME" != "/" ]] || fail "HOME must be an absolute non-root path"
    case "$HOME" in
        *$'\n'*|*$'\r'*|*$'\t'*) fail "HOME contains unsafe control characters" ;;
    esac
    [[ -d "$HOME" ]] || fail "HOME does not exist: $HOME"
    [[ ! -L "$HOME" ]] || fail "a symlink HOME is not supported"
    HOME_REAL=$(CDPATH= cd -P -- "$HOME" && pwd -P)
    [[ "$HOME_REAL" = /* && "$HOME_REAL" != "/" ]] || fail "HOME resolved unsafely"

    LOCAL_ROOT="$HOME_REAL/.local"
    MANAGED_ROOT="$LOCAL_ROOT/share/daita"
    PUBLIC_BIN_DIR="$LOCAL_ROOT/bin"
    PUBLIC_LAUNCHER="$PUBLIC_BIN_DIR/daita"
    INSTALL_STATE="$MANAGED_ROOT/install-state"
    GENERATIONS_DIR="$MANAGED_ROOT/generations"
    STAGING_DIR="$MANAGED_ROOT/staging"
    LOCK_DIR="$INSTALL_STATE/mutation.lock"
    UV_ROOT="$MANAGED_ROOT/installer/uv/$UV_VERSION"
    PYTHON_ROOT="$MANAGED_ROOT/python"
    CACHE_ROOT="$MANAGED_ROOT/cache"
    PYTHON_INSTALL_DIR="$PYTHON_ROOT"
    UV_CACHE_SCOPE="$CACHE_ROOT"

    [[ "$MANAGED_ROOT" == "$HOME_REAL/.local/share/daita" ]] || fail "managed root escaped HOME"
    local path
    for path in "$LOCAL_ROOT" "$LOCAL_ROOT/share" "$MANAGED_ROOT" "$PUBLIC_BIN_DIR"; do
        if [[ -L "$path" ]]; then
            fail "managed installation path must not be a symlink: $path"
        fi
    done
}

reject_elevated_execution() {
    if [[ "${EUID:-$(id -u)}" == "0" || -n "${SUDO_USER:-}" ]]; then
        fail "run the Daita installer as the ordinary target user, not through sudo or as root"
    fi
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command is unavailable: $1"
}

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$path" | awk '{print $1}'
    else
        fail "a SHA-256 utility (sha256sum or shasum) is required"
    fi
}

sha256_stdin() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 | awk '{print $1}'
    else
        fail "a SHA-256 utility (sha256sum or shasum) is required"
    fi
}

validate_sha256_literal() {
    local value="$1"
    [[ "$value" =~ ^[0-9a-f]{64}$ ]] || fail "release SHA-256 literal is unresolved or invalid"
}

validate_url_literal() {
    local value="$1"
    [[ "$value" == https://* ]] || fail "release URL must be immutable HTTPS"
    case "$value" in
        *UNRESOLVED*|*latest*|*LATEST*) fail "release URL is unresolved or mutable: $value" ;;
    esac
}

validate_release_literals() {
    local value
    for value in "$INSTALLER_VERSION" "$RELEASE_SEQUENCE" "$UV_VERSION" "$PYTHON_REQUEST" \
        "$PYTHON_IDENTITY" "$UV_ARCHIVE" "$UV_ARCHIVE_MEMBER"; do
        [[ "$value" != *UNRESOLVED* && -n "$value" ]] || \
            fail "this installer is not release-ready: required artifact literals are unresolved"
    done
    [[ "$RELEASE_SEQUENCE" =~ ^[1-9][0-9]*$ ]] || fail "release sequence must be a positive integer"
    validate_url_literal "$WHEEL_URL"
    validate_url_literal "$UV_ARCHIVE_URL"
    validate_sha256_literal "$WHEEL_SHA256"
    validate_sha256_literal "$UV_ARCHIVE_SHA256"
    local release_value
    for release_value in \
        "$UV_DARWIN_ARM64_ARCHIVE" "$UV_DARWIN_ARM64_MEMBER" "$UV_DARWIN_ARM64_URL" \
        "$UV_DARWIN_ARM64_SHA256" "$PYTHON_DARWIN_ARM64_IDENTITY" \
        "$UV_DARWIN_X86_64_ARCHIVE" "$UV_DARWIN_X86_64_MEMBER" "$UV_DARWIN_X86_64_URL" \
        "$UV_DARWIN_X86_64_SHA256" "$PYTHON_DARWIN_X86_64_IDENTITY" \
        "$UV_LINUX_ARM64_ARCHIVE" "$UV_LINUX_ARM64_MEMBER" "$UV_LINUX_ARM64_URL" \
        "$UV_LINUX_ARM64_SHA256" "$PYTHON_LINUX_ARM64_IDENTITY" \
        "$UV_LINUX_X86_64_ARCHIVE" "$UV_LINUX_X86_64_MEMBER" "$UV_LINUX_X86_64_URL" \
        "$UV_LINUX_X86_64_SHA256" "$PYTHON_LINUX_X86_64_IDENTITY"; do
        [[ "$release_value" != *UNRESOLVED* && -n "$release_value" ]] || \
            fail "this installer is not release-ready: the supported target table is incomplete"
    done
    validate_url_literal "$UV_DARWIN_ARM64_URL"
    validate_url_literal "$UV_DARWIN_X86_64_URL"
    validate_url_literal "$UV_LINUX_ARM64_URL"
    validate_url_literal "$UV_LINUX_X86_64_URL"
    validate_sha256_literal "$UV_DARWIN_ARM64_SHA256"
    validate_sha256_literal "$UV_DARWIN_X86_64_SHA256"
    validate_sha256_literal "$UV_LINUX_ARM64_SHA256"
    validate_sha256_literal "$UV_LINUX_X86_64_SHA256"
    [[ "$WHEEL_FILENAME" != */* && "$WHEEL_FILENAME" == *.whl ]] || fail "wheel filename is unsafe"
    [[ "$UV_ARCHIVE" != */* && "$UV_ARCHIVE" == *.tar.gz ]] || fail "uv archive filename is unsafe"
    [[ "$UV_ARCHIVE_MEMBER" != /* && "$UV_ARCHIVE_MEMBER" != *".."* ]] || \
        fail "uv archive member is unsafe"
    [[ "$PYTHON_REQUEST" == cpython-3.12* && "$PYTHON_REQUEST" != */* ]] || \
        fail "managed Python request is not an exact CPython 3.12 request"
    local archive_literal
    for archive_literal in \
        "$UV_DARWIN_ARM64_ARCHIVE" "$UV_DARWIN_X86_64_ARCHIVE" \
        "$UV_LINUX_ARM64_ARCHIVE" "$UV_LINUX_X86_64_ARCHIVE"; do
        [[ "$archive_literal" != */* && "$archive_literal" == *.tar.gz ]] || \
            fail "supported target contains an unsafe uv archive filename"
    done
    local member_literal
    for member_literal in \
        "$UV_DARWIN_ARM64_MEMBER" "$UV_DARWIN_X86_64_MEMBER" \
        "$UV_LINUX_ARM64_MEMBER" "$UV_LINUX_X86_64_MEMBER"; do
        [[ "$member_literal" != /* && "$member_literal" != *".."* ]] || \
            fail "supported target contains an unsafe uv archive member"
    done
    local identity_literal
    for identity_literal in \
        "$PYTHON_DARWIN_ARM64_IDENTITY" "$PYTHON_DARWIN_X86_64_IDENTITY" \
        "$PYTHON_LINUX_ARM64_IDENTITY" "$PYTHON_LINUX_X86_64_IDENTITY"; do
        [[ "$identity_literal" == cpython-3.12* && "$identity_literal" != */* && "$identity_literal" != *".."* ]] || \
            fail "supported target contains an unsafe managed Python identity"
    done
}

resolve_platform() {
    require_command uname
    local os
    local architecture
    os=$(uname -s)
    architecture=$(uname -m)
    case "$os:$architecture" in
        Darwin:arm64|Darwin:aarch64)
            TARGET="macos-arm64"
            UV_ARCHIVE="$UV_DARWIN_ARM64_ARCHIVE"
            UV_ARCHIVE_MEMBER="$UV_DARWIN_ARM64_MEMBER"
            UV_ARCHIVE_URL="$UV_DARWIN_ARM64_URL"
            UV_ARCHIVE_SHA256="$UV_DARWIN_ARM64_SHA256"
            PYTHON_IDENTITY="$PYTHON_DARWIN_ARM64_IDENTITY"
            ;;
        Darwin:x86_64|Darwin:amd64)
            TARGET="macos-x86_64"
            UV_ARCHIVE="$UV_DARWIN_X86_64_ARCHIVE"
            UV_ARCHIVE_MEMBER="$UV_DARWIN_X86_64_MEMBER"
            UV_ARCHIVE_URL="$UV_DARWIN_X86_64_URL"
            UV_ARCHIVE_SHA256="$UV_DARWIN_X86_64_SHA256"
            PYTHON_IDENTITY="$PYTHON_DARWIN_X86_64_IDENTITY"
            ;;
        Linux:aarch64|Linux:arm64)
            require_glibc
            TARGET="linux-arm64-glibc"
            UV_ARCHIVE="$UV_LINUX_ARM64_ARCHIVE"
            UV_ARCHIVE_MEMBER="$UV_LINUX_ARM64_MEMBER"
            UV_ARCHIVE_URL="$UV_LINUX_ARM64_URL"
            UV_ARCHIVE_SHA256="$UV_LINUX_ARM64_SHA256"
            PYTHON_IDENTITY="$PYTHON_LINUX_ARM64_IDENTITY"
            ;;
        Linux:x86_64|Linux:amd64)
            require_glibc
            TARGET="linux-x86_64-glibc"
            UV_ARCHIVE="$UV_LINUX_X86_64_ARCHIVE"
            UV_ARCHIVE_MEMBER="$UV_LINUX_X86_64_MEMBER"
            UV_ARCHIVE_URL="$UV_LINUX_X86_64_URL"
            UV_ARCHIVE_SHA256="$UV_LINUX_X86_64_SHA256"
            PYTHON_IDENTITY="$PYTHON_LINUX_X86_64_IDENTITY"
            ;;
        *) fail "unsupported installer target: $os $architecture" ;;
    esac
}

require_glibc() {
    local libc=""
    if command -v getconf >/dev/null 2>&1; then
        libc=$(getconf GNU_LIBC_VERSION 2>/dev/null || true)
    fi
    if [[ "$libc" != glibc\ * ]] && command -v ldd >/dev/null 2>&1; then
        libc=$(ldd --version 2>&1 | head -n 1 || true)
    fi
    case "$libc" in
        *glibc*|*"GNU libc"*|*"GNU C Library"*) return ;;
    esac
    fail "Linux managed installation currently requires glibc; musl is unsupported"
}

state_value() {
    local path="$1"
    local key="$2"
    [[ -f "$path" && ! -L "$path" ]] || return 1
    awk -F= -v wanted="$key" '$1 == wanted {print substr($0, length($1) + 2); found++} END {if (found != 1) exit 1}' "$path"
}

write_atomic_file() {
    local destination="$1"
    local source="$2"
    local temporary="${destination}.tmp.$$"
    cp -- "$source" "$temporary"
    chmod 600 "$temporary"
    mv -f -- "$temporary" "$destination"
}

validate_owner_state() {
    local owner="$INSTALL_STATE/owner"
    [[ -f "$owner" && ! -L "$owner" ]] || return 1
    [[ "$(state_value "$owner" marker)" == "$OWNER_MARKER" ]] || return 1
    [[ "$(state_value "$owner" root)" == "$MANAGED_ROOT" ]] || return 1
}

admit_managed_root() {
    if [[ -e "$MANAGED_ROOT" ]]; then
        [[ -d "$MANAGED_ROOT" && ! -L "$MANAGED_ROOT" ]] || fail "managed root is not a safe directory"
        validate_owner_state || fail "existing managed root is not owned by this Daita installer: $MANAGED_ROOT"
    fi
}

launcher_is_owned() {
    [[ -f "$PUBLIC_LAUNCHER" && ! -L "$PUBLIC_LAUNCHER" ]] || return 1
    [[ "$(head -n 2 "$PUBLIC_LAUNCHER" | tail -n 1)" == "$LAUNCHER_MARKER" ]] || return 1
    local state="$INSTALL_STATE/launcher"
    [[ -f "$state" && ! -L "$state" ]] || return 1
    local expected
    expected=$(state_value "$state" sha256) || return 1
    [[ "$(sha256_file "$PUBLIC_LAUNCHER")" == "$expected" ]] || return 1
}

reject_launcher_collision() {
    if [[ -e "$PUBLIC_LAUNCHER" || -L "$PUBLIC_LAUNCHER" ]]; then
        if ! launcher_is_owned; then
            fail "refusing to overwrite foreign or pipx-owned $PUBLIC_LAUNCHER; preserve or archive ~/.daita, explicitly remove the old application, then rerun this installer"
        fi
    fi
}

acquire_lock() {
    mkdir -p -- "$INSTALL_STATE"
    chmod 700 "$MANAGED_ROOT" "$INSTALL_STATE"
    if ! mkdir -- "$LOCK_DIR" 2>/dev/null; then
        local holder="unknown"
        if [[ -f "$LOCK_DIR/pid" ]]; then
            holder=$(head -n 1 "$LOCK_DIR/pid" 2>/dev/null || printf 'unknown')
        fi
        fail "another Daita installer mutation holds the lock (pid $holder)"
    fi
    LOCK_HELD=1
    printf '%s\n' "$$" >"$LOCK_DIR/pid"
}

download() {
    local url="$1"
    local destination="$2"
    curl --fail --silent --show-error --location \
        --proto '=https' --tlsv1.2 \
        --retry 3 --retry-delay 1 --retry-connrefused \
        --connect-timeout 10 --max-time 300 \
        --output "$destination" "$url"
}

verify_checksum() {
    local path="$1"
    local expected="$2"
    local actual
    actual=$(sha256_file "$path")
    [[ "$actual" == "$expected" ]] || fail "SHA-256 mismatch for $(basename "$path")"
}

extract_uv() {
    local archive="$1"
    local destination="$2"
    local listing="$STAGE/uv-member-listing"
    tar -tvzf "$archive" "$UV_ARCHIVE_MEMBER" >"$listing"
    [[ "$(wc -l <"$listing" | tr -d ' ')" == "1" ]] || fail "uv archive member is missing or duplicated"
    [[ "$(head -c 1 "$listing")" == "-" ]] || fail "uv archive member is not a regular file"
    tar -xOzf "$archive" "$UV_ARCHIVE_MEMBER" >"$destination"
    [[ -s "$destination" ]] || fail "uv archive produced an empty binary"
    chmod 700 "$destination"
}

prepare_uv() {
    local installed="$UV_ROOT/uv"
    local installed_state="$UV_ROOT/manifest"
    if [[ -x "$installed" && -f "$installed_state" && ! -L "$installed" ]]; then
        local recorded_archive
        local recorded_binary
        recorded_archive=$(state_value "$installed_state" archive_sha256 || true)
        recorded_binary=$(state_value "$installed_state" binary_sha256 || true)
        if [[ "$recorded_archive" == "$UV_ARCHIVE_SHA256" && -n "$recorded_binary" && "$(sha256_file "$installed")" == "$recorded_binary" ]]; then
            UV_BIN="$installed"
            return
        fi
        fail "the managed uv binary is damaged; rerun --repair after removing no files manually"
    fi

    local archive="$STAGE/$UV_ARCHIVE"
    local binary="$STAGE/uv"
    download "$UV_ARCHIVE_URL" "$archive"
    verify_checksum "$archive" "$UV_ARCHIVE_SHA256"
    extract_uv "$archive" "$binary"
    local binary_sha
    binary_sha=$(sha256_file "$binary")
    PENDING_UV_ROOT="$STAGE/uv-publish"
    mkdir -- "$PENDING_UV_ROOT"
    mv -- "$binary" "$PENDING_UV_ROOT/uv"
    {
        printf 'marker=%s\n' "$OWNER_MARKER"
        printf 'version=%s\n' "$UV_VERSION"
        printf 'target=%s\n' "$TARGET"
        printf 'archive_sha256=%s\n' "$UV_ARCHIVE_SHA256"
        printf 'binary_sha256=%s\n' "$binary_sha"
    } >"$PENDING_UV_ROOT/manifest"
    chmod 600 "$PENDING_UV_ROOT/manifest"
    UV_BIN="$PENDING_UV_ROOT/uv"
}

prepare_python() {
    local installed_python="$PYTHON_ROOT/$PYTHON_IDENTITY/bin/python3.12"
    if [[ -x "$installed_python" ]]; then
        PYTHON_INSTALL_DIR="$PYTHON_ROOT"
        PYTHON_BIN="$installed_python"
        return
    fi
    if [[ -e "$PYTHON_ROOT/$PYTHON_IDENTITY" ]]; then
        fail "the exact managed Python identity is damaged; preserve the active generation and rerun after release support review"
    fi
    PENDING_PYTHON_ROOT="$STAGE/python"
    PYTHON_INSTALL_DIR="$PENDING_PYTHON_ROOT"
    PYTHON_BIN="$PENDING_PYTHON_ROOT/$PYTHON_IDENTITY/bin/python3.12"
    if [[ -x "$PYTHON_BIN" ]]; then
        return
    fi
    mkdir -p -- "$PENDING_PYTHON_ROOT"
    scoped_uv "$UV_BIN" python install "$PYTHON_REQUEST"
    [[ -x "$PYTHON_BIN" ]] || \
        fail "uv did not resolve the exact managed Python identity: $PYTHON_IDENTITY"
    local identity
    identity=$("$PYTHON_BIN" -I -c 'import platform, sys; print(f"{sys.implementation.name}-{platform.python_version()}")')
    [[ "$identity" == cpython-3.12.* ]] || fail "managed Python is not CPython 3.12"
}

scoped_uv() {
    env \
        -u PIP_CONSTRAINT \
        -u PIP_FIND_LINKS \
        -u PIP_NO_INDEX \
        -u PIP_REQUIREMENT \
        -u PIP_TRUSTED_HOST \
        -u UV_CONSTRAINT \
        -u UV_FIND_LINKS \
        -u UV_INDEX_URL \
        -u UV_OVERRIDE \
        UV_TOOL_DIR="$GENERATION_STAGE/tool" \
        UV_TOOL_BIN_DIR="$GENERATION_STAGE/bin" \
        UV_PYTHON_INSTALL_DIR="$PYTHON_INSTALL_DIR" \
        UV_CACHE_DIR="$UV_CACHE_SCOPE" \
        UV_NO_CONFIG=1 \
        UV_NO_BUILD=1 \
        UV_PRERELEASE=disallow \
        UV_PYTHON_PREFERENCE=only-managed \
        UV_PYTHON_DOWNLOADS=manual \
        UV_DEFAULT_INDEX=https://pypi.org/simple \
        UV_INDEX_URL=https://pypi.org/simple \
        PIP_CONFIG_FILE=/dev/null \
        PIP_INDEX_URL=https://pypi.org/simple \
        PIP_EXTRA_INDEX_URL= \
        UV_INDEX= \
        UV_EXTRA_INDEX_URL= \
        "$@"
}

publish_bootstrap() {
    if [[ -n "$PENDING_UV_ROOT" ]]; then
        mkdir -p -- "$(dirname "$UV_ROOT")"
        [[ ! -e "$UV_ROOT" ]] || fail "managed uv destination became occupied"
        mv -- "$PENDING_UV_ROOT" "$UV_ROOT"
        PENDING_UV_ROOT=""
        UV_BIN="$UV_ROOT/uv"
    fi
    if [[ -n "$PENDING_PYTHON_ROOT" ]]; then
        mkdir -p -- "$PYTHON_ROOT"
        [[ ! -e "$PYTHON_ROOT/$PYTHON_IDENTITY" ]] || \
            fail "managed Python destination became occupied"
        mv -- "$PENDING_PYTHON_ROOT/$PYTHON_IDENTITY" "$PYTHON_ROOT/$PYTHON_IDENTITY"
        PENDING_PYTHON_ROOT=""
        PYTHON_INSTALL_DIR="$PYTHON_ROOT"
        PYTHON_BIN="$PYTHON_ROOT/$PYTHON_IDENTITY/bin/python3.12"
    fi
    mkdir -p -- "$CACHE_ROOT"
    UV_CACHE_SCOPE="$CACHE_ROOT"
}

verify_wheel_metadata() {
    local wheel="$1"
    "$PYTHON_BIN" -I - "$wheel" "$DAITA_VERSION" "$WHEEL_REQUIRES_PYTHON" <<'PY'
from email.parser import Parser
from pathlib import Path, PurePosixPath
import configparser
import stat
import sys
import zipfile

wheel = Path(sys.argv[1])
expected_version = sys.argv[2]
expected_python = sys.argv[3]
with zipfile.ZipFile(wheel) as archive:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise SystemExit("wheel contains duplicate archive members")
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise SystemExit("wheel contains an unsafe path")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise SystemExit("wheel contains a symbolic link")
    metadata = [item for item in infos if item.filename.endswith(".dist-info/METADATA")]
    entries = [item for item in infos if item.filename.endswith(".dist-info/entry_points.txt")]
    wheel_files = [item for item in infos if item.filename.endswith(".dist-info/WHEEL")]
    if len(metadata) != 1 or len(entries) != 1 or len(wheel_files) != 1:
        raise SystemExit("wheel must contain one metadata, entry-point, and WHEEL document")
    roots = {item.filename.split("/", 1)[0] for item in (*metadata, *entries, *wheel_files)}
    if len(roots) != 1:
        raise SystemExit("wheel metadata documents do not share one distribution root")
    if roots != {f"daita_agents-{expected_version}.dist-info"}:
        raise SystemExit("wheel distribution metadata path is inconsistent")
    message = Parser().parsestr(archive.read(metadata[0]).decode("utf-8"))
    if message.get_all("Name") != ["daita-agents"]:
        raise SystemExit("wheel distribution name does not match daita-agents")
    if message.get_all("Version") != [expected_version]:
        raise SystemExit("wheel version does not match the pinned Daita version")
    if message.get_all("Requires-Python") != [expected_python]:
        raise SystemExit("wheel Requires-Python does not match the reviewed constraint")
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.read_string(archive.read(entries[0]).decode("utf-8"))
    if set(parser.sections()) != {"console_scripts"}:
        raise SystemExit("wheel contains unexpected entry-point groups")
    if dict(parser.items("console_scripts")) != {"daita": "daita.cli:main"}:
        raise SystemExit("wheel console entry point does not match daita.cli:main")
PY
}

next_generation_name() {
    local prefix="$DAITA_VERSION-${WHEEL_SHA256:0:12}-"
    local highest=0
    local path
    local suffix
    if [[ -d "$GENERATIONS_DIR" ]]; then
        for path in "$GENERATIONS_DIR"/"$prefix"*; do
            [[ -d "$path" && ! -L "$path" ]] || continue
            suffix=${path##*-}
            if [[ "$suffix" =~ ^[0-9]+$ ]] && ((suffix > highest)); then
                highest=$suffix
            fi
        done
    fi
    printf '%s%s\n' "$prefix" "$((highest + 1))"
}

installed_metadata_check() {
    local python="$1"
    local expected_version="$2"
    local expected_python="$3"
    "$python" -I -c '
from importlib import metadata
import sys
d = metadata.distribution("daita-agents")
entries = {e.name: e.value for e in d.entry_points if e.group == "console_scripts"}
assert d.version == sys.argv[1]
assert d.metadata.get_all("Name") == ["daita-agents"]
assert d.metadata.get_all("Requires-Python") == [sys.argv[2]]
assert entries == {"daita": "daita.cli:main"}
' "$expected_version" "$expected_python"
}

lazy_import_check() {
    local python="$1"
    "$python" -I -c '
import sys
import daita
import daita.cli
blocked = {"anthropic", "asyncpg", "google", "keyring", "openai", "prompt_toolkit", "rich", "sqlglot", "xlsxwriter"}
loaded = blocked.intersection({name.split(".")[0] for name in sys.modules})
assert not loaded, sorted(loaded)
'
}

generation_python() {
    local generation="$1"
    printf '%s\n' "$generation/tool/daita-agents/bin/python"
}

verify_generation() {
    local generation="$1"
    local expected_version="$2"
    local expected_python="$3"
    local entrypoint="$generation/bin/daita"
    [[ -x "$entrypoint" && ! -L "$entrypoint" ]] || fail "generation entry point is missing or unsafe"
    local tool_environment="$generation/tool/daita-agents"
    [[ -d "$tool_environment" && ! -L "$tool_environment" ]] || \
        fail "generation tool environment is missing or unsafe"
    local python
    python=$(generation_python "$generation")
    [[ -x "$python" ]] || fail "generation Python is missing"
    local prefix
    prefix=$("$python" -I -c 'import sys; print(sys.prefix)')
    [[ "$prefix" = "$generation"/* ]] || fail "generation Python prefix escaped its generation"
    local version_output
    version_output=$(HOME="$HOME_REAL" "$entrypoint" --version)
    [[ "$version_output" == "daita $expected_version" ]] || fail "staged Daita version check failed"
    HOME="$HOME_REAL" "$entrypoint" --help | grep -F "usage: daita" >/dev/null || fail "staged help check failed"
    scoped_uv "$UV_BIN" pip check --python "$python"
    installed_metadata_check "$python" "$expected_version" "$expected_python"
    lazy_import_check "$python"
}

write_generation_manifest() {
    local generation="$1"
    local manifest="$generation/manifest"
    local python
    python=$(generation_python "$generation")
    {
        printf 'marker=%s\n' "$OWNER_MARKER"
        printf 'installer_version=%s\n' "$INSTALLER_VERSION"
        printf 'release_sequence=%s\n' "$RELEASE_SEQUENCE"
        printf 'app_version=%s\n' "$DAITA_VERSION"
        printf 'wheel_filename=%s\n' "$WHEEL_FILENAME"
        printf 'wheel_url=%s\n' "$WHEEL_URL"
        printf 'wheel_sha256=%s\n' "$WHEEL_SHA256"
        printf 'requires_python=%s\n' "$WHEEL_REQUIRES_PYTHON"
        printf 'uv_version=%s\n' "$UV_VERSION"
        printf 'uv_target=%s\n' "$TARGET"
        printf 'uv_archive_sha256=%s\n' "$UV_ARCHIVE_SHA256"
        printf 'python_request=%s\n' "$PYTHON_REQUEST"
        printf 'python_identity=%s\n' "$PYTHON_IDENTITY"
        printf 'generation_python=%s\n' "${python#"$generation"/}"
    } >"$manifest"
    chmod 600 "$manifest"
}

validate_generation_path() {
    local link="$1"
    [[ -L "$link" ]] || return 1
    local target
    target=$(readlink "$link")
    [[ "$target" == generations/* && "$target" != *".."* && "$target" != */*/* ]] || return 1
    local generation="$MANAGED_ROOT/$target"
    [[ -d "$generation" && ! -L "$generation" ]] || return 1
    printf '%s\n' "$generation"
}

verify_manifest() {
    local generation="$1"
    local manifest="$generation/manifest"
    [[ -f "$manifest" && ! -L "$manifest" ]] || fail "generation manifest is missing or unsafe"
    [[ "$(state_value "$manifest" marker)" == "$OWNER_MARKER" ]] || fail "generation ownership marker is invalid"
    local app_version
    app_version=$(state_value "$manifest" app_version) || fail "generation version is missing"
    [[ "$app_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "generation version is invalid"
    local sequence
    sequence=$(state_value "$manifest" release_sequence) || fail "generation sequence is missing"
    [[ "$sequence" =~ ^[1-9][0-9]*$ ]] || fail "generation sequence is invalid"
    local wheel_sha
    wheel_sha=$(state_value "$manifest" wheel_sha256) || fail "generation wheel digest is missing"
    validate_sha256_literal "$wheel_sha"
    local uv_version
    local uv_archive_sha
    uv_version=$(state_value "$manifest" uv_version) || fail "generation uv version is missing"
    uv_archive_sha=$(state_value "$manifest" uv_archive_sha256) || fail "generation uv archive digest is missing"
    validate_sha256_literal "$uv_archive_sha"
    UV_BIN="$MANAGED_ROOT/installer/uv/$uv_version/uv"
    local uv_state="$MANAGED_ROOT/installer/uv/$uv_version/manifest"
    [[ -x "$UV_BIN" && ! -L "$UV_BIN" && -f "$uv_state" ]] || fail "managed uv is missing or unsafe"
    [[ "$(state_value "$uv_state" archive_sha256)" == "$uv_archive_sha" ]] || fail "managed uv archive digest is inconsistent"
    [[ "$(sha256_file "$UV_BIN")" == "$(state_value "$uv_state" binary_sha256)" ]] || fail "managed uv binary digest is invalid"
    local requires_python
    requires_python=$(state_value "$manifest" requires_python) || fail "generation Python requirement is missing"
    [[ -n "$requires_python" ]] || fail "generation Python requirement is empty"
    verify_generation "$generation" "$app_version" "$requires_python"
}

verify_installation() {
    local verify_path="${1:-1}"
    validate_owner_state || fail "managed installation ownership is missing or invalid"
    launcher_is_owned || fail "managed launcher ownership or digest is invalid"
    local generation
    generation=$(validate_generation_path "$MANAGED_ROOT/current") || fail "current generation link is invalid"
    verify_manifest "$generation"
    local path_state="$INSTALL_STATE/path"
    if [[ -f "$path_state" && "$verify_path" == 1 ]]; then
        verify_path_suffix "$path_state" || fail "recorded Daita PATH block has changed"
    fi
    say "Verified Daita $(state_value "$generation/manifest" app_version)"
    say "Launcher: $PUBLIC_LAUNCHER"
    say "Generation: $generation"
}

current_manifest_value() {
    local key="$1"
    local generation
    generation=$(validate_generation_path "$MANAGED_ROOT/current") || return 1
    state_value "$generation/manifest" "$key"
}

create_launcher() {
    local destination="$1"
    local quoted_root
    printf -v quoted_root '%q' "$MANAGED_ROOT"
    {
        printf '#!/usr/bin/env bash\n'
        printf '%s\n' "$LAUNCHER_MARKER"
        printf 'set -euo pipefail\n'
        printf 'managed_root=%s\n' "$quoted_root"
        cat <<'EOF'
current="$managed_root/current"
[[ -L "$current" ]] || { printf 'Daita managed installation is damaged; run the managed installer with --repair.\n' >&2; exit 1; }
generation=$(CDPATH= cd -P -- "$current" && pwd -P)
case "$generation" in
    "$managed_root"/generations/*) ;;
    *) printf 'Daita managed generation escaped its installation root.\n' >&2; exit 1 ;;
esac
target="$generation/bin/daita"
[[ -x "$target" && ! -L "$target" ]] || { printf 'Daita managed entry point is damaged; run the managed installer with --repair.\n' >&2; exit 1; }
export DAITA_MANAGED_INSTALL_ROOT="$managed_root"
exec "$target" "$@"
EOF
    } >"$destination"
    chmod 700 "$destination"
}

publish_launcher() {
    local staged="$1"
    mkdir -p -- "$PUBLIC_BIN_DIR"
    local temporary="$PUBLIC_BIN_DIR/.daita.tmp.$$"
    cp -- "$staged" "$temporary"
    chmod 700 "$temporary"
    mv -f -- "$temporary" "$PUBLIC_LAUNCHER"
    NEW_LAUNCHER_PUBLISHED=1
    local state_source="$STAGE/launcher-state"
    {
        printf 'marker=%s\n' "$OWNER_MARKER"
        printf 'path=%s\n' "$PUBLIC_LAUNCHER"
        printf 'sha256=%s\n' "$(sha256_file "$PUBLIC_LAUNCHER")"
    } >"$state_source"
    write_atomic_file "$INSTALL_STATE/launcher" "$state_source"
}

path_file_for_shell() {
    local shell_name
    shell_name=$(basename "${SHELL:-}")
    case "$shell_name" in
        zsh) printf '%s\n' "$HOME_REAL/.zshrc" ;;
        bash) printf '%s\n' "$HOME_REAL/.bashrc" ;;
        *) return 1 ;;
    esac
}

path_decision() {
    if ((NO_MODIFY_PATH == 1)); then
        printf 'disabled (--no-modify-path)\n'
        return
    fi
    case ":${PATH:-}:" in
        *:"$PUBLIC_BIN_DIR":*) printf 'already available in current PATH\n'; return ;;
    esac
    local file
    if file=$(path_file_for_shell); then
        printf 'append one owned block to %s\n' "$file"
    else
        printf 'no automatic edit for shell %s\n' "${SHELL:-unknown}"
    fi
}

verify_path_suffix() {
    local state="$1"
    local path
    local digest
    local length
    path=$(state_value "$state" path) || return 1
    digest=$(state_value "$state" sha256) || return 1
    length=$(state_value "$state" bytes) || return 1
    [[ "$path" == "$HOME_REAL/.zshrc" || "$path" == "$HOME_REAL/.bashrc" ]] || return 1
    [[ "$length" =~ ^[1-9][0-9]*$ && -f "$path" && ! -L "$path" ]] || return 1
    [[ "$(tail -c "$length" "$path" | sha256_stdin)" == "$digest" ]]
}

ensure_path() {
    if ((NO_MODIFY_PATH == 1)); then
        say "PATH unchanged (--no-modify-path). Run: $PUBLIC_LAUNCHER"
        return
    fi
    case ":${PATH:-}:" in
        *:"$PUBLIC_BIN_DIR":*) say "PATH already includes $PUBLIC_BIN_DIR"; return ;;
    esac
    local file
    if ! file=$(path_file_for_shell); then
        say "PATH unchanged for unsupported shell ${SHELL:-unknown}. Run: $PUBLIC_LAUNCHER"
        return
    fi
    [[ ! -L "$file" ]] || { warn "PATH file is a symlink; preserving it: $file"; return; }
    local path_state="$INSTALL_STATE/path"
    if [[ -f "$path_state" ]]; then
        if verify_path_suffix "$path_state"; then
            say "Daita PATH block already present in $file"
        else
            warn "recorded Daita PATH block changed; preserving $file without another edit"
        fi
        return
    fi
    if [[ -f "$file" ]] && { grep -F "$PATH_START" "$file" >/dev/null 2>&1 || grep -F "$PATH_END" "$file" >/dev/null 2>&1; }; then
        warn "unowned Daita PATH marker found; preserving $file without another edit"
        return
    fi
    local suffix="$STAGE/path-suffix"
    : >"$suffix"
    if [[ -s "$file" ]]; then
        local last
        last=$(tail -c 1 "$file" || true)
        if [[ -n "$last" ]]; then
            printf '\n' >>"$suffix"
        fi
    fi
    {
        printf '%s\n' "$PATH_START"
        printf 'case ":$PATH:" in\n'
        printf '  *:%q:*) ;;\n' "$PUBLIC_BIN_DIR"
        printf '  *) export PATH=%q:"$PATH" ;;\n' "$PUBLIC_BIN_DIR"
        printf 'esac\n'
        printf '%s\n' "$PATH_END"
    } >>"$suffix"
    local temporary="${file}.daita.tmp.$$"
    if [[ -f "$file" ]]; then
        cp -- "$file" "$temporary"
    else
        : >"$temporary"
    fi
    cat "$suffix" >>"$temporary"
    chmod 600 "$temporary"
    mv -f -- "$temporary" "$file"
    local state_source="$STAGE/path-state"
    {
        printf 'marker=%s\n' "$OWNER_MARKER"
        printf 'path=%s\n' "$file"
        printf 'sha256=%s\n' "$(sha256_file "$suffix")"
        printf 'bytes=%s\n' "$(wc -c <"$suffix" | tr -d ' ')"
    } >"$state_source"
    write_atomic_file "$path_state" "$state_source"
    say "Added one Daita PATH block to $file"
}

remove_path() {
    local state="$INSTALL_STATE/path"
    [[ -f "$state" ]] || return 0
    if ! verify_path_suffix "$state"; then
        warn "Daita PATH block changed; preserving the shell startup file"
        return
    fi
    local path
    local length
    path=$(state_value "$state" path)
    length=$(state_value "$state" bytes)
    local size
    size=$(wc -c <"$path" | tr -d ' ')
    local keep=$((size - length))
    local temporary="${path}.daita.tmp.$$"
    if ((keep > 0)); then
        dd if="$path" of="$temporary" bs=1 count="$keep" 2>/dev/null
    else
        : >"$temporary"
    fi
    chmod 600 "$temporary"
    mv -f -- "$temporary" "$path"
}

dry_run() {
    say "Action: $ACTION (dry-run; no downloads or writes)"
    say "Installer version: $INSTALLER_VERSION"
    say "Release sequence: $RELEASE_SEQUENCE"
    say "Daita version: $DAITA_VERSION"
    say "Target: $TARGET"
    say "Wheel: $WHEEL_FILENAME"
    say "Wheel URL: $WHEEL_URL"
    say "Wheel SHA-256: $WHEEL_SHA256"
    say "uv version: $UV_VERSION"
    say "uv archive: $UV_ARCHIVE"
    say "uv URL: $UV_ARCHIVE_URL"
    say "uv SHA-256: $UV_ARCHIVE_SHA256"
    say "Python request: $PYTHON_REQUEST"
    say "Python identity: $PYTHON_IDENTITY"
    say "Managed root: $MANAGED_ROOT"
    say "Public launcher: $PUBLIC_LAUNCHER"
    say "Application data: separate and unchanged"
    say "PATH: $(path_decision)"
    if [[ "$WHEEL_URL" == *UNRESOLVED* || "$UV_ARCHIVE_URL" == *UNRESOLVED* ]]; then
        say "Release readiness: blocked by unresolved immutable artifact literals"
    fi
}

write_owner() {
    local source="$STAGE/owner"
    {
        printf 'marker=%s\n' "$OWNER_MARKER"
        printf 'root=%s\n' "$MANAGED_ROOT"
    } >"$source"
    write_atomic_file "$INSTALL_STATE/owner" "$source"
}

install_generation() {
    local repair="$1"
    local installed_sequence=""
    local installed_sha=""
    if installed_sequence=$(current_manifest_value release_sequence 2>/dev/null); then
        [[ "$installed_sequence" =~ ^[1-9][0-9]*$ ]] || fail "active release sequence is invalid"
        if ((installed_sequence > RELEASE_SEQUENCE)); then
            fail "this older installer refuses to replace newer release sequence $installed_sequence"
        fi
        installed_sha=$(current_manifest_value wheel_sha256 2>/dev/null || true)
        if ((repair == 0)) && [[ "$installed_sequence" == "$RELEASE_SEQUENCE" && "$installed_sha" == "$WHEEL_SHA256" ]]; then
            verify_installation 0
            ensure_path
            say "Daita $DAITA_VERSION is already installed and verified."
            return
        fi
    fi
    local active_generation=""
    if active_generation=$(validate_generation_path "$MANAGED_ROOT/current" 2>/dev/null); then
        if (verify_manifest "$active_generation") >/dev/null 2>&1; then
            OLD_CURRENT_VERIFIED=1
        elif ((repair == 0)); then
            fail "the active generation is damaged; rerun this installer with --repair"
        fi
    fi

    mkdir -p -- "$GENERATIONS_DIR" "$STAGING_DIR"
    STAGE="$STAGING_DIR/transaction-$$"
    [[ ! -e "$STAGE" ]] || fail "staging collision"
    mkdir -- "$STAGE"
    UV_CACHE_SCOPE="$STAGE/cache"
    mkdir -- "$UV_CACHE_SCOPE"
    local generation_name
    generation_name=$(next_generation_name)
    NEW_GENERATION="$GENERATIONS_DIR/$generation_name"
    [[ ! -e "$NEW_GENERATION" ]] || fail "generation destination collision"
    GENERATION_STAGE="$NEW_GENERATION"
    mkdir -p -- "$GENERATION_STAGE/tool" "$GENERATION_STAGE/bin"

    write_owner
    prepare_uv
    test_failpoint "after-uv"
    prepare_python
    test_failpoint "after-python"

    local wheel="$STAGE/$WHEEL_FILENAME"
    download "$WHEEL_URL" "$wheel"
    verify_checksum "$wheel" "$WHEEL_SHA256"
    verify_wheel_metadata "$wheel"
    test_failpoint "after-wheel"
    publish_bootstrap
    test_failpoint "after-bootstrap"

    scoped_uv "$UV_BIN" tool install --python "$PYTHON_BIN" --force "$wheel"
    local tool_entrypoint="$GENERATION_STAGE/tool/daita-agents/bin/daita"
    [[ -x "$tool_entrypoint" ]] || fail "uv did not install the expected Daita tool entry point"
    cp -- "$tool_entrypoint" "$GENERATION_STAGE/bin/.daita.tmp"
    chmod 700 "$GENERATION_STAGE/bin/.daita.tmp"
    mv -f -- "$GENERATION_STAGE/bin/.daita.tmp" "$GENERATION_STAGE/bin/daita"
    test_failpoint "after-tool-install"
    verify_generation "$GENERATION_STAGE" "$DAITA_VERSION" "$WHEEL_REQUIRES_PYTHON"
    test_failpoint "after-staged-checks"
    write_generation_manifest "$GENERATION_STAGE"
    test_failpoint "after-manifest"
    local launcher="$STAGE/daita-launcher"
    create_launcher "$launcher"

    OLD_CURRENT=$(readlink "$MANAGED_ROOT/current" 2>/dev/null || true)
    OLD_PREVIOUS=$(readlink "$MANAGED_ROOT/previous" 2>/dev/null || true)
    ACTIVATION_STARTED=1
    atomic_symlink "generations/$generation_name" "$MANAGED_ROOT/current"
    test_failpoint "after-current-switch"
    verify_generation "$NEW_GENERATION" "$DAITA_VERSION" "$WHEEL_REQUIRES_PYTHON"
    if [[ -n "$OLD_CURRENT" && "$OLD_CURRENT_VERIFIED" == 1 ]]; then
        atomic_symlink "$OLD_CURRENT" "$MANAGED_ROOT/previous"
    elif [[ -z "$OLD_PREVIOUS" && -L "$MANAGED_ROOT/previous" ]]; then
        rm -f -- "$MANAGED_ROOT/previous"
    fi
    test_failpoint "after-previous-switch"
    if [[ ! -e "$PUBLIC_LAUNCHER" && ! -L "$PUBLIC_LAUNCHER" ]]; then
        publish_launcher "$launcher"
    fi
    test_failpoint "after-launcher"
    HOME="$HOME_REAL" "$PUBLIC_LAUNCHER" --version | grep -Fx "daita $DAITA_VERSION" >/dev/null || \
        fail "activated public command failed its version check"
    ACTIVATION_COMMITTED=1
    NEW_LAUNCHER_PUBLISHED=0

    ensure_path
    say "Installed Daita $DAITA_VERSION"
    say "Launcher: $PUBLIC_LAUNCHER"
    say "Generation: $NEW_GENERATION"
    say "Binary rollback does not roll back application data."
    maybe_onboard
}

rollback_installation() {
    validate_owner_state || fail "managed installation ownership is missing or invalid"
    launcher_is_owned || fail "managed launcher ownership or digest is invalid"
    local current
    local previous
    current=$(validate_generation_path "$MANAGED_ROOT/current") || fail "current generation is invalid"
    previous=$(validate_generation_path "$MANAGED_ROOT/previous") || fail "no verified previous generation is available"
    verify_manifest "$current"
    verify_manifest "$previous"
    local current_target
    local previous_target
    current_target=$(readlink "$MANAGED_ROOT/current")
    previous_target=$(readlink "$MANAGED_ROOT/previous")
    OLD_CURRENT="$current_target"
    OLD_PREVIOUS="$previous_target"
    ACTIVATION_STARTED=1
    atomic_symlink "$previous_target" "$MANAGED_ROOT/current"
    verify_manifest "$previous"
    atomic_symlink "$current_target" "$MANAGED_ROOT/previous"
    HOME="$HOME_REAL" "$PUBLIC_LAUNCHER" --version >/dev/null
    ACTIVATION_COMMITTED=1
    say "Rolled back the Daita binary to $(state_value "$previous/manifest" app_version)."
    say "Application data was not changed or rolled back."
}

uninstall_application() {
    validate_owner_state || fail "managed installation ownership is missing or invalid"
    if [[ -e "$PUBLIC_LAUNCHER" || -L "$PUBLIC_LAUNCHER" ]]; then
        if launcher_is_owned; then
            rm -f -- "$PUBLIC_LAUNCHER"
        else
            warn "launcher changed or is foreign; preserving $PUBLIC_LAUNCHER"
        fi
    fi
    remove_path
    [[ "$MANAGED_ROOT" == "$HOME_REAL/.local/share/daita" ]] || fail "managed root validation changed"
    rm -rf -- "$MANAGED_ROOT"
    LOCK_HELD=0
    STAGE=""
    say "Uninstalled the managed Daita application."
    say "Application data and keychain entries were preserved."
}

maybe_onboard() {
    if ((NO_ONBOARD == 1)); then
        say "Onboarding skipped. Run later: $PUBLIC_LAUNCHER"
        return
    fi
    if [[ -t 1 && -c /dev/tty ]] && (: </dev/tty) 2>/dev/null; then
        "$PUBLIC_LAUNCHER" </dev/tty >/dev/tty 2>/dev/tty || \
            warn "onboarding exited unsuccessfully; run later: $PUBLIC_LAUNCHER"
    else
        say "Onboarding not launched without a controlling terminal. Run later: $PUBLIC_LAUNCHER"
    fi
}

preflight_mutation() {
    reject_elevated_execution
    require_command curl
    require_command tar
    require_command awk
    require_command grep
    require_command cp
    require_command mv
    require_command rm
    require_command mkdir
    require_command chmod
    require_command readlink
    require_command env
    printf 'daita-installer-hash-preflight' | sha256_stdin >/dev/null
    admit_managed_root
    reject_launcher_collision
    mkdir -p -- "$MANAGED_ROOT" "$INSTALL_STATE" "$STAGING_DIR"
    chmod 700 "$MANAGED_ROOT" "$INSTALL_STATE" "$STAGING_DIR"
    acquire_lock
    test_failpoint "after-lock"
}

main() {
    parse_arguments "$@"
    validate_home
    resolve_platform

    if ((DRY_RUN == 1)); then
        dry_run
        return
    fi

    case "$ACTION" in
        verify)
            UV_BIN="$UV_ROOT/uv"
            GENERATION_STAGE="$MANAGED_ROOT/current"
            verify_installation
            ;;
        install|repair)
            validate_release_literals
            preflight_mutation
            local repair=0
            if [[ "$ACTION" == "repair" ]]; then
                repair=1
            fi
            install_generation "$repair"
            ;;
        rollback)
            reject_elevated_execution
            admit_managed_root
            reject_launcher_collision
            mkdir -p -- "$STAGING_DIR"
            acquire_lock
            STAGE="$STAGING_DIR/rollback-$$"
            mkdir -- "$STAGE"
            UV_BIN="$UV_ROOT/uv"
            GENERATION_STAGE="$MANAGED_ROOT/current"
            rollback_installation
            ;;
        uninstall)
            reject_elevated_execution
            admit_managed_root
            reject_launcher_collision
            mkdir -p -- "$STAGING_DIR"
            acquire_lock
            STAGE="$STAGING_DIR/uninstall-$$"
            mkdir -- "$STAGE"
            uninstall_application
            ;;
        *) fail "internal unsupported action: $ACTION" ;;
    esac
}

main "$@"
