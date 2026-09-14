#!/usr/bin/env bash
# Daita managed installer host bootstrap; run once as root on Lightsail.

set -euo pipefail
IFS=$'\n\t'
umask 077

readonly DEPLOY_USER="daita-installer-deploy"
readonly DEPLOY_HOME="/var/lib/$DEPLOY_USER"
readonly HOST_ROOT="/srv/daita-installer"
readonly PROGRAM_ROOT="/usr/local/libexec/daita-installer"
readonly PROGRAM="$PROGRAM_ROOT/managed_installer_host.py"
readonly NGINX_PROGRAM="$PROGRAM_ROOT/managed_installer_nginx.py"
readonly NGINX_SNIPPET="$PROGRAM_ROOT/daita-installer-location.nginx.conf"
readonly MARKER_ROOT="/etc/daita-installer-host"
SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly SOURCE_ROOT

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

if (($# != 1)); then
    fail "usage: sudo bash scripts/bootstrap_managed_installer_host.sh /absolute/path/to/deploy-key.pub"
fi
if ((EUID != 0)); then
    fail "bootstrap must run as root"
fi

readonly PUBLIC_KEY_FILE="$1"
[[ "$PUBLIC_KEY_FILE" == /* ]] || fail "public-key path must be absolute"
[[ -f "$PUBLIC_KEY_FILE" && ! -L "$PUBLIC_KEY_FILE" ]] || \
    fail "public-key path must be one regular file"
[[ "$(wc -l <"$PUBLIC_KEY_FILE" | tr -d ' ')" == "1" ]] || \
    fail "public-key file must contain exactly one line"
IFS=' ' read -r key_type key_body _ <"$PUBLIC_KEY_FILE"
[[ "$key_type" == "ssh-ed25519" && "$key_body" == AAAA* ]] || \
    fail "deployment key must be one OpenSSH Ed25519 public key"
ssh-keygen -l -f "$PUBLIC_KEY_FILE" >/dev/null || fail "deployment public key is invalid"

readonly HOST_PROGRAM_SOURCE="$SOURCE_ROOT/scripts/managed_installer_host.py"
readonly NGINX_PROGRAM_SOURCE="$SOURCE_ROOT/scripts/managed_installer_nginx.py"
readonly NGINX_SNIPPET_SOURCE="$SOURCE_ROOT/release/daita-installer-location.nginx.conf"
for source in "$HOST_PROGRAM_SOURCE" "$NGINX_PROGRAM_SOURCE" "$NGINX_SNIPPET_SOURCE"; do
    [[ -f "$source" && ! -L "$source" ]] || fail "missing reviewed source: $source"
done
/usr/bin/python3 - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("managed installer host requires Python 3.10 or newer")
PY

for managed_path in "$DEPLOY_HOME" "$HOST_ROOT" "$PROGRAM_ROOT" "$MARKER_ROOT"; do
    [[ ! -L "$managed_path" ]] || fail "managed path must not be a symlink: $managed_path"
done

if id "$DEPLOY_USER" >/dev/null 2>&1; then
    [[ -d "$MARKER_ROOT" ]] || fail "managed installer marker directory is missing"
    [[ -f "$MARKER_ROOT/managed" && ! -L "$MARKER_ROOT/managed" ]] || \
        fail "refusing to adopt an existing unmanaged $DEPLOY_USER account"
    [[ "$(<"$MARKER_ROOT/managed")" == "daita-managed-installer-host-v1" ]] || \
        fail "managed installer marker is invalid"
else
    for existing_path in "$DEPLOY_HOME" "$HOST_ROOT" "$PROGRAM_ROOT" "$MARKER_ROOT"; do
        [[ ! -e "$existing_path" ]] || \
            fail "refusing to replace an existing unmanaged path: $existing_path"
    done
    install -d -o root -g root -m 0755 "$MARKER_ROOT"
    useradd --system --create-home --home-dir "$DEPLOY_HOME" --shell /bin/bash \
        "$DEPLOY_USER"
    passwd --lock "$DEPLOY_USER" >/dev/null
    printf '%s\n' "daita-managed-installer-host-v1" >"$MARKER_ROOT/managed"
    chown root:root "$MARKER_ROOT/managed"
    chmod 0444 "$MARKER_ROOT/managed"
fi
ACCOUNT_RECORD="$(getent passwd "$DEPLOY_USER")"
readonly ACCOUNT_RECORD
IFS=: read -r account_name _ account_uid _ _ account_home account_shell \
    <<<"$ACCOUNT_RECORD"
[[ "$account_name" == "$DEPLOY_USER" && "$account_uid" != "0" ]] || \
    fail "managed deployment account identity is invalid"
[[ "$account_home" == "$DEPLOY_HOME" && "$account_shell" == "/bin/bash" ]] || \
    fail "managed deployment account home or shell is invalid"
DEPLOY_GROUP="$(id -gn "$DEPLOY_USER")"
readonly DEPLOY_GROUP
[[ -d "$DEPLOY_HOME" && ! -L "$DEPLOY_HOME" ]] || \
    fail "managed deployment account home must be a real directory"
chown root:root "$DEPLOY_HOME"
chmod 0755 "$DEPLOY_HOME"

install -d -o root -g root -m 0755 "$PROGRAM_ROOT" "$MARKER_ROOT"
install -o root -g root -m 0555 \
    "$HOST_PROGRAM_SOURCE" "$PROGRAM"
install -o root -g root -m 0555 \
    "$NGINX_PROGRAM_SOURCE" "$NGINX_PROGRAM"
install -o root -g root -m 0444 \
    "$NGINX_SNIPPET_SOURCE" "$NGINX_SNIPPET"

install -d -o "$DEPLOY_USER" -g "$DEPLOY_GROUP" -m 0755 \
    "$HOST_ROOT" "$HOST_ROOT/releases"
install -d -o root -g root -m 0755 \
    "$DEPLOY_HOME/.ssh"
for managed_directory in \
    "$DEPLOY_HOME" "$DEPLOY_HOME/.ssh" "$HOST_ROOT" "$HOST_ROOT/releases" \
    "$PROGRAM_ROOT" "$MARKER_ROOT"; do
    [[ -d "$managed_directory" && ! -L "$managed_directory" ]] || \
        fail "managed path must be a real directory: $managed_directory"
done

readonly AUTHORIZED_KEYS="$DEPLOY_HOME/.ssh/authorized_keys"
if [[ -e "$AUTHORIZED_KEYS" || -L "$AUTHORIZED_KEYS" ]]; then
    [[ -f "$AUTHORIZED_KEYS" && ! -L "$AUTHORIZED_KEYS" ]] || \
        fail "authorized_keys must be one regular file"
fi
AUTHORIZED_KEYS_TEMP="$(mktemp "$DEPLOY_HOME/.ssh/.authorized_keys.XXXXXX")"
readonly AUTHORIZED_KEYS_TEMP
readonly FORCED_COMMAND="/usr/bin/python3 $PROGRAM --root $HOST_ROOT forced-promote"
trap 'rm -f -- "$AUTHORIZED_KEYS_TEMP"' EXIT
printf 'restrict,command="%s" %s %s daita-installer-release\n' \
    "$FORCED_COMMAND" "$key_type" "$key_body" \
    >"$AUTHORIZED_KEYS_TEMP"
chown root:root "$AUTHORIZED_KEYS_TEMP"
chmod 0444 "$AUTHORIZED_KEYS_TEMP"
mv -f -- "$AUTHORIZED_KEYS_TEMP" "$AUTHORIZED_KEYS"

printf 'Configured %s with a forced promotion command.\n' "$DEPLOY_USER"
printf 'Installer root: %s\n' "$HOST_ROOT"
printf 'Next: promote the current release, then run sudo %s cutover.\n' "$NGINX_PROGRAM"
