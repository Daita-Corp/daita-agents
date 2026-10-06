"""Select disposable PostgreSQL qualification explicitly; never use ambient DSNs."""


def pytest_addoption(parser):
    parser.addoption(
        "--postgres",
        action="store_true",
        help="Run real disposable TLS PostgreSQL storage qualification (requires Docker and OpenSSL)",
    )


def pytest_ignore_collect(collection_path, config):
    if collection_path.name == "postgres" and collection_path.parent.name == "storage":
        return not config.getoption("--postgres")
    return None
