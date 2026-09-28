import subprocess
import sys


def test_old_openssl_gets_a_clear_error():
    # the check runs before Julia starts, so this doesn't initialize Julia
    code = (
        "import ssl\n"
        "ssl.OPENSSL_VERSION_INFO = (3, 0, 13, 0, 0)\n"
        "ssl.OPENSSL_VERSION = 'OpenSSL 3.0.13 30 Jan 2024'\n"
        "import galley_jl_python\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode != 0
    assert "OpenSSL 3.5 or newer" in result.stderr
    assert "OpenSSL 3.0.13" in result.stderr
