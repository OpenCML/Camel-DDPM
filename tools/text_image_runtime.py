"""Host-side launch helpers. All model computation runs in Camel."""
import hashlib
import os
from pathlib import Path
import subprocess
import threading

PROJECT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def environment(gpu):
    home = PROJECT / ".camel"
    env = os.environ.copy()
    env.update(CAMEL_HOME=str(home), CAMEL_STD_LIB=str(home / "stdlib"),
               CAMEL_PACKAGES=str(PROJECT / "src"), CAMEL_TILES_DEVICE=str(gpu),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env["LD_LIBRARY_PATH"] = str(home / "libs") + ":/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    libs = [home / "libs" / "libstdc++.so.6", home / "libs" / "libgcc_s.so.1"]
    if all(path.is_file() for path in libs):
        env["LD_PRELOAD"] = ":".join(map(str, libs)) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    return env


def run_native(source, arguments, log, gpu, timeout=600):
    command = [str(PROJECT / ".camel/bin/camel")]
    for key, value in arguments.items():
        command += ["--arg", key + "=" + str(value)]
    command += [source, "std::macro", "std::fvm"]
    # Stream output, retaining the exact command outcome in a separate run log.
    with Path(log).open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=PROJECT, env=environment(gpu),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = threading.Timer(timeout, process.kill)
        deadline.daemon = True
        deadline.start()
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait(timeout=timeout)
        except BaseException:
            process.terminate()
            process.wait(timeout=10)
            raise
        finally:
            deadline.cancel()
    if code:
        raise RuntimeError("Camel exited with code " + str(code) + ": " + source)
