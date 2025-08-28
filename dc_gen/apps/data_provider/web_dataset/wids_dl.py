import fcntl
import os
import shutil
import sys
import time
from collections import deque
from datetime import datetime
from urllib.parse import urlparse

recent_downloads = deque(maxlen=1000)

open_objects = {}
max_open_objects = 100


def pipe_download(remote, local):
    """Perform a download for a pipe: url."""
    assert remote.startswith("pipe:")
    cmd = remote[5:]
    cmd = cmd.format(local=local)
    assert os.system(cmd) == 0, "Command failed: %s" % cmd


def copy_file(remote, local):
    remote = urlparse(remote)
    assert remote.scheme in ["file", ""]
    # use absolute path
    remote = os.path.abspath(remote.path)
    local = urlparse(local)
    assert local.scheme in ["file", ""]
    local = os.path.abspath(local.path)
    if remote == local:
        return
    # check if the local file exists
    shutil.copyfile(remote, local)


verbose_cmd = int(os.environ.get("WIDS_VERBOSE_CMD", "0"))


def vcmd(flag, verbose_flag=""):
    return verbose_flag if verbose_cmd else flag


default_cmds = {
    "posixpath": copy_file,
    "file": copy_file,
    "pipe": pipe_download,
    "http": "curl " + vcmd("-s") + " -L {url} -o {local}",
    "https": "curl " + vcmd("-s") + " -L {url} -o {local}",
    "ftp": "curl " + vcmd("-s") + " -L {url} -o {local}",
    "ftps": "curl " + vcmd("-s") + " -L {url} -o {local}",
    "gs": "gsutil " + vcmd("-q") + " cp {url} {local}",
    "s3": "aws s3 cp {url} {local}",
}

# TODO(ligeng): change HTTPS download to python requests library


def download_file_no_log(remote, local, handlers=default_cmds):
    """Download a file from a remote url to a local path.
    The remote url can be a pipe: url, in which case the remainder of
    the url is treated as a command template that is executed to perform the download.
    """

    if remote.startswith("pipe:"):
        schema = "pipe"
    else:
        schema = urlparse(remote).scheme
    if schema is None or schema == "":
        schema = "posixpath"
    # get the handler
    handler = handlers.get(schema)
    if handler is None:
        raise ValueError("Unknown schema: %s" % schema)
    # call the handler
    if callable(handler):
        handler(remote, local)
    else:
        assert isinstance(handler, str)
        cmd = handler.format(url=remote, local=local)
        assert os.system(cmd) == 0, "Command failed: %s" % cmd
    return local


def download_file(remote, local, handlers=default_cmds, verbose=False):
    start = time.time()
    try:
        return download_file_no_log(remote, local, handlers=handlers)
    finally:
        recent_downloads.append((remote, local, time.time(), time.time() - start))
        if verbose:
            print(
                "downloaded",
                remote,
                "to",
                local,
                "in",
                time.time() - start,
                "seconds",
                file=sys.stderr,
            )


def download_and_open(remote, local, mode="rb", handlers=default_cmds, verbose=False):
    assert os.path.exists(remote), f"File {remote} must be downloaded to local first."

    result = open(remote, mode)

    if open_objects is not None:
        for k, v in list(open_objects.items()):
            if v.closed:
                del open_objects[k]
        if len(open_objects) > max_open_objects:
            raise RuntimeError("Too many open objects")
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        key = tuple(str(x) for x in [remote, local, mode, current_time])
        open_objects[key] = result
    return result
