#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 1998-2021 all rights reserved


# externals
import os
import re
import subprocess


# assemble the prompt
def prompt():
    """
    Assemble and print the prompt
    """
    # assemble
    prompt = ''.join(generate())
    # and print
    print(prompt, end='')
    # all done
    return


# the prompt generator
def generate():
    """
    Make a prompt string for the bash shell
    """
    # compute a pretty cwd
    cwd = getcwd()
    # decorate the title of the window
    yield from decorateWindow(cwd=cwd)

    # if git is installed
    try:
        # and we are within a git repository, show its status
        yield from gitRepository()
    # otherwise
    except:
        # ignore
        pass

    # add a final separator
    yield from finalize(cwd=cwd)

    # all done
    return


# support for git
def gitRepository():
    """
    Show the status of {git} repositories
    """
    # set up the command
    cmd = [ "git", "status", "--porcelain", "--branch"]
    # settings
    options = {
        "executable": "git",
        "args": cmd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "universal_newlines": True,
        "shell": False }
    # invoke
    with subprocess.Popen(**options) as git:
        # collect the output
        stdout, stderr = git.communicate()
        # if there was an error
        if git.returncode != 0:
            # this is not a git repository, so we are done
            return
        # otherwise, parse the output
        summary = gitParseStatus(status=stdout.splitlines())
        # get the branch name
        branch = summary["local"]
        # get the divergence information
        ahead = summary["ahead"]
        behind = summary["behind"]
        # get change info
        index = summary["index"]
        worktree = summary["worktree"]
        conflicts = summary["conflicts"]

        # start rendering
        yield f"{prompt_normal}["

        # if there are conflicts
        if conflicts:
            # mark it
            status = branch_conflicts
        # if both the index and the worktree have changes
        elif worktree > 0 and index > 0:
            # mark it
            status = branch_modified
        # if there are changes in the worktree
        elif worktree > 0:
            # mark it
            status = branch_modified_worktree
        # if there are changes in the index
        elif index > 0:
            # mark it
            status = branch_modified_index
        # otherwise
        else:
            # mark it as clean
            status = branch_clean

        # render the branch name in a color appropriate for its status
        yield f"{status}{branch}"

        # divergence information
        if ahead or behind:
            # add a separator
            yield f"{prompt_normal}:"
        # if we are ahead of the remote branch
        if ahead:
            # render
            yield f"{branch_ahead}+{ahead}"
        # if we are behind
        if behind:
            # render
            yield f"{branch_behind}-{behind}"

        # wrap up
        yield f"{prompt_normal}]:"

    # all done
    return


def gitParseStatus(status):
    # the summary table
    table = {
        "local": None,
        "remote": None,
        "ahead": 0,
        "behind": 0,
        "worktree": 0,
        "index": 0,
        "conflicts": 0,
    }
    # go through the lines in {status}
    for line in status:
        # attempt to match it
        match = gitParser.match(line)
        # if we couldn't
        if match is None:
            # ignore and move on
            continue
        # get the enclosing group name
        case = match.lastgroup
        # look up the handler and dispatch
        gitDispatcher[case](table=table, match=match)
    # all done
    return table


def gitNoCommits(table, match):
    """
    Extract the branch name of a newly created repository
    """
    # set the branch name
    table["local"] = match.group("new")
    # all done
    return table


def gitDetached(table, match):
    """
    In detached HEAD state
    """
    # let's find out the hash of the current commit
    cmd = [ "git", "log", "--format=format:%h", "-n", "1"]
    # settings
    options = {
        "executable": "git",
        "args": cmd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "universal_newlines": True,
        "shell": False }
    # invoke
    with subprocess.Popen(**options) as git:
        # collect the output
        stdout, stderr = git.communicate()
        # get the hash
        commit = stdout.strip()
        # use it as a marker
        table["local"] = commit

    # all done
    return table


def gitTracking(table, match):
    """
    Extract the branch name from a repository that tracks a remote one
    """
    # get the match group dictionary
    info = match.groupdict()
    # set the branch name
    table["local"] = info["local"]
    table["remote"] = info["remote"]
    table["ahead"] = 0 if info["ahead"] is None else int(info["ahead"])
    table["behind"] = 0 if info["behind"] is None else int(info["behind"])
    # all done
    return table


def gitLocal(table, match):
    """
    Extract the branch name from a repository that tracks a remote one
    """
    # get the match group dictionary
    info = match.groupdict()
    # set the branch name
    table["local"] = info["branch"]
    # all done
    return table


def gitMoved(table, match):
    """
    Compute the number of moved files
    """
    # all done
    return table


def gitChanged(table, match):
    """
    Compute the number of changed files
    """
    # get the match group dictionary
    info = match.groupdict()
    # get the code
    code = info["CODE"]

    # if this is an untracked file
    if code in gitUntracked:
        # all done
        return table

    # if this is an ignored file
    if code in gitIgnored:
        # all done
        return table

    # if this is a conflict indicator
    if code in gitConflicts:
        # update the counter
        table["conflicts"] += 1
        # all done
        return table

    # if the code has any info on the index side
    if code[0] != ' ':
        # update the index count
        table["index"] += 1

    # if the code has any info on the worktree side
    if code[1] != ' ':
        table["worktree"] += 1

    # all done
    return table


# the regex
gitParser = re.compile("|".join([
    # brand new repositories without any commits
    r"(?P<no_commits>## No commits yet on (?P<new>.+))$",
    # detached HEAD
    r"(?P<detached>## HEAD \(no branch\))$",
    # repositories at a known branch that tack a remote
    r"(?P<tracking>## " +
        # the local branch name
        r"(?P<local>.+)" +
        # the separator
        r"\.\.\." +
        # the remote info
        r"(" +
            # the remote branch name
            r"(?P<remote>[^\s]+)" +
            # divergence information
            r"(" +
                r" \[(ahead (?P<ahead>\d+))?(, )?(behind (?P<behind>\d+))?\]" +
            r")?" +
        r")?" +
    r")$",
    # no remote repository
    r"(?P<non_tracking>## (?P<branch>.+))$",
    # files that have been copied/renamed
    r"(?P<moved>(?P<code>..) (?P<source>.+) -> (?P<destination>.+))$",
    # files with modifications
    r"(?P<changed>(?P<CODE>..) (?P<filename>.+))$"
]))


# dispatch table
gitDispatcher = {
    "no_commits": gitNoCommits,
    "detached": gitDetached,
    "tracking": gitTracking,
    "non_tracking": gitLocal,
    "moved": gitMoved,
    "changed": gitChanged,
}


# git status code sets
gitUntracked = { "??" }
gitIgnored = { "!!" }
gitConflicts = { "DD", "AU", "UD", "UA", "DU", "AA", "UU" }


# utilities
def finalize(cwd):
    """
    Add the trailing bit
    """
    # render
    yield f"{normal}{cwd}>"
    # all done
    return


def decorateWindow(cwd):
    """
    Decorate an xterm window
    """
    # get the terminal type
    term = os.environ["TERM"]
    # if it's not xterm compatible
    if not term.startswith('xterm'):
        # bail
        return
    # get the instance name
    hostname = "@INSTANCE@"
    # assemble the title
    title = f"isce3@{hostname}:{cwd}"
    # and decorate the window
    yield xtermSetWindowTitle.format(title=title)
    # all done
    return


def getcwd():
    """
    Build a nice representation of the current working directory
    """
    # get the current working directory
    cwd = os.environ["PWD"]
    # and the user's home
    home = os.environ["HOME"]
    # if the current working directory is a subdirectory of the user's home
    if cwd.startswith(home):
        # replace the leading part with a '~'
        cwd = '~' + cwd[len(home):]
    # all done
    return cwd


# color generators
def csi3(code):
    """
    Make a control sequence for the given color code
    """
    # easy wnough
    return f"{rl_hide}{ascii_esc}[{code}m{rl_unhide}"


def csi24(red, green, blue):
    """
    Make a control sequence for the given color
    """
    # easy enough
    return f"{rl_hide}{ascii_esc}[38;2;{red};{green};{blue}m{rl_unhide}"


# constants
# ascii codes
ascii_nul = "\x00"
ascii_soh = "\x01"
ascii_stx = "\x02"
ascii_etx = "\x03"
ascii_eot = "\x04"
ascii_enq = "\x05"
ascii_ack = "\x06"
ascii_bel = "\x07"
ascii_bs  = "\x08"
ascii_tab = "\x09"
ascii_lf  = "\x0a"
ascii_vt  = "\x0b"
ascii_ff  = "\x0c"
ascii_cr  = "\x0d"
ascii_so  = "\x0e"
ascii_si  = "\x0f"
ascii_esc = "\x1b"
ascii_del = "\x7f"

# readline hidden characters escapes
rl_hide = ascii_soh
rl_unhide = ascii_stx

# commands
xtermSetWindowTitle = (
    rl_hide +
    ascii_esc + "]0;{title}" + ascii_bel +
    rl_unhide
)

# colors
# ansi
normal = csi3(code=0)

# 24bit
amber = csi24(red=255, green=191, blue=0)
burlywood = csi24(red=0xde, green=0xb8, blue=0x87)
dark_goldenrod = csi24(red=0xb8, green=0x86, blue=0x0b)
dark_khaki = csi24(red=0xbd, green=0xb7, blue=0x6b)
dark_orange = csi24(red=0xff, green=0x8c, blue=0x00)
dark_sea_green = csi24(red=0x8f, green=0xbc, blue=0x8f)
firebrick = csi24(red=0xb2, green=0x22, blue=0x22)
gray10 = csi24(red=0x19, green=0x19, blue=0x19)
gray30 = csi24(red=0x4c, green=0x4c, blue=0x4c)
gray41 = csi24(red=0x69, green=0x69, blue=0x69)
gray50 = csi24(red=0x80, green=0x80, blue=0x80)
gray66 = csi24(red=169, green=169, blue=169)
gray75 = csi24(red=0xbe, green=0xbe, blue=0xbe)
hot_pink = csi24(red=0xff, green=0x69, blue=0xb4)
indian_red = csi24(red=0xcd, green=0x5c, blue=0x5c)
lavender = csi24(red=192, green=176, blue=224)
light_green = csi24(red=0x90, green=0xee, blue=0x90)
light_steel_blue = csi24(red=0xb0, green=0xc4, blue=0xde)
lime_green = csi24(red=0x32, green=0xcd, blue=0x32)
navajo_white = csi24(red=0xff, green=0xde, blue=0xad)
olive_drab = csi24(red=0x6b, green=0x8e, blue=0x23)
peach_puff = csi24(red=0xff, green=0xda, blue=0xb9)
sage = csi24(red=176, green=208, blue=176)

# functional
# prompt
prompt_normal = gray41
# branch
branch_ahead = prompt_normal
branch_behind = prompt_normal
branch_clean = olive_drab
branch_modified = indian_red
branch_modified_worktree = amber
branch_modified_index = dark_sea_green
branch_conflicts = firebrick


# bootstrap
if __name__ == "__main__":
    # generate the prompt
    status = prompt()
    # and share the status with the shell
    raise SystemExit(status)


# end of file
