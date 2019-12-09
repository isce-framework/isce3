# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import collections, functools, io, os, pwd, stat


# helpers
def _unaryDispatch(f):
    """
    Wrapper for functions that require the string representation of path objects
    """
    # declare and wrap my helper
    @functools.wraps(f)
    def dispatch(self, *args, **kwds):
        # build my rep and forward to the wrapped function
        return f(self, *args, **kwds)
    # return the function to leave behind
    return dispatch


# declaration
class Path(tuple):
    """
    A representation of a path
    """

    # types
    from .exceptions import PathError, SymbolicLinkLoopError

    # string constants
    _CWD = '.'
    _SEP = '/'

    # path constants
    root = None


    # interface
    # methods about me and my parts implemented as properties
    @property
    def parts(self):
        """
        Build an iterator over my components
        """
        # easy enough
        return iter(self)


    @property
    def names(self):
        """
        Build an iterator over the names of my components, skipping the root marker, if present
        """
        # grab my parts
        parts = self.parts
        # if I am an absolute path
        if self.anchor:
            # advance the counter
            next(parts)
        # and return the iterator
        return parts


    @property
    def anchor(self):
        """
        Return the representation of the root of the path, if present
        """
        # if i am empty
        if len(self) == 0:
            # i can't be absolute
            return ''
        # get my first part
        first = self[0]
        # if it is my separator
        if first == self._SEP:
            # i have a root
            return first
        # otherwise, I don't
        return ''


    @property
    def parents(self):
        """
        Generate a sequence of the logical ancestors of the path
        """
        # get my type
        cls = type(self)
        # generate a sequence of lengths so i can build subsequences
        for pos in range(1,len(self)):
            # build a path out of a subsequence that doesn't include the last level
            yield super().__new__(cls, self[:-pos])
        # all done
        return


    @property
    def crumbs(self):
        """
        Generate a sequence of paths from here to the root
        """
        # first me
        yield self
        # and now my ancestors
        yield from self.parents
        # all done
        return


    @property
    def parent(self):
        """
        Build a path that is my logical ancestor

        Note that this is purely a lexical operation and is not guaranteed to yield correct
        results unless this path has been fully resolved
        """
        # the root
        if self == self.root:
            # is it's own parent
            return self
        # for the rest, generate a sequence of length one shorter than me
        return super().__new__(type(self), self[:-1])


    @property
    def name(self):
        """
        Return the final path component
        """
        # when i am empty
        if len(self) == 0:
            # the last component is the empty string
            return ''
        # otherwise, get the last part of the path
        name = self[-1]
        # return it, unless it's the separator, in which case return the empty string
        return name if name != self._SEP else ''


    @property
    def path(self):
        """
        Return a string representing the full path
        """
        # easy enough
        return str(self)


    @property
    def suffix(self):
        """
        The file extension of the final path component
        """
        # grab my name
        name = self.name
        # look for the last '.'
        pos = name.rfind('.')
        # if not there
        if pos == -1:
            # we have nothing
            return ''
        # otherwise
        return name[pos:]


    @property
    def suffixes(self):
        """
        Return an iterable over the extensions in name
        """
        # get my name and skip any leading dots
        name = self.name.lstrip('.')
        # split on the '.', skip the first bit and return the rest with a leading '.'
        return ('.' + suffix for suffix in name.split('.')[1:])


    @property
    def stem(self):
        """
        The final path component without any suffixes
        """
        # grab my name
        name = self.name
        # look for the last '.'
        pos = name.rfind('.')
        # if not there
        if pos == -1:
            # my stem is my name
            return name
        # otherwise, drop the suffix
        return name[:pos]


    @property
    def contents(self):
        """
        Generate a sequence of my contents
        """
        # go through my contents
        for name in os.listdir(self):
            # make a path and hand it to the caller
            yield self / name
        # all done
        return


    # introspection methods
    def as_posix(self):
        """
        Return a POSIX compliant representation
        """
        # i know how to do this
        return str(self)


    def as_uri(self):
        """
        Return a POSIX compliant representation
        """
        # if i am an absolute path
        if self.anchor:
            # splice my representation into a valid 'file:' uri
            return f"file://{self}"
        # otherwise, build an error message
        msg = f"'{self}' is a relative path and can't be expressed as a URI"
        # and complain
        raise ValueError(msg)


    def isAbsolute(self):
        """
        Check whether the path is absolute or not
        """
        # get my last part
        return True if self.anchor else False


    def isReserved(self):
        """
        Check whether the path is reserved or not
        """
        # always false
        return False


    # methods about me and others
    def join(self, *others):
        """
        Combine me with {others} and make a new path
        """
        # get my type
        cls = type(self)
        # that's just what my constructor does
        return cls.__new__(cls, self, *others)


    def relativeTo(self, other):
        """
        Find a {path} such that {other} / {path} == {self}
        """
        # coerce {other} into a path
        other = self.coerce(other)

        # the {path} exists iff {other} is a subsequence of {self}
        if len(other) > len(self):
            # no way
            raise ValueError(f"'{other}' is not a parent of '{self}'")
        # now check the individual levels
        for mine, hers in zip(self, other):
            # if they are not identical
            if mine != hers:
                # build the message
                error = f"'{self}' does not start with '{other}'"
                location = f"'{mine}' doesn't match '{hers}'"
                # and complain
                raise ValueError(f"{error}: {location}")

        # what's left is the answer
        return super().__new__(type(self), self[len(other):])


    def withName(self, name):
        """
        Build a new path with my name replaced by {name}
        """
        # check that the name has no separators in it
        if self._SEP in name:
            # complain
            raise ValueError(f"invalid name '{name}'")
        # replace my name and build a new path
        return super().__new__(type(self), self[:-1] + (name,))


    def withSuffix(self, suffix=None):
        """
        Build a new path with my suffix replaced by {suffix}
        """
        # check that the suffix is valid
        if suffix and (not suffix.startswith('.') or self._SEP in suffix):
            # complain
            raise ValueError(f"invalid suffix '{suffix}'")
        # get my name
        name = self.name
        # get my suffix
        mine = self.suffix
        # and my stem
        stem = self.stem

        # if the suffix is {None}
        if suffix is None:
            # and i have one, remove it; otherwise, clone me
            return self.withName(stem) if mine else self

        # if a suffix were supplied, append it to my stem and build a path
        return self.withName(name=stem+suffix)


    # real path interface
    @classmethod
    def cwd(cls):
        """
        Build a path that points to the current working directory
        """
        # get the directory and turn it into a path
        return cls(os.getcwd())


    @classmethod
    def home(cls, user=''):
        """
        Build a path that points to the {user}'s home directory
        """
        # grab the {pwd} support
        import pwd
        # if we don't have a user
        if not user:
            # assume the current user
            dir = pwd.getpwuid(os.getuid()).pw_dir
        # otherwise
        else:
            # attempt to
            try:
                # index the {passwd} database using the user
                dir = pwd.getpwnam(user).pw_dir
            # if this fails
            except KeyError:
                # most likely cause is
                msg = f"the user '{user}' is not in the password database"
                # so complain
                raise RuntimeError(msg)
        # if we get this far, we have the name of the path; build a path and return it
        return cls(dir)


    def resolve(self):
        """
        Build an equivalent absolute normalized path that is free of symbolic links
        """
        # if I'm empty
        if len(self) == 0:
            # return the current working directory
            return self.cwd()
        # if I am the root
        if self == self.root:
            # I am already resolved
            return self
        # otherwise, get the guy to do his thing
        return self._resolve(resolved={})


    def expanduser(self):
        """
        Build a path with '~' and '~user' patterns expanded
        """
        # grab the user spec, which must be the last path component
        spec = self[:-1]
        # if it doesn't start with the magic character
        if spec[0] != '~':
            # we are done
            return self
        # otherwise, use it to look up the user's home directory; the user name is what follows
        # the marker, and our implementation of {home} interprets a blank user name as the
        # current user
        home = self.home(user=spec[1:])
        # build the new path and return it
        return super().__new__(type(self), home + self[1:])


    # real path introspection
    def exists(self):
        """
        Check whether I exist
        """
        # MGA - 20160121
        # N.B. do not be tempted to return {self} on success and {None} on failure: our
        # representation of the {cwd} is an empty tuple, and that would always fail the
        # existence test. at least until we short circuit {__bool__} to always return
        # {True}. an idea whose merits were not clear at the time of this note

        # attempt to
        try:
            # get my stat record
            self.stat()
        # if i don't exist or i am a broken link
        except (FileNotFoundError, NotADirectoryError):
            # stat is unhappy, so i don't exist
            return False
        # if i got this far, i exist
        return True


    def isBlockDevice(self):
        """
        Check whether I am a block device
        """
        # check with {stat}
        return self.mask(stat.S_IFBLK)


    def isCharacterDevice(self):
        """
        Check whether I am a character device
        """
        # check with {stat}
        return self.mask(stat.S_IFCHR)


    def isDirectory(self):
        """
        Check whether I am a directory
        """
        # check with {stat}
        return self.mask(stat.S_IFDIR)


    def isFile(self):
        """
        Check whether I am a regular file
        """
        # check with {stat}
        return self.mask(stat.S_IFREG)


    def isNamedPipe(self):
        """
        Check whether I am a socket
        """
        # check with {stat}
        return self.mask(stat.S_IFIFO)


    def isSocket(self):
        """
        Check whether I am a socket
        """
        # check with {stat}
        return self.mask(stat.S_IFSOCK)


    def isSymlink(self):
        """
        Check whether I am a symbolic link
        """
        # attempt to
        try:
            # get my stat record
            mode = self.lstat().st_mode
        # if anything goes wrong:
        except (AttributeError, FileNotFoundError, NotADirectoryError):
            # links are probably not supported here, so maybe not...
            return False
        # otherwise, check with my stat record
        return stat.S_ISLNK(mode)


    def mask(self, mask):
        """
        Get my stat record and filter me through {mask}
        """
        # attempt to
        try:
            # get my stat record
            mode = self.stat().st_mode
        # if i don't exist or i am a broken link
        except (FileNotFoundError, NotADirectoryError):
            # probably not...
            return False
        # otherwise, check with {mask}
        return stat.S_IFMT(mode) == mask


    # physical path interface
    # forwarding to standard library functions
    chdir = _unaryDispatch(os.chdir)
    chmod = _unaryDispatch(os.chmod)
    lstat = _unaryDispatch(os.lstat)
    stat = _unaryDispatch(os.stat)
    open = _unaryDispatch(io.open)
    rmdir = _unaryDispatch(os.rmdir)
    unlink = _unaryDispatch(os.unlink)


    # local implementations of the physical path interface
    def mkdir(self, parents=False, exist_ok=False, **kwds):
        """
        Create a directory at my location.

        If {parents} is {True}, create all necessary intermediate levels; if {exist_ok} is
        {True}, do not raise an exception if the directory exists already
        """
        # if we were not asked to build the intermediate levels
        if not parents:
            # attempt to
            try:
                # create the directory
                return os.mkdir(self, **kwds)
            # if the directory exists already
            except FileExistsError:
                # and we care
                if not exist_ok:
                    # complain
                    raise
        # if we are supposed to build the intermediate levels, delegate to the system routine
        return os.makedirs(self, exist_ok=exist_ok, **kwds)


    def touch(self,  mode=0x666, exist_ok=True):
        """
        Create a file at his path
        """
        # all done
        raise NotImplementedError('NYI!')


    # constructors
    @classmethod
    def coerce(cls, *args):
        """
        Build a path out of the given arguments
        """
        # my normal constructor does this
        return cls(*args)


    # meta-methods
    def __new__(cls, *args):
        """
        Build a new path out of strings or other paths
        """
        # if i have only one argument and it is a path
        if len(args) == 1 and isinstance(args[0], cls):
            # return it
            return args[0]
        # otherwise, parse the arguments and chain up to build my instance
        return super().__new__(cls, cls._parse(args))


    def __str__(self):
        """
        Assemble my parts into a string
        """
        # if i am empty
        if len(self) == 0:
            # i represent the current working directory
            return self._CWD
        # grab my separator
        sep = self._SEP
        # set up an iterator over myparts
        i = iter(self)
        # if i am an absolute path
        if self[0] == sep:
            # advance the iterator to skip the root marker
            next(i)
            # but remember it
            marker = sep
        # otherwise
        else:
            # leave no marker in the beginning
            marker = ''
        # build the body out of the remaining parts
        body = sep.join(i)
        # ok, let's put this all together
        return f'{marker}{body}'


    # implement the {os.PathLike} protocol
    __fspath__ = __str__


    def __bool__(self):
        """
        Test for non null values
        """
        # there are no conditions under which I am false since empty tuple means '.'
        return True


    # arithmetic; pure sugar but slower than other methods of assembling paths
    def __truediv__(self, other):
        """
        Syntactic sugar for assembling paths
        """
        # get my type
        cls = type(self)
        # too easy
        return cls.__new__(cls, self, other)


    def __rtruediv__(self, other):
        """
        Syntactic sugar for assembling paths
        """
        # get my type
        cls = type(self)
        # too easy
        return cls.__new__(cls, other, self)


    # implementation details
    @classmethod
    def _parse(cls, args, sep=_SEP, fragments=None):
        """
        Recognize each entry in {args} and distill its contribution to the path under construction
        """
        # initialize the pile
        if fragments is None:
            # empty it out
            fragments = []

        # go through the {args}
        for arg in args:
            # if {arg} is another path
            if isinstance(arg, cls):
                # check whether it is an absolute path
                if len(arg) > 0 and arg[0] == sep:
                    # clear out the current pile
                    fragments.clear()
                # append its part to mine
                fragments.extend(arg)
            # if {arg} is a string
            elif isinstance(arg, str):
                # check whether it starts with my separator
                if arg and arg[0] == sep:
                    # in which case, clear out the current pile
                    fragments.clear()
                    # and start a new absolute path
                    fragments.append(sep)
                # split on separator and remove blanks caused by multiple consecutive separators
                fragments.extend(filter(None, arg.split(sep)))
            # more general iterables
            elif isinstance(arg, collections.abc.Iterable):
                # recurse with their contents
                cls._parse(args=arg, sep=sep, fragments=fragments)
            # anything else
            else:
                # is an error
                msg = f"can't parse '{arg}', of type {type(arg)}"
                # so complain
                raise TypeError(msg)

        # all done
        return fragments


    def _resolve(self, base=None, resolved=None):
        """
        Workhorse for path resolution
        """
        # what's left to resolve
        workload = self.parts
        # if i am an absolute path
        if self.anchor:
            # set my starting point
            base = self.root
            # skip the leasing root marker
            next(workload)
        # if i am a relative path
        else:
            # my starting point is the current working directory, which is guaranteed to be
            # free of symbolic links
            base = self.cwd() if base is None else base

        # at this point, {base} is known to be a fully resolved path
        # go through my parts
        for part in workload:
            # empty or parts that are '.'
            if not part or part=='.':
                # are skipped
                continue
            # parent directory markers
            if part == '..':
                # back me up by one level
                base = base.parent
                # and carry on
                continue
            # splice the part onto base
            newpath = base / part
            # check
            try:
                # whether we have been here before
                resolution = resolved[newpath]
            # if not
            except KeyError:
                # carry on
                pass
            # if yes
            else:
                # if {base} has a null resolution
                if resolution is None:
                    # we probably got a loop, so complain
                    raise self.SymbolicLinkLoopError(path=self, loop=newpath)
                # otherwise, replace {base} with its resolution
                base = resolution
                # and carry on
                continue

            # now we need to know whether what we have so far is a symbolic link
            if newpath.isSymlink():
                # add it to the pile, but mark it unresolved
                resolved[newpath] = None
                # find out what it points to
                link = type(self)(os.readlink(str(newpath)))
                # resolve it in my context
                base = link._resolve(resolved=resolved, base=base)
                # remember this
                resolved[newpath] = base
            # if not
            else:
                # save it and carry on
                base = newpath

        return base


# patches
Path.root = Path(Path._SEP)


# end of file
