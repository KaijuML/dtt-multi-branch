"""
Where we store everything which is in common between our scripts
"""

import more_itertools
import overrides
import json
import os


class FileIterable:
    def __init__(self, iterable, filename=None):
        self._iterable = more_itertools.seekable(iterable)
        self._filename = filename
        
        self._ptr = 0  # pointer to the next item
        self._len = None
        
    @classmethod
    def from_filename(cls, path, func=None, fmt='txt'):
        if not path.endswith(fmt):
            print(f'\nWARNING: path is {path} but format is {fmt}')
        
        return cls(cls.read_file(path, func, fmt), path)
    
    @staticmethod
    def read_file(path, func=None, fmt='txt'):
        """
        ARGS:
            path (str): hopefully a valid path. 
                        Each line contains a sentence
                        Tokens are separated by a space ' '
            func (NoneType): Should be None in this simple case
            fmt (str): How we procede with each line [txt | jl]
        """
        if fmt == 'txt':
            def _read_line(line): return line.strip().split()
        elif fmt == 'jl':
            def _read_line(line): return json.loads(line)
        else:
            raise ValueError(f'Unkown file format {fmt}')
        
        assert func is None
        with open(path, mode="r", encoding="utf8") as f:
            for line in f:
                yield _read_line(line)
    
    def __getitem__(self, n):
        self._iterable.seek(n)
        self._ptr = n
        try:
            return next(self)
        except StopIteration:
            raise IndexError(f'{n} is outside this generator')
            
        return ret
    
    def __next__(self):
        self._ptr += 1
        return next(self._iterable)
    
    def _length_from_ilen(self):
        ptr = self._ptr  # remember the position of the pointer
        length = more_itertools.ilen(self._iterable)  # count remaining items
        _ = self[ptr - 1]  # set back the pointer
        return length + ptr
    
    def __len__(self):
        if self._len is None:
            if self._filename is not None:
                self._len = int(os.popen(f'wc -l < {self._filename}').read())
            else:
                self._len = self._length_from_ilen()
        return self._len


class TaggedFileIterable(FileIterable):
    
    @staticmethod
    @overrides.overrides
    def read_file(path, func=None, fmt='txt'):
        """
        Accumulate lines until empty line, then yield and do it again
        ARGS:
            path (str): hopefully a valid path. 
                        Each line contains a token and its tag(s)
                        Sentences are separated by an empty line
            func (func): will be applied to each line. 
                         Defaults to identity func `lambda x: x`
                         Use this to convert tags to correct type
            fmt (str): Should always be 'txt'
        """
        if func is None:
            def func(*args):
                return args
        assert fmt == 'txt', f'Unkown file format {fmt}'
        
        sentence = list()
        with open(path, mode='r', encoding='utf8') as f:
            for line in f:
                if line.strip():
                    sentence.append(func(*line.strip().split()))
                else:
                    yield sentence
                    sentence = list()
        if sentence: 
            yield sentence
            
    @overrides.overrides
    def __len__(self):
        if self._len is None:
            self._len = self._length_from_ilen()
        return self._len