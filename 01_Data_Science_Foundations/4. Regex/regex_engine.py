"""
Regex Result Adapter & Memory-Efficient Streamer Module

Provides:
1. RegexMatch: Standardized dataclass representing any single match.
2. RegexResult: Unified adapter wrapping results across search, match, findall, and finditer.
3. StreamRegexParser: Generator-based streaming parser for large datasets/files.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Pattern,
    Tuple,
    Union,
)


@dataclass(frozen=True)
class RegexMatch:
    """Standardized representation of a single regular expression match."""
    raw_match: str
    span: Tuple[int, int]
    groups: Tuple[Optional[str], ...] = field(default_factory=tuple)
    named_groups: Dict[str, Optional[str]] = field(default_factory=dict)

    @classmethod
    def from_re_match(cls, m: re.Match) -> RegexMatch:
        """Construct from a native re.Match object."""
        return cls(
            raw_match=m.group(0),
            span=m.span(),
            groups=m.groups(),
            named_groups=m.groupdict(),
        )

    @classmethod
    def from_raw_string(cls, val: str, span: Tuple[int, int] = (0, 0)) -> RegexMatch:
        """Construct from raw string fallback (e.g. from findall)."""
        return cls(
            raw_match=val,
            span=span,
            groups=(val,),
            named_groups={},
        )

    def get(self, key: Union[int, str], default: Optional[str] = None) -> Optional[str]:
        """Safely retrieve a group value by index or group name."""
        if isinstance(key, str):
            return self.named_groups.get(key, default)
        elif isinstance(key, int):
            if 0 == key:
                return self.raw_match
            if 1 <= key <= len(self.groups):
                val = self.groups[key - 1]
                return val if val is not None else default
        return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert match metadata into a serializable dictionary."""
        return {
            "match": self.raw_match,
            "span": self.span,
            "groups": self.groups,
            "named_groups": self.named_groups,
        }


class RegexResult(Iterable[RegexMatch]):
    """
    Unified Adapter for regular expression query results.
    Standardizes output across search, match, fullmatch, findall, and finditer.
    """

    def __init__(self, matches: List[RegexMatch], source_pattern: str = "") -> None:
        self._matches: List[RegexMatch] = matches
        self.source_pattern: str = source_pattern

    @classmethod
    def from_search(cls, match: Optional[re.Match], pattern: str = "") -> RegexResult:
        """Create an adapter result from re.search, re.match, or re.fullmatch."""
        if match is None:
            return cls([], source_pattern=pattern)
        return cls([RegexMatch.from_re_match(match)], source_pattern=pattern)

    @classmethod
    def from_finditer(
        cls, iterator: Iterator[re.Match], pattern: str = ""
    ) -> RegexResult:
        """Create an adapter result from re.finditer."""
        matches = [RegexMatch.from_re_match(m) for m in iterator]
        return cls(matches, source_pattern=pattern)

    @classmethod
    def from_findall(
        cls, results: Union[List[str], List[Tuple[str, ...]]], pattern: str = ""
    ) -> RegexResult:
        """Create an adapter result from re.findall."""
        matches: List[RegexMatch] = []
        for item in results:
            if isinstance(item, str):
                matches.append(RegexMatch.from_raw_string(item))
            elif isinstance(item, tuple):
                raw = "".join(item)
                matches.append(
                    RegexMatch(
                        raw_match=raw,
                        span=(0, 0),
                        groups=item,
                        named_groups={},
                    )
                )
        return cls(matches, source_pattern=pattern)

    @property
    def has_match(self) -> bool:
        """Check if at least one match was found."""
        return len(self._matches) > 0

    @property
    def first(self) -> Optional[RegexMatch]:
        """Get the first match if available."""
        return self._matches[0] if self._matches else None

    @property
    def all(self) -> List[RegexMatch]:
        """Get all parsed matches."""
        return self._matches

    @property
    def count(self) -> int:
        """Return total matches found."""
        return len(self._matches)

    def named_dicts(self) -> List[Dict[str, Optional[str]]]:
        """Return list of groupdict dictionaries for all matched items."""
        return [m.named_groups for m in self._matches if m.named_groups]

    def __iter__(self) -> Iterator[RegexMatch]:
        return iter(self._matches)

    def __len__(self) -> int:
        return len(self._matches)

    def __repr__(self) -> str:
        return f"<RegexResult matches={len(self._matches)} pattern='{self.source_pattern}'>"


class StreamRegexParser:
    """
    Memory-efficient generator-based streaming parser.
    Processes file streams or line iterators chunk-by-chunk with constant O(1) memory.
    """

    def __init__(self, pattern: Union[str, Pattern[str]], flags: int = 0) -> None:
        if isinstance(pattern, str):
            self.compiled: Pattern[str] = re.compile(pattern, flags)
        else:
            self.compiled = pattern

    def stream_lines(
        self, lines: Iterable[str]
    ) -> Generator[RegexMatch, None, None]:
        """
        Stream matches line-by-line using re.finditer to avoid accumulating memory.
        """
        for line in lines:
            for m in self.compiled.finditer(line):
                yield RegexMatch.from_re_match(m)

    def stream_file(
        self, file_path: str, chunk_size: int = 64 * 1024
    ) -> Generator[RegexMatch, None, None]:
        """
        Stream matches across a large file buffer without reading the entire file at once.
        Preserves boundary overlaps between chunks.
        """
        overlap_size = 512
        remainder = ""

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    if remainder:
                        for m in self.compiled.finditer(remainder):
                            yield RegexMatch.from_re_match(m)
                    break

                text_block = remainder + chunk
                last_match_end = 0

                for m in self.compiled.finditer(text_block):
                    if m.end() <= len(text_block) - overlap_size:
                        yield RegexMatch.from_re_match(m)
                        last_match_end = m.end()

                if last_match_end > 0:
                    remainder = text_block[last_match_end:]
                else:
                    remainder = text_block[-overlap_size:]


# ==========================================
# Demonstration & Usage Example
# ==========================================
if __name__ == "__main__":
    print("=== 1. Unified RegexResult Adapter Demo ===")
    pattern = r"(?:(?P<area>\d{3})-)?(?P<local>\d{3})"
    
    # 1. Single search match
    res_search = RegexResult.from_search(re.search(pattern, "123-456"), pattern)
    print("Search Result:", res_search)
    if res_search.first:
        print("  Matched Text :", res_search.first.raw_match)
        print("  Span         :", res_search.first.span)
        print("  Area Code    :", res_search.first.get("area"))
        print("  Local Code   :", res_search.first.get("local"))

    # 2. Multi-match finditer adapter
    text = "Call 123-456 or 789 or 555-888"
    res_iter = RegexResult.from_finditer(re.finditer(pattern, text), pattern)
    print("\nFinditer Result Count:", res_iter.count)
    for idx, match in enumerate(res_iter, start=1):
        print(f"  Match {idx}: {match.raw_match} -> {match.named_groups}")

    print("\n=== 2. Memory-Efficient Generator Streaming Demo ===")
    sample_log_lines = [
        '2026-08-27 21:30:01 [INFO] (auth) user_id=101 logged in',
        '2026-08-27 21:30:02 [ERROR] (db) connection failed to host 10.0.0.4',
        '2026-08-27 21:30:03 [WARNING] (cache) key expired user_101',
    ]

    log_pattern = (
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
        r"\[(?P<level>\w+)\]\s+"
        r"\((?P<service>\w+)\)\s+"
        r"(?P<msg>.*)"
    )

    streamer = StreamRegexParser(log_pattern)
    print("Streaming log generator output:")
    for record in streamer.stream_lines(sample_log_lines):
        print(f"  [{record.get('level')}] ({record.get('service')}): {record.get('msg')}")
