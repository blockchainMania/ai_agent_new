"""
문서 전처리 모듈

특허 문서의 전처리를 담당합니다.
- 불용어 제거
- 특허 형식어 제거 (특허번호, 도면번호, 청구항 번호 등)
- 텍스트 정규화
"""

import re
import unicodedata
from typing import List
import nltk
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """텍스트 전처리 클래스"""

    def __init__(self, custom_stopwords: List[str] = None):
        """
        Args:
            custom_stopwords: 사용자 정의 불용어 리스트
        """
        # NLTK 데이터 다운로드 (최초 1회)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt_tab', quiet=True)

        # 기본 불용어 설정
        self.domain_stopwords = {
            "refrigerator", "body", "box", "lid", "plate", "rack", "space",
            "assembly", "assemblies", "housing", "housings", "case", "cases",
            "compartment", "compartments", "chamber", "chambers", "container",
            "containers", "shelf", "shelves", "drawer", "drawers", "bin", "bins",
            "basket", "baskets", "tray", "trays", "partition", "partitions",
            "panel", "panels", "cover", "covers", "frame", "frames", "bracket",
            "brackets", "guide", "guides", "groove", "grooves", "opening",
            "openings", "closure", "closures", "hinge", "hinges", "handle",
            "handles", "support", "supports", "member", "element", "component",
            "structure", "mechanism", "unit", "modules", "apparatus", "device",
            "storage", "control", "controller", "controllers", "position",
            "evaporator", "evaporators", "condenser", "condensers", "compressor",
            "compressors", "fan", "fans", "blower", "blowers", "duct", "ducts",
            "nozzle", "nozzles", "tube", "tubes", "pipe", "pipes", "conduit",
            "conduits", "manifold", "manifolds", "aperture", "apertures", "edge",
            "edges", "corner", "corners", "sidewall", "sidewalls", "composition",
            "agent", "parts", "weight", "mass", "type", "sample"
        }

        if custom_stopwords:
            self.domain_stopwords.update(custom_stopwords)

        # 정규표현식 패턴 정의
        self._compile_patterns()

    def _compile_patterns(self):
        """정규표현식 패턴을 미리 컴파일"""
        self.patterns = {
            'patent_num': re.compile(r'\b(?:us|wo|kr|ep|jp|cn)\s*\d+[a-z]*\d*\b', re.I),
            'claim_en': re.compile(r'\b(claim|claims)\s*\d+\b', re.I),
            'claim_ko': re.compile(r'제\s*\d+\s*항'),
            'claim_ko_inline': re.compile(
                r'(?:청구항\s*\d+(?:\s*(?:[,~–—\-]|내지|및|또는)\s*\d+)*)'
                r'|(?:제\s*\d+\s*항(?:\s*(?:[,~–—\-]|내지|및|또는)\s*제?\s*\d+\s*항)*)',
                re.I
            ),
            'fig': re.compile(r'(?:fig\.?|figure|도)\s*\d+[a-z]*', re.I),
            'table': re.compile(r'table\s*\d+', re.I),
            'example_hdr': re.compile(r'\b(ex|example|examples)\s*\d+[a-z]*\b', re.I),
            'section_hdr_ko': re.compile(
                r'(발명의\s*분야|기술\s*분야|배경\s*기술|발명의\s*요약|과제|해결수단|효과|'
                r'도면의\s*간단한\s*설명|발명의\s*상세한\s*설명|목적)\b'
            ),
            'feature_phrase': re.compile(r'것을\s*특징으로\s*하는(?:\s*것)?', re.I),
            'ref_label': re.compile(r'\b(?:[a-z]{1,3}\d{1,4}|\d{1,4}[a-z]{1,3})\b', re.I),
            'num_unit': re.compile(
                r'\b-?\d+(?:\.\d+)?\s*(?:nm|mm|cm|μm|um|kg|g|mg|l|ml|rpm|hz|khz|mhz|ghz|'
                r'db|v|kv|ma|a|w|kw|wh|ppm|bar|psi|kpa|pa|°c|℃|°f|℉|%)\b',
                re.I
            ),
            'wt_vol_mol': re.compile(r'\b(?:wt|vol|mol)\.?\s*%\b', re.I),
            'range': re.compile(r'\b-?\d+(?:\.\d+)?\s*(?:~|–|—|-|to)\s*-?\d+(?:\.\d+)?\s*(?:%|ppm)?\b', re.I),
            'symbols': re.compile(r'[±≤≥≦≧≈∼~×™®©•·…§†º‡°‰℃℉µΩΔαβγμΩ※]'),
            'paren_num': re.compile(r'\(\s*\d{1,3}[a-z]?\s*\)'),
            'digit': re.compile(r'\d+'),
            'inline_formula': re.compile(
                r'(?:\|[^\|>]+>\s*(?:=|\+|-|\*|/)?\s*)+'
                r'|(?:[α-ωΑ-Ω]\s*(?:=|\+|-|\*|/)?\s*)+'
                r'|(?:[A-Za-z]\s*=\s*[^,;\n]+)',
                re.UNICODE
            )
        }

    def normalize_text(self, text: str) -> str:
        """
        텍스트 정규화

        Args:
            text: 원본 텍스트

        Returns:
            정규화된 텍스트 (소문자, 특수문자 제거)
        """
        if not isinstance(text, str):
            return ""

        # 유니코드 정규화
        text = unicodedata.normalize("NFKC", text).lower()

        # 하이픈/슬래시 제외 특수문자 제거
        text = re.sub(r"[^\w\s\-/]", " ", text)

        return text

    def remove_patent_elements(self, text: str) -> str:
        """
        특허 문서 고유의 형식어 제거

        Args:
            text: 입력 텍스트

        Returns:
            특허 형식어가 제거된 텍스트
        """
        # 특허번호, 도면, 청구항 등 제거
        for pattern_name in ['patent_num', 'claim_en', 'claim_ko', 'claim_ko_inline',
                            'fig', 'table', 'example_hdr', 'section_hdr_ko',
                            'feature_phrase']:
            text = self.patterns[pattern_name].sub(" ", text)

        # 참조 라벨, 숫자-단위, 범위 등 제거
        for pattern_name in ['ref_label', 'paren_num', 'range', 'wt_vol_mol',
                            'num_unit', 'symbols', 'digit', 'inline_formula']:
            text = self.patterns[pattern_name].sub(" ", text)

        # 공백 정리
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize_and_filter(self, text: str) -> List[str]:
        """
        토큰화 및 불용어 필터링

        Args:
            text: 입력 텍스트

        Returns:
            필터링된 토큰 리스트
        """
        # 토큰화
        tokens = word_tokenize(text)

        # 불용어, 숫자, 1글자 단어 제거
        filtered_tokens = [
            token for token in tokens
            if (token not in self.domain_stopwords and
                len(token) > 1 and
                not token.isdigit())
        ]

        return filtered_tokens

    def preprocess(self, texts: List[str]) -> List[str]:
        """
        전체 전처리 파이프라인 실행

        Args:
            texts: 문서 리스트

        Returns:
            전처리된 문서 리스트
        """
        processed_texts = []

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                processed_texts.append("")
                continue

            # 1. 정규화
            text = self.normalize_text(text)

            # 2. 특허 형식어 제거
            text = self.remove_patent_elements(text)

            # 3. 토큰화 및 불용어 제거
            tokens = self.tokenize_and_filter(text)

            # 4. 결과 문자열로 변환
            processed_texts.append(" ".join(tokens))

        return processed_texts


def get_default_stopwords() -> List[str]:
    """
    기본 불용어 리스트 반환

    Returns:
        불용어 리스트
    """
    return [
        # 여기에 도메인 특화 불용어를 추가할 수 있습니다
    ]
